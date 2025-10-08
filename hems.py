#!/usr/bin/env python3
# hems.py
import sqlite3
import argparse
import datetime as dt
from dataclasses import dataclass
from typing import Dict, Tuple

DB_PATH = "hems.db"

# ---------- Domain data ----------
@dataclass
class Measurements:
    base_load_kw: float            # non-flex household load (lights, appliances)
    solar_kw: float                # current PV production (kW)
    house_temp_c: float            # current indoor temperature (°C)
    battery_energy_kwh: float      # current battery energy (kWh)
    people_presence_pct: float     # % presence proxy (0-100)
    desired_charger_kw: float      # user's desired EV charging power (kW)
    needs_heating: bool            # whether there is a heating demand right now
    step_hours: float = 1.0        # control interval hours (assume 1h for kWh↔kW)

@dataclass
class Setpoints:
    heatpump_kw: float
    charger_kw: float
    battery_charge_kw: float   # positive = charging from grid/PV; negative = discharge
    net_grid_kw: float         # resulting grid import (>0) or export (<0)

# ---------- DB helpers ----------
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ruletype VARCHAR(25) NOT NULL,
    importance INTEGER NOT NULL,
    description VARCHAR(255) NOT NULL,
    value INTEGER NOT NULL,
    unit VARCHAR(10) NOT NULL,
    startdate VARCHAR(25),
    enddate VARCHAR(25),
    starttime VARCHAR(25),
    endtime VARCHAR(25),
    status INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS metarules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rulename VARCHAR(25) NOT NULL,
    ruletype VARCHAR(25) NOT NULL,
    importance INTEGER NOT NULL,
    description VARCHAR(255) NOT NULL,
    unit VARCHAR(10) NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    decided_at TEXT NOT NULL,
    base_load_kw REAL NOT NULL,
    solar_kw REAL NOT NULL,
    house_temp_c REAL NOT NULL,
    battery_energy_kwh REAL NOT NULL,
    people_presence_pct REAL NOT NULL,
    desired_charger_kw REAL NOT NULL,
    needs_heating INTEGER NOT NULL,
    step_hours REAL NOT NULL,
    heatpump_kw REAL NOT NULL,
    charger_kw REAL NOT NULL,
    battery_charge_kw REAL NOT NULL,
    net_grid_kw REAL NOT NULL
);
"""

SEED_RULES = [
    ("grid:connection",0,"max household energy supply",17,"kW"),
    ("production:solar:min",2,"min solar production for smart stuff",2,"kW"),
    ("state:battery:min",2,"min state of charge",1,"kWh"),
    ("state:battery:max",2,"max state of charge",18,"kWh"),
    ("consumption:battery:max",2,"max power",5,"kW"),
    ("consumption:heatpump:max",1,"max power consumption",7,"kW"),
    ("consumption:heatpump:min",1,"min steady state power consumption",7,"kW"),
    ("consumption:charger:max",2,"max power consumption",11,"kW"),
    ("consumption:charger:min",2,"min power consumption",7,"kW"),
    ("state:house:min",1,"min temp at home",15,"C"),
    ("state:house:max",1,"min temp at home",20,"C"),
    ("people:presense:min",2,"movement at home",5,"%"),
]

SEED_METARULES = [
    ("sufficient power","grid",0,"Verify rest power available","kW"),
    ("charge battery 1","battery",0,"Charge battery between min and max energy","kWh"),
    ("charge battery 2","battery",0,"Battery between min and max power","kW"),
    ("can charge battery","grid|battery",0,"Safe power","kW"),
]

def init_db(conn: sqlite3.Connection):
    conn.executescript(SCHEMA_SQL)
    # seed only if empty
    cur = conn.execute("SELECT COUNT(*) FROM rules")
    if cur.fetchone()[0] == 0:
        conn.executemany(
            "INSERT INTO rules (ruletype,importance,description,value,unit) VALUES (?,?,?,?,?)",
            SEED_RULES,
        )
    cur = conn.execute("SELECT COUNT(*) FROM metarules")
    if cur.fetchone()[0] == 0:
        conn.executemany(
            "INSERT INTO metarules (rulename,ruletype,importance,description,unit) VALUES (?,?,?,?,?)",
            SEED_METARULES,
        )
    conn.commit()

def load_rules(conn: sqlite3.Connection) -> Dict[str, Tuple[int, int, str]]:
    """
    Returns dict ruletype -> (value, importance, unit)
    """
    out: Dict[str, Tuple[int,int,str]] = {}
    for row in conn.execute("SELECT ruletype, value, importance, unit FROM rules"):
        out[row[0]] = (row[1], row[2], row[3])
    return out

# ---------- Policy / Control ----------
class HEMSController:
    def __init__(self, rules: Dict[str, Tuple[int,int,str]]):
        self.rules = rules

        # convenience getters (with sane defaults if missing)
        self.grid_limit_kw = float(self._get("grid:connection", 17))
        self.solar_min_kw = float(self._get("production:solar:min", 0))
        self.house_min_c = float(self._get("state:house:min", 15))
        self.house_max_c = float(self._get("state:house:max", 20))

        self.batt_min_kwh = float(self._get("state:battery:min", 0))
        self.batt_max_kwh = float(self._get("state:battery:max", 18))
        self.batt_pmax_kw = float(self._get("consumption:battery:max", 5))  # charge OR discharge cap

        self.hp_min_kw = float(self._get("consumption:heatpump:min", 0))
        self.hp_max_kw = float(self._get("consumption:heatpump:max", 7))

        self.chg_min_kw = float(self._get("consumption:charger:min", 0))
        self.chg_max_kw = float(self._get("consumption:charger:max", 11))

    def _get(self, key: str, default: float) -> float:
        return self.rules.get(key, (default, 0, ""))[0]

    def decide(self, m: Measurements) -> Setpoints:
        """
        Greedy, rule-safe allocator:
        1) Satisfy mandatory heating if house below min and there is demand.
        2) Keep grid <= limit; first try to use PV, then discharge battery if needed.
        3) If PV surplus/headroom: charge battery (within SoC & P limits).
        4) Allocate EV charger from remaining headroom (respect min/max).
        """

        # Helper lambdas
        def energy_room_to_max_kwh() -> float:
            return max(0.0, self.batt_max_kwh - m.battery_energy_kwh)

        def energy_above_min_kwh() -> float:
            return max(0.0, m.battery_energy_kwh - self.batt_min_kwh)

        # Initial setpoints
        hp_kw = 0.0
        ch_kw = 0.0
        batt_kw = 0.0  # +charge / -discharge

        # Step 0: base net grid with *no* controllables
        # grid import = base + hp + charger + charge - discharge - solar
        # Here hp=0, charger=0, batt=0 initially:
        net_grid_kw = m.base_load_kw - m.solar_kw

        # Step 1: Heat pump logic (safety & comfort)
        if m.needs_heating and m.house_temp_c < self.house_min_c:
            hp_kw = min(self.hp_max_kw, max(self.hp_min_kw, self.hp_min_kw))
        else:
            hp_kw = 0.0
        net_grid_kw += hp_kw  # add heat pump draw

        # Step 2: Ensure we don't exceed grid cap by discharging battery if necessary
        if net_grid_kw > self.grid_limit_kw:
            deficit = net_grid_kw - self.grid_limit_kw
            possible_discharge = min(self.batt_pmax_kw, energy_above_min_kwh()/m.step_hours)
            d = min(deficit, possible_discharge)
            batt_kw -= d
            net_grid_kw -= d

        # Step 3: With PV surplus/headroom, consider battery charging
        # Define headroom relative to the grid cap (how much extra we can add before hitting limit)
        headroom = self.grid_limit_kw - net_grid_kw
        can_charge_batt = (m.solar_kw >= self.solar_min_kw) and (energy_room_to_max_kwh() > 0.0)
        if headroom > 0 and can_charge_batt:
            possible_charge = min(self.batt_pmax_kw, energy_room_to_max_kwh()/m.step_hours, headroom)
            batt_kw += possible_charge
            net_grid_kw += possible_charge  # charging increases grid import unless PV already covering
            headroom -= possible_charge

        # Step 4: Allocate EV charging within remaining headroom
        # If we can't at least meet the configured min, set to 0 (avoid chatter)
        desired = max(0.0, min(m.desired_charger_kw, self.chg_max_kw))
        alloc = min(desired, headroom)
        if alloc >= min(self.chg_min_kw, desired):
            ch_kw = alloc
            net_grid_kw += ch_kw
            headroom -= ch_kw
        else:
            ch_kw = 0.0

        # Final safety clip: If somehow above grid cap (numeric round), clip with battery if possible
        if net_grid_kw > self.grid_limit_kw + 1e-6:
            deficit = net_grid_kw - self.grid_limit_kw
            possible_discharge = min(self.batt_pmax_kw + max(0.0, -batt_kw), energy_above_min_kwh()/m.step_hours)
            d = min(deficit, possible_discharge)
            batt_kw -= d
            net_grid_kw -= d

        return Setpoints(
            heatpump_kw=round(hp_kw, 3),
            charger_kw=round(ch_kw, 3),
            battery_charge_kw=round(batt_kw, 3),
            net_grid_kw=round(net_grid_kw, 3),
        )

# ---------- Logging ----------
def log_decision(conn: sqlite3.Connection, m: Measurements, sp: Setpoints):
    conn.execute(
        """INSERT INTO decisions (
            decided_at, base_load_kw, solar_kw, house_temp_c, battery_energy_kwh, people_presence_pct,
            desired_charger_kw, needs_heating, step_hours, heatpump_kw, charger_kw, battery_charge_kw, net_grid_kw
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            dt.datetime.now(dt.UTC).isoformat(timespec="seconds")+"Z",
            m.base_load_kw, m.solar_kw, m.house_temp_c, m.battery_energy_kwh, m.people_presence_pct,
            m.desired_charger_kw, int(m.needs_heating), m.step_hours,
            sp.heatpump_kw, sp.charger_kw, sp.battery_charge_kw, sp.net_grid_kw
        ),
    )
    conn.commit()

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="HEMS policy engine using SQLite rules/metarules.")
    p.add_argument("--db", default=DB_PATH, help="SQLite path (default hems.db)")
    p.add_argument("--base-load", type=float, help="Base household load (kW)")
    p.add_argument("--solar", type=float, help="Current PV production (kW)")
    p.add_argument("--temp", type=float, help="Indoor temperature (°C)")
    p.add_argument("--battery", type=float, help="Battery energy (kWh)")
    p.add_argument("--presence", type=float, help="Presence %% (0-100)")
    p.add_argument("--ev", type=float, help="Desired EV charging power (kW)")
    p.add_argument("--needs-heating", action="store_true", help="Flag: heating demand present")
    p.add_argument("--step-hours", type=float, default=1.0, help="Control interval in hours (default 1h)")
    p.add_argument("--print-rules", action="store_true", help="Print loaded rule values and exit")
    return p.parse_args()

def interactive_measurements() -> Measurements:
    def ask(prompt, cast=float, default=None):
        s = input(f"{prompt}{' ['+str(default)+']' if default is not None else ''}: ").strip()
        if s == "" and default is not None:
            return default
        return cast(s)
    base = ask("Base load kW", float, 1.5)
    solar = ask("PV kW", float, 3.0)
    temp = ask("Indoor temp °C", float, 18.0)
    batt = ask("Battery energy kWh", float, 10.0)
    pres = ask("Presence %", float, 80.0)
    ev   = ask("Desired EV kW", float, 7.0)
    need = ask("Needs heating? (0/1)", int, 1) == 1
    step = ask("Step hours", float, 1.0)
    return Measurements(base, solar, temp, batt, pres, ev, need, step)

def main():
    args = parse_args()
    conn = sqlite3.connect(args.db)
    init_db(conn)
    rules = load_rules(conn)

    if args.print_rules:
        print("Loaded rule values:")
        for k,(v,imp,u) in sorted(rules.items()):
            print(f" - {k:<26} = {v} {u} (importance {imp})")
        return

    # build measurements (CLI or interactive)
    if all(getattr(args, x) is not None for x in ["base_load","solar","temp","battery","presence","ev"]):
        m = Measurements(
            base_load_kw=args.base_load,
            solar_kw=args.solar,
            house_temp_c=args.temp,
            battery_energy_kwh=args.battery,
            people_presence_pct=args.presence,
            desired_charger_kw=args.ev,
            needs_heating=args.needs_heating,
            step_hours=args.step_hours,
        )
    else:
        print("Interactive mode (use --help to pass inputs via CLI):")
        m = interactive_measurements()

    controller = HEMSController(rules)
    sp = controller.decide(m)
    log_decision(conn, m, sp)

    # Pretty print result
    print("\n=== HEMS Decision ===")
    print(f"Grid connection limit:   {controller.grid_limit_kw:.2f} kW")
    print(f"Base load:               {m.base_load_kw:.2f} kW")
    print(f"PV production:           {m.solar_kw:.2f} kW")
    print(f"Heat pump setpoint:      {sp.heatpump_kw:.2f} kW")
    if sp.battery_charge_kw >= 0:
        print(f"Battery charge power:    {sp.battery_charge_kw:.2f} kW")
    else:
        print(f"Battery discharge power: {-sp.battery_charge_kw:.2f} kW")
    print(f"EV charger setpoint:     {sp.charger_kw:.2f} kW")
    print(f"Resulting net grid:      {sp.net_grid_kw:.2f} kW {'import' if sp.net_grid_kw>=0 else 'export'}")
    print("\nLogged to decisions table.")
    conn.close()

if __name__ == "__main__":
    main()
