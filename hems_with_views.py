#!/usr/bin/env python3
# hems_with_views.py (with headers when showing views)
import sqlite3
import argparse
import datetime as dt
from dataclasses import dataclass
from typing import Dict, Tuple

DB_PATH = "hems.db"

# ---------- Domain data ----------
@dataclass
class Measurements:
    base_load_kw: float
    solar_kw: float
    house_temp_c: float
    battery_energy_kwh: float
    people_presence_pct: float
    desired_charger_kw: float
    needs_heating: bool
    step_hours: float = 1.0

@dataclass
class Setpoints:
    heatpump_kw: float
    charger_kw: float
    battery_charge_kw: float
    net_grid_kw: float

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
    if conn.execute("SELECT COUNT(*) FROM rules").fetchone()[0] == 0:
        conn.executemany(
            "INSERT INTO rules (ruletype,importance,description,value,unit) VALUES (?,?,?,?,?)",
            SEED_RULES)
    if conn.execute("SELECT COUNT(*) FROM metarules").fetchone()[0] == 0:
        conn.executemany(
            "INSERT INTO metarules (rulename,ruletype,importance,description,unit) VALUES (?,?,?,?,?)",
            SEED_METARULES)
    conn.commit()

def create_views(conn: sqlite3.Connection):
    conn.executescript("""
    CREATE VIEW IF NOT EXISTS decision_with_rules AS
    SELECT 
        d.*,
        r_grid.value         AS grid_limit_kw,
        r_solar_min.value    AS solar_min_kw,
        r_bmin.value         AS battery_min_kwh,
        r_bmax.value         AS battery_max_kwh,
        r_bpow.value         AS battery_pmax_kw,
        r_hp_min.value       AS heatpump_min_kw,
        r_hp_max.value       AS heatpump_max_kw,
        r_ch_min.value       AS charger_min_kw,
        r_ch_max.value       AS charger_max_kw,
        r_tmin.value         AS house_min_c,
        r_tmax.value         AS house_max_c
    FROM decisions d
    LEFT JOIN rules AS r_grid      ON r_grid.ruletype      = 'grid:connection'
    LEFT JOIN rules AS r_solar_min ON r_solar_min.ruletype = 'production:solar:min'
    LEFT JOIN rules AS r_bmin      ON r_bmin.ruletype      = 'state:battery:min'
    LEFT JOIN rules AS r_bmax      ON r_bmax.ruletype      = 'state:battery:max'
    LEFT JOIN rules AS r_bpow      ON r_bpow.ruletype      = 'consumption:battery:max'
    LEFT JOIN rules AS r_hp_min    ON r_hp_min.ruletype    = 'consumption:heatpump:min'
    LEFT JOIN rules AS r_hp_max    ON r_hp_max.ruletype    = 'consumption:heatpump:max'
    LEFT JOIN rules AS r_ch_min    ON r_ch_min.ruletype    = 'consumption:charger:min'
    LEFT JOIN rules AS r_ch_max    ON r_ch_max.ruletype    = 'consumption:charger:max'
    LEFT JOIN rules AS r_tmin      ON r_tmin.ruletype      = 'state:house:min'
    LEFT JOIN rules AS r_tmax      ON r_tmax.ruletype      = 'state:house:max';

    CREATE VIEW IF NOT EXISTS decision_explain AS
    SELECT
        dwr.*,
        CASE WHEN dwr.net_grid_kw < dwr.grid_limit_kw THEN 1 ELSE 0 END AS mr_sufficient_power_ok,
        CASE WHEN dwr.solar_kw >= dwr.solar_min_kw
                  AND dwr.battery_energy_kwh < dwr.battery_max_kwh
             THEN 1 ELSE 0 END AS mr_charge_battery_energy_ok,
        CASE WHEN ABS(dwr.battery_charge_kw) <= dwr.battery_pmax_kw
             THEN 1 ELSE 0 END AS mr_charge_battery_power_ok,
        CASE WHEN dwr.net_grid_kw < dwr.grid_limit_kw
                  AND dwr.battery_energy_kwh < dwr.battery_max_kwh
             THEN 1 ELSE 0 END AS mr_can_charge_battery_ok,
        (SELECT GROUP_CONCAT(rulename || ': ' || description, ' | ')
           FROM metarules) AS metarules_all
    FROM decision_with_rules AS dwr;
    """)
    conn.commit()

def load_rules(conn: sqlite3.Connection) -> Dict[str, Tuple[int, int, str]]:
    out: Dict[str, Tuple[int,int,str]] = {}
    for row in conn.execute("SELECT ruletype, value, importance, unit FROM rules"):
        out[row[0]] = (row[1], row[2], row[3])
    return out

# ---------- Policy / Control ----------
class HEMSController:
    def __init__(self, rules: Dict[str, Tuple[int,int,str]]):
        self.rules = rules
        self.grid_limit_kw = float(self._get("grid:connection", 17))
        self.solar_min_kw  = float(self._get("production:solar:min", 0))
        self.house_min_c   = float(self._get("state:house:min", 15))
        self.house_max_c   = float(self._get("state:house:max", 20))
        self.batt_min_kwh  = float(self._get("state:battery:min", 0))
        self.batt_max_kwh  = float(self._get("state:battery:max", 18))
        self.batt_pmax_kw  = float(self._get("consumption:battery:max", 5))
        self.hp_min_kw     = float(self._get("consumption:heatpump:min", 0))
        self.hp_max_kw     = float(self._get("consumption:heatpump:max", 7))
        self.chg_min_kw    = float(self._get("consumption:charger:min", 0))
        self.chg_max_kw    = float(self._get("consumption:charger:max", 11))

    def _get(self, key: str, default: float) -> float:
        return self.rules.get(key, (default, 0, ""))[0]

    def decide(self, m: Measurements) -> Setpoints:
        def room_to_max_kwh() -> float: return max(0.0, self.batt_max_kwh - m.battery_energy_kwh)
        def above_min_kwh() -> float:   return max(0.0, m.battery_energy_kwh - self.batt_min_kwh)
        hp_kw = ch_kw = batt_kw = 0.0
        net_grid_kw = m.base_load_kw - m.solar_kw
        if m.needs_heating and m.house_temp_c < self.house_min_c:
            hp_kw = min(self.hp_max_kw, self.hp_min_kw)
        net_grid_kw += hp_kw
        if net_grid_kw > self.grid_limit_kw:
            deficit = net_grid_kw - self.grid_limit_kw
            d = min(deficit, self.batt_pmax_kw, above_min_kwh()/m.step_hours)
            batt_kw -= d; net_grid_kw -= d
        headroom = self.grid_limit_kw - net_grid_kw
        if headroom > 0 and m.solar_kw >= self.solar_min_kw and room_to_max_kwh() > 0:
            c = min(self.batt_pmax_kw, room_to_max_kwh()/m.step_hours, headroom)
            batt_kw += c; net_grid_kw += c; headroom -= c
        desired = min(m.desired_charger_kw, self.chg_max_kw)
        alloc = min(desired, headroom)
        if alloc >= min(self.chg_min_kw, desired):
            ch_kw = alloc; net_grid_kw += ch_kw
        if net_grid_kw > self.grid_limit_kw + 1e-6:
            deficit = net_grid_kw - self.grid_limit_kw
            d = min(deficit, self.batt_pmax_kw + max(0.0, -batt_kw), above_min_kwh()/m.step_hours)
            batt_kw -= d; net_grid_kw -= d
        return Setpoints(round(hp_kw,3), round(ch_kw,3), round(batt_kw,3), round(net_grid_kw,3))

def log_decision(conn, m, sp):
    conn.execute("""INSERT INTO decisions (
        decided_at, base_load_kw, solar_kw, house_temp_c, battery_energy_kwh,
        people_presence_pct, desired_charger_kw, needs_heating, step_hours,
        heatpump_kw, charger_kw, battery_charge_kw, net_grid_kw)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (dt.datetime.utcnow().isoformat(timespec="seconds")+"Z",
         m.base_load_kw, m.solar_kw, m.house_temp_c, m.battery_energy_kwh,
         m.people_presence_pct, m.desired_charger_kw, int(m.needs_heating),
         m.step_hours, sp.heatpump_kw, sp.charger_kw, sp.battery_charge_kw, sp.net_grid_kw))
    conn.commit()

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="HEMS policy engine (SQLite) with views and headers.")
    p.add_argument("--db", default=DB_PATH)
    p.add_argument("--base-load", type=float)
    p.add_argument("--solar", type=float)
    p.add_argument("--temp", type=float)
    p.add_argument("--battery", type=float)
    p.add_argument("--presence", type=float)
    p.add_argument("--ev", type=float)
    p.add_argument("--needs-heating", action="store_true")
    p.add_argument("--step-hours", type=float, default=1.0)
    p.add_argument("--print-rules", action="store_true")
    p.add_argument("--show-dwr", action="store_true")
    p.add_argument("--show-explain", action="store_true")
    p.add_argument("--limit", type=int, default=5)
    return p.parse_args()

def print_query_with_headers(conn, query):
    cur = conn.execute(query)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    print("\n" + " | ".join(cols))
    print("-" * (len(cols)*10))
    for r in rows:
        print(" | ".join(str(x) for x in r))

def interactive_measurements() -> Measurements:
    def ask(p, c=float, d=None):
        s = input(f"{p}{' ['+str(d)+']' if d is not None else ''}: ").strip()
        return c(s) if s else d
    return Measurements(
        ask("Base load kW",float,1.5),
        ask("PV kW",float,3.0),
        ask("Indoor temp °C",float,18.0),
        ask("Battery energy kWh",float,10.0),
        ask("Presence %",float,80.0),
        ask("Desired EV kW",float,7.0),
        ask("Needs heating? (0/1)",int,1)==1,
        ask("Step hours",float,1.0)
    )

def main():
    args = parse_args()
    conn = sqlite3.connect(args.db)
    init_db(conn); create_views(conn)

    if args.print_rules:
        print("Loaded rules:")
        for k,(v,imp,u) in sorted(load_rules(conn).items()):
            print(f"{k:25} {v} {u} (importance {imp})")
        return
    if args.show_dwr:
        print_query_with_headers(conn, f"SELECT * FROM decision_with_rules ORDER BY decided_at DESC LIMIT {args.limit}")
        return
    if args.show_explain:
        print_query_with_headers(conn, f"SELECT * FROM decision_explain ORDER BY decided_at DESC LIMIT {args.limit}")
        return

    # normal decision
    rules = load_rules(conn)
    if all(getattr(args, x) is not None for x in ["base_load","solar","temp","battery","presence","ev"]):
        m = Measurements(args.base_load, args.solar, args.temp, args.battery,
                         args.presence, args.ev, args.needs_heating, args.step_hours)
    else:
        print("Interactive mode:")
        m = interactive_measurements()

    controller = HEMSController(rules)
    sp = controller.decide(m)
    log_decision(conn, m, sp)

    print("\n=== HEMS Decision ===")
    print(f"Grid limit:        {controller.grid_limit_kw:.2f} kW")
    print(f"Net grid:          {sp.net_grid_kw:.2f} kW")
    print(f"Heat pump:         {sp.heatpump_kw:.2f} kW")
    print(f"EV charger:        {sp.charger_kw:.2f} kW")
    print(f"Battery power:     {sp.battery_charge_kw:.2f} kW")
    print("\nLogged to decisions table.")
    conn.close()

if __name__ == "__main__":
    main()
