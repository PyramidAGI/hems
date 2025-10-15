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
    ("people:presence:min",2,"movement at home",5,"%"),
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

# ---------- Rule-based Policy / Control ----------
from abc import ABC, abstractmethod
from typing import List, Any

class RuleEvaluator(ABC):
    """Base class for rule evaluators that handle specific units and constraints"""
    
    def __init__(self, name: str, priority: int = 0):
        self.name = name
        self.priority = priority
        
    @abstractmethod
    def evaluate(self, measurements: Measurements, current_setpoints: Setpoints, rules: Dict[str, Tuple[int,int,str]]) -> Setpoints:
        """Evaluate this rule and return modified setpoints"""
        pass
        
    def get_rule_value(self, rules: Dict[str, Tuple[int,int,str]], key: str, default: float) -> float:
        return rules.get(key, (default, 0, ""))[0]

class PowerLimitEvaluator(RuleEvaluator):
    """Evaluates power limits in kW units"""
    
    def evaluate(self, m: Measurements, sp: Setpoints, rules: Dict[str, Tuple[int,int,str]]) -> Setpoints:
        grid_limit_kw = self.get_rule_value(rules, "grid:connection", 17)
        hp_max_kw = self.get_rule_value(rules, "consumption:heatpump:max", 7)
        chg_max_kw = self.get_rule_value(rules, "consumption:charger:max", 11)
        batt_pmax_kw = self.get_rule_value(rules, "consumption:battery:max", 5)
        
        # Enforce power limits
        sp.heatpump_kw = min(sp.heatpump_kw, hp_max_kw)
        sp.charger_kw = min(sp.charger_kw, chg_max_kw)
        sp.battery_charge_kw = max(-batt_pmax_kw, min(batt_pmax_kw, sp.battery_charge_kw))
        sp.net_grid_kw = min(sp.net_grid_kw, grid_limit_kw)
        
        return sp

class EnergyConstraintEvaluator(RuleEvaluator):
    """Evaluates energy constraints in kWh units"""
    
    def evaluate(self, m: Measurements, sp: Setpoints, rules: Dict[str, Tuple[int,int,str]]) -> Setpoints:
        batt_min_kwh = self.get_rule_value(rules, "state:battery:min", 0)
        batt_max_kwh = self.get_rule_value(rules, "state:battery:max", 18)
        batt_pmax_kw = self.get_rule_value(rules, "consumption:battery:max", 5)
        
        # Calculate energy limits for this time step
        room_to_max_kwh = max(0.0, batt_max_kwh - m.battery_energy_kwh)
        above_min_kwh = max(0.0, m.battery_energy_kwh - batt_min_kwh)
        
        # Limit battery charging based on available capacity
        if sp.battery_charge_kw > 0:  # Charging
            max_charge_kw = min(batt_pmax_kw, room_to_max_kwh / m.step_hours)
            sp.battery_charge_kw = min(sp.battery_charge_kw, max_charge_kw)
        elif sp.battery_charge_kw < 0:  # Discharging
            max_discharge_kw = min(batt_pmax_kw, above_min_kwh / m.step_hours)
            sp.battery_charge_kw = max(sp.battery_charge_kw, -max_discharge_kw)
            
        return sp

class TemperatureControlEvaluator(RuleEvaluator):
    """Evaluates temperature control rules in Celsius"""
    
    def evaluate(self, m: Measurements, sp: Setpoints, rules: Dict[str, Tuple[int,int,str]]) -> Setpoints:
        house_min_c = self.get_rule_value(rules, "state:house:min", 15)
        house_max_c = self.get_rule_value(rules, "state:house:max", 20)
        hp_min_kw = self.get_rule_value(rules, "consumption:heatpump:min", 7)  # From seed data
        hp_max_kw = self.get_rule_value(rules, "consumption:heatpump:max", 7)
        
        # Determine heating need based on temperature and presence
        if m.needs_heating and m.house_temp_c < house_min_c:
            # Use minimum steady state power for heating
            sp.heatpump_kw = max(sp.heatpump_kw, hp_min_kw)
        elif m.house_temp_c > house_max_c:
            sp.heatpump_kw = 0.0  # No heating when too warm
            
        return sp

class GridBalanceEvaluator(RuleEvaluator):
    """Evaluates grid balance and manages load distribution"""
    
    def evaluate(self, m: Measurements, sp: Setpoints, rules: Dict[str, Tuple[int,int,str]]) -> Setpoints:
        grid_limit_kw = self.get_rule_value(rules, "grid:connection", 17)
        batt_pmax_kw = self.get_rule_value(rules, "consumption:battery:max", 5)
        
        # Calculate current grid load
        current_grid_kw = m.base_load_kw - m.solar_kw + sp.heatpump_kw + sp.charger_kw + sp.battery_charge_kw
        
        # If exceeding grid limit, discharge battery to compensate
        if current_grid_kw > grid_limit_kw:
            deficit = current_grid_kw - grid_limit_kw
            battery_discharge = min(deficit, batt_pmax_kw)
            sp.battery_charge_kw -= battery_discharge
            
        # Update net grid calculation
        sp.net_grid_kw = m.base_load_kw - m.solar_kw + sp.heatpump_kw + sp.charger_kw + sp.battery_charge_kw
        
        return sp

class SolarOptimizationEvaluator(RuleEvaluator):
    """Evaluates solar energy optimization rules"""
    
    def evaluate(self, m: Measurements, sp: Setpoints, rules: Dict[str, Tuple[int,int,str]]) -> Setpoints:
        solar_min_kw = self.get_rule_value(rules, "production:solar:min", 0)
        grid_limit_kw = self.get_rule_value(rules, "grid:connection", 17)
        batt_max_kwh = self.get_rule_value(rules, "state:battery:max", 18)
        batt_pmax_kw = self.get_rule_value(rules, "consumption:battery:max", 5)
        
        # If sufficient solar and battery has room, prioritize battery charging
        if m.solar_kw >= solar_min_kw and m.battery_energy_kwh < batt_max_kwh:
            current_grid_kw = m.base_load_kw - m.solar_kw + sp.heatpump_kw + sp.charger_kw
            headroom = grid_limit_kw - current_grid_kw
            
            if headroom > 0:
                room_to_max_kwh = max(0.0, batt_max_kwh - m.battery_energy_kwh)
                max_charge = min(batt_pmax_kw, room_to_max_kwh / m.step_hours, headroom)
                sp.battery_charge_kw = max(sp.battery_charge_kw, max_charge)
                
        return sp

class ChargerPriorityEvaluator(RuleEvaluator):
    """Evaluates EV charger priority and minimum power rules"""
    
    def evaluate(self, m: Measurements, sp: Setpoints, rules: Dict[str, Tuple[int,int,str]]) -> Setpoints:
        chg_min_kw = self.get_rule_value(rules, "consumption:charger:min", 0)
        chg_max_kw = self.get_rule_value(rules, "consumption:charger:max", 11)
        grid_limit_kw = self.get_rule_value(rules, "grid:connection", 17)
        
        # Calculate available power for charging
        current_grid_kw = m.base_load_kw - m.solar_kw + sp.heatpump_kw + sp.battery_charge_kw
        available_power = grid_limit_kw - current_grid_kw
        
        # Determine charger allocation
        desired = min(m.desired_charger_kw, chg_max_kw)
        alloc = min(desired, available_power)
        
        # Only allocate if we can meet minimum or desired power
        if alloc >= min(chg_min_kw, desired):
            sp.charger_kw = alloc
        else:
            sp.charger_kw = 0.0
            
        return sp

class HEMSController:
    def __init__(self, rules: Dict[str, Tuple[int,int,str]]):
        self.rules = rules
        self.evaluators: List[RuleEvaluator] = [
            TemperatureControlEvaluator("temperature_control", priority=1),
            PowerLimitEvaluator("power_limits", priority=2),
            EnergyConstraintEvaluator("energy_constraints", priority=3),
            SolarOptimizationEvaluator("solar_optimization", priority=4),
            ChargerPriorityEvaluator("charger_priority", priority=5),
            GridBalanceEvaluator("grid_balance", priority=6),
        ]
        # Sort evaluators by priority
        self.evaluators.sort(key=lambda x: x.priority)

    def decide(self, m: Measurements) -> Setpoints:
        # Initialize setpoints
        sp = Setpoints(heatpump_kw=0.0, charger_kw=0.0, battery_charge_kw=0.0, net_grid_kw=0.0)
        
        # Apply each rule evaluator in priority order
        for evaluator in self.evaluators:
            sp = evaluator.evaluate(m, sp, self.rules)
        
        # Round final values
        sp.heatpump_kw = round(sp.heatpump_kw, 3)
        sp.charger_kw = round(sp.charger_kw, 3)
        sp.battery_charge_kw = round(sp.battery_charge_kw, 3)
        sp.net_grid_kw = round(sp.net_grid_kw, 3)
        
        return sp
    
    def decide_with_trace(self, m: Measurements) -> Tuple[Setpoints, List[str]]:
        """Decision with evaluation trace for debugging"""
        trace = []
        sp = Setpoints(heatpump_kw=0.0, charger_kw=0.0, battery_charge_kw=0.0, net_grid_kw=0.0)
        trace.append(f"Initial: HP={sp.heatpump_kw:.3f}, CH={sp.charger_kw:.3f}, BAT={sp.battery_charge_kw:.3f}, GRID={sp.net_grid_kw:.3f}")
        
        for evaluator in self.evaluators:
            sp_before = Setpoints(sp.heatpump_kw, sp.charger_kw, sp.battery_charge_kw, sp.net_grid_kw)
            sp = evaluator.evaluate(m, sp, self.rules)
            
            changes = []
            if abs(sp.heatpump_kw - sp_before.heatpump_kw) > 1e-6:
                changes.append(f"HP: {sp_before.heatpump_kw:.3f}→{sp.heatpump_kw:.3f}")
            if abs(sp.charger_kw - sp_before.charger_kw) > 1e-6:
                changes.append(f"CH: {sp_before.charger_kw:.3f}→{sp.charger_kw:.3f}")
            if abs(sp.battery_charge_kw - sp_before.battery_charge_kw) > 1e-6:
                changes.append(f"BAT: {sp_before.battery_charge_kw:.3f}→{sp.battery_charge_kw:.3f}")
            if abs(sp.net_grid_kw - sp_before.net_grid_kw) > 1e-6:
                changes.append(f"GRID: {sp_before.net_grid_kw:.3f}→{sp.net_grid_kw:.3f}")
                
            if changes:
                trace.append(f"{evaluator.name}: {', '.join(changes)}")
            else:
                trace.append(f"{evaluator.name}: no changes")
        
        # Round final values
        sp.heatpump_kw = round(sp.heatpump_kw, 3)
        sp.charger_kw = round(sp.charger_kw, 3)
        sp.battery_charge_kw = round(sp.battery_charge_kw, 3)
        sp.net_grid_kw = round(sp.net_grid_kw, 3)
        
        return sp, trace

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
    p.add_argument("--show-rules-eval", action="store_true", help="Show rule evaluation process")
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
    
    if args.show_rules_eval:
        sp, trace = controller.decide_with_trace(m)
        print("\n=== Rule Evaluation Trace ===")
        for line in trace:
            print(line)
    else:
        sp = controller.decide(m)
    
    log_decision(conn, m, sp)

    print("\n=== HEMS Decision ===")
    grid_limit_kw = controller.evaluators[0].get_rule_value(rules, "grid:connection", 17)
    print(f"Grid limit:        {grid_limit_kw:.2f} kW")
    print(f"Net grid:          {sp.net_grid_kw:.2f} kW")
    print(f"Heat pump:         {sp.heatpump_kw:.2f} kW")
    print(f"EV charger:        {sp.charger_kw:.2f} kW")
    print(f"Battery power:     {sp.battery_charge_kw:.2f} kW")
    print("\nLogged to decisions table.")
    conn.close()

if __name__ == "__main__":
    main()
