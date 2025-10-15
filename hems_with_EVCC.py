#!/usr/bin/env python3
# hems_with_EVCC.py - HEMS with EVCC-compatible REST API
import sqlite3
import argparse
import datetime as dt
import json
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List, Optional, Any
from abc import ABC, abstractmethod
import threading
import time
import os

try:
    from flask import Flask, request, jsonify, Response
    from functools import wraps
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Warning: Flask not available. Install with: pip install flask")

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

def load_metarules(conn: sqlite3.Connection) -> Dict[str, Tuple[str, int, str, str]]:
    """Load metarules: rulename -> (ruletype, importance, description, unit)"""
    out: Dict[str, Tuple[str, int, str, str]] = {}
    for row in conn.execute("SELECT rulename, ruletype, importance, description, unit FROM metarules"):
        out[row[0]] = (row[1], row[2], row[3], row[4])
    return out

# ---------- Rule-based Policy / Control ----------

class RuleEvaluator(ABC):
    """Base class for rule evaluators that handle specific metarule logic"""
    
    def __init__(self, metarule_name: str):
        self.metarule_name = metarule_name
        
    @abstractmethod
    def evaluate(self, measurements: Measurements, current_setpoints: Setpoints, rules: Dict[str, Tuple[int,int,str]]) -> Setpoints:
        """Evaluate this rule and return modified setpoints"""
        pass
        
    def get_rule_value(self, rules: Dict[str, Tuple[int,int,str]], key: str, default: float) -> float:
        return rules.get(key, (default, 0, ""))[0]

class SufficientPowerEvaluator(RuleEvaluator):
    """Implements 'sufficient power' metarule - verify rest power available"""
    
    def evaluate(self, m: Measurements, sp: Setpoints, rules: Dict[str, Tuple[int,int,str]]) -> Setpoints:
        grid_limit_kw = self.get_rule_value(rules, "grid:connection", 17)
        
        # Calculate current grid usage
        current_grid_kw = m.base_load_kw - m.solar_kw + sp.heatpump_kw + sp.charger_kw + sp.battery_charge_kw
        
        # If exceeding grid limit, reduce battery charging or increase discharging
        if current_grid_kw > grid_limit_kw:
            batt_pmax_kw = self.get_rule_value(rules, "consumption:battery:max", 5)
            deficit = current_grid_kw - grid_limit_kw
            battery_adjustment = min(deficit, batt_pmax_kw)
            sp.battery_charge_kw -= battery_adjustment
            
        # Update net grid calculation
        sp.net_grid_kw = m.base_load_kw - m.solar_kw + sp.heatpump_kw + sp.charger_kw + sp.battery_charge_kw
        return sp

class ChargeBattery1Evaluator(RuleEvaluator):
    """Implements 'charge battery 1' metarule - charge battery between min and max energy"""
    
    def evaluate(self, m: Measurements, sp: Setpoints, rules: Dict[str, Tuple[int,int,str]]) -> Setpoints:
        batt_min_kwh = self.get_rule_value(rules, "state:battery:min", 0)
        batt_max_kwh = self.get_rule_value(rules, "state:battery:max", 18)
        batt_pmax_kw = self.get_rule_value(rules, "consumption:battery:max", 5)
        solar_min_kw = self.get_rule_value(rules, "production:solar:min", 2)
        grid_limit_kw = self.get_rule_value(rules, "grid:connection", 17)
        
        # Only charge if we have sufficient solar and battery has room
        if m.solar_kw >= solar_min_kw and m.battery_energy_kwh < batt_max_kwh:
            current_grid_kw = m.base_load_kw - m.solar_kw + sp.heatpump_kw + sp.charger_kw
            available_power = grid_limit_kw - current_grid_kw
            
            if available_power > 0:
                room_to_max_kwh = max(0.0, batt_max_kwh - m.battery_energy_kwh)
                max_charge_kw = min(batt_pmax_kw, room_to_max_kwh / m.step_hours, available_power)
                sp.battery_charge_kw = max(sp.battery_charge_kw, max_charge_kw)
                
        return sp

class ChargeBattery2Evaluator(RuleEvaluator):
    """Implements 'charge battery 2' metarule - battery between min and max power"""
    
    def evaluate(self, m: Measurements, sp: Setpoints, rules: Dict[str, Tuple[int,int,str]]) -> Setpoints:
        batt_min_kwh = self.get_rule_value(rules, "state:battery:min", 0)
        batt_pmax_kw = self.get_rule_value(rules, "consumption:battery:max", 5)
        
        # Enforce power limits
        sp.battery_charge_kw = max(-batt_pmax_kw, min(batt_pmax_kw, sp.battery_charge_kw))
        
        # Ensure we don't discharge below minimum energy
        if sp.battery_charge_kw < 0:  # Discharging
            above_min_kwh = max(0.0, m.battery_energy_kwh - batt_min_kwh)
            max_discharge_kw = min(batt_pmax_kw, above_min_kwh / m.step_hours)
            sp.battery_charge_kw = max(sp.battery_charge_kw, -max_discharge_kw)
            
        return sp

class CanChargeBatteryEvaluator(RuleEvaluator):
    """Implements 'can charge battery' metarule - safe power management"""
    
    def evaluate(self, m: Measurements, sp: Setpoints, rules: Dict[str, Tuple[int,int,str]]) -> Setpoints:
        grid_limit_kw = self.get_rule_value(rules, "grid:connection", 17)
        batt_max_kwh = self.get_rule_value(rules, "state:battery:max", 18)
        house_min_c = self.get_rule_value(rules, "state:house:min", 15)
        hp_min_kw = self.get_rule_value(rules, "consumption:heatpump:min", 7)
        chg_min_kw = self.get_rule_value(rules, "consumption:charger:min", 7)
        chg_max_kw = self.get_rule_value(rules, "consumption:charger:max", 11)
        
        # Handle heating priority
        if m.needs_heating and m.house_temp_c < house_min_c:
            sp.heatpump_kw = max(sp.heatpump_kw, hp_min_kw)
            
        # Handle EV charging with available power
        current_grid_kw = m.base_load_kw - m.solar_kw + sp.heatpump_kw + sp.battery_charge_kw
        available_power = grid_limit_kw - current_grid_kw
        
        if available_power > 0:
            desired = min(m.desired_charger_kw, chg_max_kw)
            alloc = min(desired, available_power)
            
            # Only allocate if we can meet minimum or desired power
            if alloc >= min(chg_min_kw, desired):
                sp.charger_kw = alloc
            else:
                sp.charger_kw = 0.0
        else:
            sp.charger_kw = 0.0
            
        return sp

class HEMSController:
    def __init__(self, rules: Dict[str, Tuple[int,int,str]], metarules: Dict[str, Tuple[str, int, str, str]]):
        self.rules = rules
        self.metarules = metarules
        
        # Create evaluators based on metarules
        self.evaluators: List[Tuple[int, RuleEvaluator]] = []
        
        for metarule_name, (ruletype, importance, description, unit) in metarules.items():
            if metarule_name == "sufficient power":
                evaluator = SufficientPowerEvaluator(metarule_name)
            elif metarule_name == "charge battery 1":
                evaluator = ChargeBattery1Evaluator(metarule_name)
            elif metarule_name == "charge battery 2":
                evaluator = ChargeBattery2Evaluator(metarule_name)
            elif metarule_name == "can charge battery":
                evaluator = CanChargeBatteryEvaluator(metarule_name)
            else:
                continue  # Skip unknown metarules
                
            self.evaluators.append((importance, evaluator))
        
        # Sort by importance (lower number = higher priority)
        self.evaluators.sort(key=lambda x: x[0])

    def decide(self, m: Measurements) -> Setpoints:
        # Initialize setpoints
        sp = Setpoints(heatpump_kw=0.0, charger_kw=0.0, battery_charge_kw=0.0, net_grid_kw=0.0)
        
        # Apply each metarule evaluator in priority order
        for importance, evaluator in self.evaluators:
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
        
        for importance, evaluator in self.evaluators:
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
                trace.append(f"{evaluator.metarule_name} (importance={importance}): {', '.join(changes)}")
            else:
                trace.append(f"{evaluator.metarule_name} (importance={importance}): no changes")
        
        # Round final values
        sp.heatpump_kw = round(sp.heatpump_kw, 3)
        sp.charger_kw = round(sp.charger_kw, 3)
        sp.battery_charge_kw = round(sp.battery_charge_kw, 3)
        sp.net_grid_kw = round(sp.net_grid_kw, 3)
        
        return sp, trace

# Legacy controller for compatibility
class LegacyHEMSController:
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

# ---------- EVCC-compatible Data Models ----------
@dataclass
class LoadPoint:
    id: int
    title: str
    mode: str = "pv"  # off, now, minpv, pv
    enabled: bool = True
    phases: int = 3
    min_current: float = 6.0
    max_current: float = 16.0
    enable_threshold: float = 0.0  # W
    disable_threshold: float = 0.0  # W
    enable_delay: int = 0  # seconds
    disable_delay: int = 0  # seconds
    priority: int = 0
    charging_power: float = 0.0  # current charging power in W
    charged_energy: float = 0.0  # kWh charged in current session
    vehicle: Optional[str] = None

@dataclass
class Vehicle:
    name: str
    title: str
    capacity: float  # kWh
    soc: Optional[float] = None  # % 
    range: Optional[float] = None  # km
    min_soc: float = 0.0  # %
    limit_soc: float = 100.0  # %
    connected: bool = False
    charging: bool = False

@dataclass
class Battery:
    soc: float = 50.0  # %
    capacity: float = 18.0  # kWh
    power: float = 0.0  # W (positive = charging, negative = discharging)
    mode: str = "normal"  # normal, hold, charge
    grid_charge_limit: Optional[float] = None
    buffer_soc: float = 0.0  # %
    buffer_start_soc: float = 0.0  # %
    priority_soc: float = 0.0  # %

@dataclass
class Site:
    grid_power: float = 0.0  # W (positive = import, negative = export)
    pv_power: float = 0.0  # W
    battery_power: float = 0.0  # W  
    home_power: float = 0.0  # W
    residual_power: float = 0.0  # W target
    tariff_grid: Optional[float] = None  # EUR/kWh
    tariff_feedin: Optional[float] = None  # EUR/kWh

# ---------- HEMS State Management ----------
class HEMSState:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.loadpoints: List[LoadPoint] = [
            LoadPoint(id=1, title="EV Charger", mode="pv")
        ]
        self.vehicles: Dict[str, Vehicle] = {
            "ev1": Vehicle(name="ev1", title="Electric Vehicle", capacity=60.0)
        }
        self.battery = Battery()
        self.site = Site()
        self.controller: Optional[HEMSController] = None
        self.last_update = dt.datetime.utcnow()
        self._lock = threading.Lock()
        self._init_database()
        self._load_controller()
        
    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        init_db(conn)
        create_views(conn)
        conn.close()
        
    def _load_controller(self):
        conn = sqlite3.connect(self.db_path)
        rules = load_rules(conn)
        metarules = load_metarules(conn)
        self.controller = HEMSController(rules, metarules)
        conn.close()
    
    def update_from_measurements(self, measurements: Measurements):
        """Update state from new measurements and run HEMS decision"""
        with self._lock:
            # Update site data
            self.site.home_power = measurements.base_load_kw * 1000  # Convert to W
            self.site.pv_power = measurements.solar_kw * 1000
            self.battery.soc = (measurements.battery_energy_kwh / self.battery.capacity) * 100
            
            # Update vehicle from presence and desired charger
            if self.vehicles["ev1"].connected:
                self.vehicles["ev1"].soc = 50.0  # Mock SoC
                
            # Run HEMS decision
            if self.controller:
                setpoints = self.controller.decide(measurements)
                
                # Update loadpoint charging power
                self.loadpoints[0].charging_power = setpoints.charger_kw * 1000  # Convert to W
                
                # Update battery power
                self.battery.power = setpoints.battery_charge_kw * 1000
                
                # Update grid power
                self.site.grid_power = setpoints.net_grid_kw * 1000
                
                # Log decision
                conn = sqlite3.connect(self.db_path)
                log_decision(conn, measurements, setpoints)
                conn.close()
                
            self.last_update = dt.datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for API responses"""
        return {
            "site": asdict(self.site),
            "loadpoints": [asdict(lp) for lp in self.loadpoints],
            "vehicles": {k: asdict(v) for k, v in self.vehicles.items()},
            "battery": asdict(self.battery),
            "updated": self.last_update.isoformat() + "Z"
        }

# ---------- Flask REST API (EVCC-compatible) ----------
if FLASK_AVAILABLE:
    # Global state instance
    hems_state = HEMSState()
    
    # Flask App Setup
    app = Flask(__name__)
    app.config['JSON_SORT_KEYS'] = False

    # Authentication (simplified)
    def require_auth(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            # Simplified auth - in production, implement proper authentication
            return f(*args, **kwargs)
        return decorated

    # Helper Functions
    def success_response(result: Any) -> Response:
        """Standard success response format"""
        return jsonify({"result": result})

    def error_response(message: str, code: int = 400) -> Tuple[Response, int]:
        """Standard error response format"""
        return jsonify({"error": message}), code

    def get_loadpoint(lp_id: int) -> Optional[LoadPoint]:
        """Get loadpoint by ID"""
        for lp in hems_state.loadpoints:
            if lp.id == lp_id:
                return lp
        return None

    def get_vehicle(name: str) -> Optional[Vehicle]:
        """Get vehicle by name"""
        return hems_state.vehicles.get(name)

    # API Endpoints
    
    # General endpoints
    @app.route('/api/health', methods=['GET'])
    def health():
        """Health check endpoint"""
        return "OK"

    @app.route('/api/state', methods=['GET'])
    def get_state():
        """Get complete system state"""
        jq_filter = request.args.get('jq', '.')
        state = hems_state.to_dict()
        
        # Simple JQ-like filtering (basic implementation)
        if jq_filter == '.':
            result = state
        elif jq_filter.startswith('.loadpoints['):
            try:
                idx = int(jq_filter.split('[')[1].split(']')[0])
                result = state['loadpoints'][idx] if idx < len(state['loadpoints']) else None
            except (IndexError, ValueError):
                result = None
        else:
            result = state
        
        return success_response(result)

    # Loadpoint endpoints
    @app.route('/api/loadpoints/<int:lp_id>/mode/<mode>', methods=['POST'])
    def set_loadpoint_mode(lp_id: int, mode: str):
        """Set loadpoint charging mode with optional parameters"""
        lp = get_loadpoint(lp_id)
        if not lp:
            return error_response(f"Loadpoint {lp_id} not found", 404)
        
        valid_modes = ["off", "now", "minpv", "pv"]
        if mode not in valid_modes:
            return error_response(f"Invalid mode. Must be one of: {valid_modes}")
        
        # Get optional parameters from request body
        data = request.get_json() if request.is_json else {}
        priority = data.get('priority', None)
        enable_threshold = data.get('enable_threshold', None)
        disable_threshold = data.get('disable_threshold', None)
        
        with hems_state._lock:
            lp.mode = mode
            
            # Apply optional parameters if provided
            if priority is not None:
                if isinstance(priority, int) and priority >= 0:
                    lp.priority = priority
                else:
                    return error_response("Priority must be a non-negative integer")
            
            if enable_threshold is not None:
                if isinstance(enable_threshold, (int, float)):
                    lp.enable_threshold = float(enable_threshold)
                else:
                    return error_response("Enable threshold must be a number (watts)")
            
            if disable_threshold is not None:
                if isinstance(disable_threshold, (int, float)):
                    lp.disable_threshold = float(disable_threshold)
                else:
                    return error_response("Disable threshold must be a number (watts)")
        
        # Return the updated loadpoint configuration
        response_data = {
            "mode": mode,
            "priority": lp.priority,
            "enable_threshold": lp.enable_threshold,
            "disable_threshold": lp.disable_threshold
        }
        
        return success_response(response_data)

    @app.route('/api/loadpoints/<int:lp_id>/phases/<int:phases>', methods=['POST'])
    def set_loadpoint_phases(lp_id: int, phases: int):
        """Set loadpoint phases"""
        lp = get_loadpoint(lp_id)
        if not lp:
            return error_response(f"Loadpoint {lp_id} not found", 404)
        
        if phases not in [0, 1, 3]:
            return error_response("Invalid phases value. Must be 0, 1, or 3", 400)
        
        with hems_state._lock:
            lp.phases = phases
        
        return success_response(phases)

    @app.route('/api/loadpoints/<int:lp_id>/mincurrent/<float:current>', methods=['POST'])
    def set_loadpoint_min_current(lp_id: int, current: float):
        """Set loadpoint minimum current"""
        lp = get_loadpoint(lp_id)
        if not lp:
            return error_response(f"Loadpoint {lp_id} not found", 404)
        
        if current < 0:
            return error_response("Current must be positive")
        
        with hems_state._lock:
            lp.min_current = current
        
        return success_response(current)

    @app.route('/api/loadpoints/<int:lp_id>/maxcurrent/<float:current>', methods=['POST'])
    def set_loadpoint_max_current(lp_id: int, current: float):
        """Set loadpoint maximum current"""
        lp = get_loadpoint(lp_id)
        if not lp:
            return error_response(f"Loadpoint {lp_id} not found", 404)
        
        if current < 0:
            return error_response("Current must be positive")
        
        with hems_state._lock:
            lp.max_current = current
        
        return success_response(current)

    @app.route('/api/loadpoints/<int:lp_id>/vehicle/<vehicle_name>', methods=['POST'])
    def assign_vehicle_to_loadpoint(lp_id: int, vehicle_name: str):
        """Assign vehicle to loadpoint"""
        lp = get_loadpoint(lp_id)
        if not lp:
            return error_response(f"Loadpoint {lp_id} not found", 404)
        
        vehicle = get_vehicle(vehicle_name)
        if not vehicle:
            return error_response(f"Vehicle {vehicle_name} not found", 404)
        
        with hems_state._lock:
            lp.vehicle = vehicle_name
            vehicle.connected = True
        
        return success_response({"vehicle": {"title": vehicle.title}})

    @app.route('/api/loadpoints/<int:lp_id>/vehicle', methods=['DELETE'])
    def remove_vehicle_from_loadpoint(lp_id: int):
        """Remove vehicle from loadpoint"""
        lp = get_loadpoint(lp_id)
        if not lp:
            return error_response(f"Loadpoint {lp_id} not found", 404)
        
        with hems_state._lock:
            if lp.vehicle:
                vehicle = get_vehicle(lp.vehicle)
                if vehicle:
                    vehicle.connected = False
            lp.vehicle = None
        
        return success_response({})

    # Vehicle endpoints
    @app.route('/api/vehicles/<vehicle_name>/minsoc/<float:soc>', methods=['POST'])
    def set_vehicle_min_soc(vehicle_name: str, soc: float):
        """Set vehicle minimum SoC"""
        vehicle = get_vehicle(vehicle_name)
        if not vehicle:
            return error_response(f"Vehicle {vehicle_name} not found", 404)
        
        if not 0 <= soc <= 100:
            return error_response("SoC must be between 0 and 100")
        
        with hems_state._lock:
            vehicle.min_soc = soc
        
        return success_response({"soc": soc})

    @app.route('/api/vehicles/<vehicle_name>/limitsoc/<float:soc>', methods=['POST'])
    def set_vehicle_limit_soc(vehicle_name: str, soc: float):
        """Set vehicle limit SoC"""
        vehicle = get_vehicle(vehicle_name)
        if not vehicle:
            return error_response(f"Vehicle {vehicle_name} not found", 404)
        
        if not 0 <= soc <= 100:
            return error_response("SoC must be between 0 and 100")
        
        with hems_state._lock:
            vehicle.limit_soc = soc
        
        return success_response({"soc": soc})

    # Battery endpoints
    @app.route('/api/batterymode/<mode>', methods=['POST'])
    def set_battery_mode(mode: str):
        """Set battery mode"""
        valid_modes = ["normal", "hold", "charge"]
        if mode not in valid_modes:
            return error_response(f"Invalid mode. Must be one of: {valid_modes}")
        
        with hems_state._lock:
            hems_state.battery.mode = mode
        
        return success_response({"mode": mode})

    @app.route('/api/batterymode', methods=['DELETE'])
    def reset_battery_mode():
        """Reset battery mode to normal"""
        with hems_state._lock:
            hems_state.battery.mode = "normal"
        
        return success_response({"mode": "normal"})

    @app.route('/api/bufferstartsoc/<float:soc>', methods=['POST'])
    def set_battery_buffer_start_soc(soc: float):
        """Set battery buffer start SoC"""
        if not 0 <= soc <= 100:
            return error_response("SoC must be between 0 and 100")
        
        with hems_state._lock:
            hems_state.battery.buffer_start_soc = soc
        
        return success_response(soc)

    @app.route('/api/buffersoc/<float:soc>', methods=['POST'])
    def set_battery_buffer_soc(soc: float):
        """Set battery buffer SoC"""
        if not 0 <= soc <= 100:
            return error_response("SoC must be between 0 and 100")
        
        with hems_state._lock:
            hems_state.battery.buffer_soc = soc
        
        return success_response(soc)

    @app.route('/api/prioritysoc/<float:soc>', methods=['POST'])
    def set_battery_priority_soc(soc: float):
        """Set battery priority SoC"""
        if not 0 <= soc <= 100:
            return error_response("SoC must be between 0 and 100")
        
        with hems_state._lock:
            hems_state.battery.priority_soc = soc
        
        return success_response(soc)

    @app.route('/api/batterygridchargelimit/<float:cost>', methods=['POST'])
    def set_battery_grid_charge_limit(cost: float):
        """Set battery grid charge limit"""
        with hems_state._lock:
            hems_state.battery.grid_charge_limit = cost
        
        return success_response(cost)

    @app.route('/api/batterygridchargelimit', methods=['DELETE'])
    def remove_battery_grid_charge_limit():
        """Remove battery grid charge limit"""
        with hems_state._lock:
            hems_state.battery.grid_charge_limit = None
        
        return success_response(None)

    # Site endpoints
    @app.route('/api/residualpower/<float:power>', methods=['POST'])
    def set_residual_power(power: float):
        """Set target residual power"""
        with hems_state._lock:
            hems_state.site.residual_power = power
        
        return success_response(power)

    # HEMS-specific endpoints
    @app.route('/api/hems/measurements', methods=['POST'])
    def update_measurements():
        """Update HEMS with new measurements"""
        try:
            data = request.json
            measurements = Measurements(
                base_load_kw=data.get('base_load_kw', 1.5),
                solar_kw=data.get('solar_kw', 0.0),
                house_temp_c=data.get('house_temp_c', 20.0),
                battery_energy_kwh=data.get('battery_energy_kwh', 10.0),
                people_presence_pct=data.get('people_presence_pct', 0.0),
                desired_charger_kw=data.get('desired_charger_kw', 0.0),
                needs_heating=data.get('needs_heating', False),
                step_hours=data.get('step_hours', 1.0)
            )
            
            hems_state.update_from_measurements(measurements)
            return success_response({"updated": hems_state.last_update.isoformat() + "Z"})
            
        except Exception as e:
            return error_response(f"Invalid measurements data: {str(e)}")

    @app.route('/api/hems/rules', methods=['GET'])
    def get_rules():
        """Get current rules and metarules"""
        conn = sqlite3.connect(hems_state.db_path)
        rules = load_rules(conn)
        metarules = load_metarules(conn)
        conn.close()
        
        return success_response({
            "rules": {k: {"value": v[0], "importance": v[1], "unit": v[2]} for k, v in rules.items()},
            "metarules": {k: {"ruletype": v[0], "importance": v[1], "description": v[2], "unit": v[3]} for k, v in metarules.items()}
        })

    def start_api_server(host="127.0.0.1", port=7070, debug=False):
        """Start the Flask API server"""
        print(f"Starting HEMS API server on {host}:{port}")
        print(f"Using database: {hems_state.db_path}")
        print("API Documentation: https://docs.evcc.io/en/docs/integrations/rest-api")
        print("\nExample endpoints:")
        print(f"  GET  http://{host}:{port}/api/health")
        print(f"  GET  http://{host}:{port}/api/state")
        print(f"  POST http://{host}:{port}/api/loadpoints/1/mode/pv")
        print(f"  POST http://{host}:{port}/api/hems/measurements")
        
        app.run(host=host, port=port, debug=debug)

else:
    def start_api_server(*args, **kwargs):
        print("Flask not available. Cannot start API server.")
        print("Install Flask with: pip install flask")

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
    p = argparse.ArgumentParser(description="HEMS with EVCC-compatible REST API")
    p.add_argument("--db", default=DB_PATH)
    
    # API server options
    p.add_argument("--api", action="store_true", help="Start REST API server")
    p.add_argument("--host", default="127.0.0.1", help="API host")
    p.add_argument("--port", type=int, default=7070, help="API port")
    p.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    # Original CLI options
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
    p.add_argument("--show-metarules-eval", action="store_true", help="Show metarule evaluation process")
    p.add_argument("--use-legacy", action="store_true", help="Use legacy monolithic controller")
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
    
    # Start API server if requested
    if args.api:
        if not FLASK_AVAILABLE:
            print("Error: Flask not available. Install with: pip install flask")
            return 1
        
        # Update global state with custom DB path
        if FLASK_AVAILABLE:
            global hems_state
            hems_state = HEMSState(args.db)
        
        start_api_server(args.host, args.port, args.debug)
        return 0
    
    # Original CLI functionality
    conn = sqlite3.connect(args.db)
    init_db(conn); create_views(conn)

    if args.print_rules:
        print("Loaded rules:")
        for k,(v,imp,u) in sorted(load_rules(conn).items()):
            print(f"{k:25} {v} {u} (importance {imp})")
        print("\nLoaded metarules:")
        for k,(rt,imp,desc,u) in sorted(load_metarules(conn).items()):
            print(f"{k:25} {rt} {u} (importance {imp}) - {desc}")
        return
    if args.show_dwr:
        print_query_with_headers(conn, f"SELECT * FROM decision_with_rules ORDER BY decided_at DESC LIMIT {args.limit}")
        return
    if args.show_explain:
        print_query_with_headers(conn, f"SELECT * FROM decision_explain ORDER BY decided_at DESC LIMIT {args.limit}")
        return

    # normal decision
    rules = load_rules(conn)
    metarules = load_metarules(conn)
    
    if all(getattr(args, x) is not None for x in ["base_load","solar","temp","battery","presence","ev"]):
        m = Measurements(args.base_load, args.solar, args.temp, args.battery,
                         args.presence, args.ev, args.needs_heating, args.step_hours)
    else:
        print("Interactive mode:")
        m = interactive_measurements()

    # Choose controller based on arguments
    if args.use_legacy:
        controller = LegacyHEMSController(rules)
        sp = controller.decide(m)
        print("Using legacy monolithic controller")
    else:
        controller = HEMSController(rules, metarules)
        if args.show_metarules_eval:
            sp, trace = controller.decide_with_trace(m)
            print("\n=== Metarule Evaluation Trace ===")
            for line in trace:
                print(line)
        else:
            sp = controller.decide(m)
        print("Using metarule-based controller")
    
    log_decision(conn, m, sp)

    print("\n=== HEMS Decision ===")
    grid_limit_kw = rules.get("grid:connection", (17, 0, ""))[0]
    print(f"Grid limit:        {grid_limit_kw:.2f} kW")
    print(f"Net grid:          {sp.net_grid_kw:.2f} kW")
    print(f"Heat pump:         {sp.heatpump_kw:.2f} kW")
    print(f"EV charger:        {sp.charger_kw:.2f} kW")
    print(f"Battery power:     {sp.battery_charge_kw:.2f} kW")
    print("\nLogged to decisions table.")
    conn.close()

if __name__ == "__main__":
    main()
