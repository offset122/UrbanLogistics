"""
Core data structures for the Branch-and-Price solver.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

class VehicleType(Enum):
    """Vehicle types in the system."""
    DRONE = "drone"
    TRUCK = "truck"
    METRO = "metro"

class ColumnType(Enum):
    """Column types for different vehicle routes."""
    DRONE_ROUTE = "drone_route"
    TRUCK_ROUTE = "truck_route"
    METRO_SCHEDULE = "metro_schedule"

@dataclass
class Column:
    """
    Represents a column (route/schedule) in the RMP.
    """
    id: str
    type: ColumnType
    vehicle_type: VehicleType
    direct_cost: float
    reduced_cost: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default details if not provided."""
        if not self.details:
            self.details = {
                'a_ip': {},  # Demand coverage coefficients
                'delta_up': {},  # Inventory up coefficients
                'delta_down': {},  # Inventory down coefficients
                'resource_usage': {},  # Resource usage per time slice
                'route': [],  # Sequence of nodes/stations
                'timing': [],  # Timing information
                'cargo': {}  # Cargo information
            }

@dataclass
class DroneLabel:
    """
    Label for drone routing in label-setting algorithm.
    """
    node: int
    time: int
    load: float
    energy: float
    cost: float
    path: List[int] = field(default_factory=list)
    cargo_up: float = 0.0
    cargo_down: float = 0.0
    
    def copy(self) -> 'DroneLabel':
        """Create a copy of the label."""
        return DroneLabel(
            node=self.node,
            time=self.time,
            load=self.load,
            energy=self.energy,
            cost=self.cost,
            path=self.path.copy(),
            cargo_up=self.cargo_up,
            cargo_down=self.cargo_down
        )
    
    def dominates(self, other: 'DroneLabel') -> bool:
        """Check if this label dominates another."""
        if self.node != other.node:
            return False
        
        return (self.time <= other.time and 
                self.load <= other.load and
                self.energy >= other.energy and
                self.cost <= other.cost and
                self.cargo_up >= other.cargo_up and
                self.cargo_down >= other.cargo_down and
                (self.time < other.time or self.load < other.load or 
                 self.energy > other.energy or self.cost < other.cost or
                 self.cargo_up > other.cargo_up or self.cargo_down > other.cargo_down))

@dataclass
class TruckLabel:
    """
    Label for truck routing in label-setting algorithm.
    """
    node: int
    time: int
    load: float
    cost: float
    path: List[int] = field(default_factory=list)
    cargo_up: float = 0.0
    cargo_down: float = 0.0
    
    def copy(self) -> 'TruckLabel':
        """Create a copy of the label."""
        return TruckLabel(
            node=self.node,
            time=self.time,
            load=self.load,
            cost=self.cost,
            path=self.path.copy(),
            cargo_up=self.cargo_up,
            cargo_down=self.cargo_down
        )
    
    def dominates(self, other: 'TruckLabel') -> bool:
        """Check if this label dominates another."""
        if self.node != other.node:
            return False
        
        return (self.time <= other.time and 
                self.load <= other.load and
                self.cost <= other.cost and
                self.cargo_up >= other.cargo_up and
                self.cargo_down >= other.cargo_down and
                (self.time < other.time or self.load < other.load or 
                 self.cost < other.cost or self.cargo_up > other.cargo_up or
                 self.cargo_down > other.cargo_down))

@dataclass
class MetroLabel:
    """
    Label for metro scheduling in label-setting algorithm.
    """
    station: int
    time: int
    load: float
    cost: float
    schedule: List[Tuple[int, int]] = field(default_factory=list)  # (station, time) pairs
    cargo_up: float = 0.0
    cargo_down: float = 0.0
    
    def copy(self) -> 'MetroLabel':
        """Create a copy of the label."""
        return MetroLabel(
            station=self.station,
            time=self.time,
            load=self.load,
            cost=self.cost,
            schedule=self.schedule.copy(),
            cargo_up=self.cargo_up,
            cargo_down=self.cargo_down
        )
    
    def dominates(self, other: 'MetroLabel') -> bool:
        """Check if this label dominates another."""
        if self.station != other.station:
            return False
        
        return (self.time <= other.time and 
                self.load <= other.load and
                self.cost <= other.cost and
                self.cargo_up >= other.cargo_up and
                self.cargo_down >= other.cargo_down and
                (self.time < other.time or self.load < other.load or 
                 self.cost < other.cost or self.cargo_up > other.cargo_up or
                 self.cargo_down > other.cargo_down))

@dataclass
class MetroBALabel:
    """
    Label for metro big-arc scheduling (optimized version).
    """
    station: int
    time: int
    load: float
    cost: float
    big_arcs: List[Dict[str, Any]] = field(default_factory=list)
    cargo_up: float = 0.0
    cargo_down: float = 0.0
    
    def copy(self) -> 'MetroBALabel':
        """Create a copy of the label."""
        return MetroBALabel(
            station=self.station,
            time=self.time,
            load=self.load,
            cost=self.cost,
            big_arcs=[arc.copy() if hasattr(arc, 'copy') else arc for arc in self.big_arcs],
            cargo_up=self.cargo_up,
            cargo_down=self.cargo_down
        )
    
    def dominates(self, other: 'MetroBALabel') -> bool:
        """Check if this label dominates another."""
        if self.station != other.station:
            return False
        
        return (self.time <= other.time and 
                self.load <= other.load and
                self.cost <= other.cost and
                self.cargo_up >= other.cargo_up and
                self.cargo_down >= other.cargo_down and
                (self.time < other.time or self.load < other.load or 
                 self.cost < other.cost or self.cargo_up > other.cargo_up or
                 self.cargo_down > other.cargo_down))

@dataclass
class BapNode:
    """
    Branch-and-Price tree node.
    """
    id: int
    parent_id: Optional[int]
    depth: int
    columns: List[Column] = field(default_factory=list)
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    lower_bound: float = float('-inf')
    upper_bound: float = float('inf')
    is_integer: bool = False
    is_feasible: bool = True
    branching_variable: Optional[str] = None
    branching_value: Optional[float] = None
    branching_direction: Optional[str] = None  # "=0" or "=1"
    
    def add_branching_constraint(self, variable: str, value: float, direction: str):
        """Add a branching constraint to this node."""
        constraint = {
            'type': 'branching',
            'variable': variable,
            'value': value,
            'direction': direction
        }
        self.constraints.append(constraint)
        self.branching_variable = variable
        self.branching_value = value
        self.branching_direction = direction

@dataclass
class NetworkNode:
    """
    Network node representing a location in the transportation network.
    """
    id: int
    x_coord: float
    y_coord: float
    node_type: str  # "depot", "customer", "metro_station"
    demand: float = 0.0
    time_window_start: int = 0  # in minutes
    time_window_end: int = 1440  # in minutes (24 hours)
    service_time: int = 0  # in minutes
    
@dataclass
class Vehicle:
    """
    Vehicle information.
    """
    id: str
    type: VehicleType
    capacity: float
    speed: float  # km/h
    cost_per_km: float = 0.0
    cost_per_hour: float = 0.0
    energy_capacity: Optional[float] = None  # for drones
    energy_consumption_rate: Optional[float] = None  # per km for drones
    
@dataclass
class MetroTimetable:
    """
    Metro timetable entry.
    """
    line_id: str
    direction: str  # "up" or "down"
    station_id: int
    arrival_time: int  # in minutes
    departure_time: int  # in minutes
    
@dataclass
class Demand:
    """
    Transportation demand.
    """
    id: str
    origin: int
    destination: int
    quantity: float
    time_window_start: int
    time_window_end: int
    priority: int = 1

@dataclass
class SolutionStats:
    """
    Solution statistics for reporting.
    """
    total_cost: float = 0.0
    total_revenue: float = 0.0
    total_demand_served: float = 0.0
    total_demand_unserved: float = 0.0
    num_drone_routes: int = 0
    num_truck_routes: int = 0
    num_metro_schedules: int = 0
    total_runtime: float = 0.0
    num_iterations: int = 0
    final_gap: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Excel output."""
        return {
            'Total Cost': self.total_cost,
            'Total Revenue': self.total_revenue,
            'Total Demand Served': self.total_demand_served,
            'Total Demand Unserved': self.total_demand_unserved,
            'Number of Drone Routes': self.num_drone_routes,
            'Number of Truck Routes': self.num_truck_routes,
            'Number of Metro Schedules': self.num_metro_schedules,
            'Total Runtime (seconds)': self.total_runtime,
            'Number of Iterations': self.num_iterations,
            'Final Gap (%)': self.final_gap * 100
        }
