"""
Data loading and preprocessing module for the Branch-and-Price solver.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import warnings

from data_structures import (
    NetworkNode, Vehicle, MetroTimetable, Demand, VehicleType
)
from utils import time_to_minutes, euclidean_distance, travel_time, normalize_header, log_message
from config import (
    DATA_DIR, HEADER_MAPPINGS, DEFAULT_DRONE_CAPACITY, DEFAULT_TRUCK_CAPACITY,
    DEFAULT_DRONE_ENERGY, DEFAULT_DRONE_SPEED, DEFAULT_TRUCK_SPEED
)

warnings.filterwarnings('ignore', category=UserWarning)

class DataLoader:
    """
    Handles loading and preprocessing of all input data.
    """
    
    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self.nodes: Dict[int, NetworkNode] = {}
        self.vehicles: Dict[str, Vehicle] = {}
        self.demands: Dict[str, Demand] = {}
        self.metro_timetables: List[MetroTimetable] = []
        self.distance_matrix: np.ndarray = None
        self.time_matrix: np.ndarray = None
        self.metro_lookup: Dict[str, List[MetroTimetable]] = {}
        self.parameters: Dict[str, Any] = {}
        
    def load_all_data(self) -> bool:
        """
        Load all required data files.
        
        Returns:
            True if all data loaded successfully, False otherwise
        """
        try:
            log_message("Starting data loading process...")
            
            # Load each required file
            self._load_nodes()
            self._load_demands()
            self._load_vehicles()
            self._load_metro_timetable()
            self._load_parameters()
            
            # Build derived data structures
            self._build_distance_matrix()
            self._build_time_matrix()
            self._build_metro_lookup()
            
            log_message("Data loading completed successfully")
            return True
            
        except Exception as e:
            log_message(f"Error during data loading: {str(e)}", "ERROR")
            return False
    
    def _load_nodes(self):
        """Load network nodes from Excel file."""
        file_path = os.path.join(self.data_dir, "nodes.xlsx")
        
        if not os.path.exists(file_path):
            # Create sample structure if file doesn't exist
            log_message(f"Nodes file not found at {file_path}, creating default structure", "WARNING")
            self._create_default_nodes()
            return
        
        try:
            df = pd.read_excel(file_path)
            df.columns = [normalize_header(col) for col in df.columns]
            
            for _, row in df.iterrows():
                node = NetworkNode(
                    id=int(row.get('node_id', row.name)),
                    x_coord=float(row.get('x_coord', 0)),
                    y_coord=float(row.get('y_coord', 0)),
                    node_type=str(row.get('node_type', 'customer')),
                    demand=float(row.get('demand', 0)),
                    time_window_start=time_to_minutes(row.get('time_window_start', '00:00')),
                    time_window_end=time_to_minutes(row.get('time_window_end', '23:59')),
                    service_time=int(row.get('service_time', 0))
                )
                self.nodes[node.id] = node
                
            log_message(f"Loaded {len(self.nodes)} nodes")
            
        except Exception as e:
            log_message(f"Error loading nodes: {str(e)}", "ERROR")
            self._create_default_nodes()
    
    def _create_default_nodes(self):
        """Create default node structure."""
        # Create a simple network with depot and customer nodes
        default_nodes = [
            NetworkNode(0, 0, 0, "depot", 0, 0, 1440, 0),
            NetworkNode(1, 10, 0, "customer", 50, 480, 1080, 15),
            NetworkNode(2, 0, 10, "customer", 30, 540, 1020, 15),
            NetworkNode(3, 5, 5, "metro_station", 0, 0, 1440, 5),
            NetworkNode(4, 15, 10, "customer", 40, 600, 960, 15),
            NetworkNode(5, 8, 12, "customer", 25, 420, 1140, 15)
        ]
        
        for node in default_nodes:
            self.nodes[node.id] = node
            
        log_message(f"Created {len(self.nodes)} default nodes")
    
    def _load_demands(self):
        """Load transportation demands from Excel file."""
        file_path = os.path.join(self.data_dir, "demands.xlsx")
        
        if not os.path.exists(file_path):
            log_message(f"Demands file not found at {file_path}, creating default structure", "WARNING")
            self._create_default_demands()
            return
        
        try:
            df = pd.read_excel(file_path)
            df.columns = [normalize_header(col) for col in df.columns]
            
            for _, row in df.iterrows():
                demand = Demand(
                    id=str(row.get('demand_id', f"demand_{row.name}")),
                    origin=int(row.get('origin', 0)),
                    destination=int(row.get('destination', 1)),
                    quantity=float(row.get('quantity', 0)),
                    time_window_start=time_to_minutes(row.get('time_window_start', '00:00')),
                    time_window_end=time_to_minutes(row.get('time_window_end', '23:59')),
                    priority=int(row.get('priority', 1))
                )
                self.demands[demand.id] = demand
                
            log_message(f"Loaded {len(self.demands)} demands")
            
        except Exception as e:
            log_message(f"Error loading demands: {str(e)}", "ERROR")
            self._create_default_demands()
    
    def _create_default_demands(self):
        """Create default demand structure."""
        default_demands = [
            Demand("d1", 0, 1, 50, 480, 1080, 1),
            Demand("d2", 0, 2, 30, 540, 1020, 1),
            Demand("d3", 0, 4, 40, 600, 960, 1),
            Demand("d4", 0, 5, 25, 420, 1140, 1)
        ]
        
        for demand in default_demands:
            self.demands[demand.id] = demand
            
        log_message(f"Created {len(self.demands)} default demands")
    
    def _load_vehicles(self):
        """Load vehicle information from Excel file."""
        file_path = os.path.join(self.data_dir, "vehicles.xlsx")
        
        if not os.path.exists(file_path):
            log_message(f"Vehicles file not found at {file_path}, creating default structure", "WARNING")
            self._create_default_vehicles()
            return
        
        try:
            df = pd.read_excel(file_path)
            df.columns = [normalize_header(col) for col in df.columns]
            
            for _, row in df.iterrows():
                vehicle_type_str = str(row.get('vehicle_type', 'truck')).lower()
                vehicle_type = VehicleType.TRUCK
                
                if 'drone' in vehicle_type_str:
                    vehicle_type = VehicleType.DRONE
                elif 'metro' in vehicle_type_str:
                    vehicle_type = VehicleType.METRO
                
                vehicle = Vehicle(
                    id=str(row.get('vehicle_id', f"vehicle_{row.name}")),
                    type=vehicle_type,
                    capacity=float(row.get('capacity', DEFAULT_TRUCK_CAPACITY)),
                    speed=float(row.get('speed', DEFAULT_TRUCK_SPEED)),
                    cost_per_km=float(row.get('cost_per_km', 1.0)),
                    cost_per_hour=float(row.get('cost_per_hour', 50.0))
                )
                
                if vehicle_type == VehicleType.DRONE:
                    vehicle.energy_capacity = float(row.get('energy_capacity', DEFAULT_DRONE_ENERGY))
                    vehicle.energy_consumption_rate = float(row.get('energy_consumption_rate', 2.0))
                
                self.vehicles[vehicle.id] = vehicle
                
            log_message(f"Loaded {len(self.vehicles)} vehicles")
            
        except Exception as e:
            log_message(f"Error loading vehicles: {str(e)}", "ERROR")
            self._create_default_vehicles()
    
    def _create_default_vehicles(self):
        """Create default vehicle structure."""
        default_vehicles = [
            Vehicle("drone1", VehicleType.DRONE, DEFAULT_DRONE_CAPACITY, DEFAULT_DRONE_SPEED, 2.0, 100.0, DEFAULT_DRONE_ENERGY, 2.0),
            Vehicle("drone2", VehicleType.DRONE, DEFAULT_DRONE_CAPACITY, DEFAULT_DRONE_SPEED, 2.0, 100.0, DEFAULT_DRONE_ENERGY, 2.0),
            Vehicle("truck1", VehicleType.TRUCK, DEFAULT_TRUCK_CAPACITY, DEFAULT_TRUCK_SPEED, 1.0, 50.0),
            Vehicle("truck2", VehicleType.TRUCK, DEFAULT_TRUCK_CAPACITY, DEFAULT_TRUCK_SPEED, 1.0, 50.0)
        ]
        
        for vehicle in default_vehicles:
            self.vehicles[vehicle.id] = vehicle
            
        log_message(f"Created {len(self.vehicles)} default vehicles")
    
    def _load_metro_timetable(self):
        """Load metro timetable from Excel file."""
        file_path = os.path.join(self.data_dir, "metro_timetable.xlsx")
        
        if not os.path.exists(file_path):
            log_message(f"Metro timetable file not found at {file_path}, creating default structure", "WARNING")
            self._create_default_metro_timetable()
            return
        
        try:
            df = pd.read_excel(file_path)
            df.columns = [normalize_header(col) for col in df.columns]
            
            for _, row in df.iterrows():
                timetable_entry = MetroTimetable(
                    line_id=str(row.get('line_id', 'line1')),
                    direction=str(row.get('direction', 'up')),
                    station_id=int(row.get('station_id', 0)),
                    arrival_time=time_to_minutes(row.get('arrival_time', '00:00')),
                    departure_time=time_to_minutes(row.get('departure_time', '00:00'))
                )
                self.metro_timetables.append(timetable_entry)
                
            log_message(f"Loaded {len(self.metro_timetables)} metro timetable entries")
            
        except Exception as e:
            log_message(f"Error loading metro timetable: {str(e)}", "ERROR")
            self._create_default_metro_timetable()
    
    def _create_default_metro_timetable(self):
        """Create default metro timetable."""
        # Create simple timetable for metro station (node 3)
        for hour in range(6, 23):  # 6 AM to 10 PM
            for minute in [0, 30]:  # Every 30 minutes
                time_minutes = hour * 60 + minute
                
                # Up direction
                self.metro_timetables.append(
                    MetroTimetable("line1", "up", 3, time_minutes, time_minutes + 2)
                )
                
                # Down direction  
                self.metro_timetables.append(
                    MetroTimetable("line1", "down", 3, time_minutes + 15, time_minutes + 17)
                )
        
        log_message(f"Created {len(self.metro_timetables)} default metro timetable entries")
    
    def _load_parameters(self):
        """Load system parameters from Excel file."""
        file_path = os.path.join(self.data_dir, "parameters.xlsx")
        
        # Default parameters
        self.parameters = {
            'drone_capacity': DEFAULT_DRONE_CAPACITY,
            'truck_capacity': DEFAULT_TRUCK_CAPACITY,
            'drone_energy': DEFAULT_DRONE_ENERGY,
            'drone_speed': DEFAULT_DRONE_SPEED,
            'truck_speed': DEFAULT_TRUCK_SPEED,
            'metro_capacity': 5000.0,
            'resource_cost_drone': 100.0,
            'resource_cost_truck': 50.0,
            'resource_cost_pilot': 75.0,
            'inventory_cost_up': 10.0,
            'inventory_cost_down': 10.0
        }
        
        if not os.path.exists(file_path):
            log_message(f"Parameters file not found at {file_path}, using defaults", "WARNING")
            return
        
        try:
            df = pd.read_excel(file_path)
            df.columns = [normalize_header(col) for col in df.columns]
            
            for _, row in df.iterrows():
                param_name = str(row.get('parameter', ''))
                param_value = row.get('value', 0)
                
                if param_name:
                    self.parameters[param_name] = float(param_value)
                    
            log_message(f"Loaded {len(self.parameters)} parameters")
            
        except Exception as e:
            log_message(f"Error loading parameters: {str(e)}", "ERROR")
    
    def _build_distance_matrix(self):
        """Build distance matrix between all nodes."""
        n_nodes = len(self.nodes)
        self.distance_matrix = np.zeros((n_nodes, n_nodes))
        
        node_ids = sorted(self.nodes.keys())
        
        for i, node_i in enumerate(node_ids):
            for j, node_j in enumerate(node_ids):
                if i == j:
                    self.distance_matrix[i, j] = 0
                else:
                    coord_i = (self.nodes[node_i].x_coord, self.nodes[node_i].y_coord)
                    coord_j = (self.nodes[node_j].x_coord, self.nodes[node_j].y_coord)
                    self.distance_matrix[i, j] = euclidean_distance(coord_i, coord_j)
        
        log_message(f"Built distance matrix for {n_nodes} nodes")
    
    def _build_time_matrix(self):
        """Build travel time matrices for different vehicle types."""
        n_nodes = len(self.nodes)
        
        # Create time matrices for each vehicle type
        self.time_matrix = {
            VehicleType.DRONE: np.zeros((n_nodes, n_nodes)),
            VehicleType.TRUCK: np.zeros((n_nodes, n_nodes))
        }
        
        node_ids = sorted(self.nodes.keys())
        
        for i, node_i in enumerate(node_ids):
            for j, node_j in enumerate(node_ids):
                if i != j:
                    distance = self.distance_matrix[i, j]
                    
                    # Drone travel time
                    drone_time = travel_time(distance, DEFAULT_DRONE_SPEED)
                    self.time_matrix[VehicleType.DRONE][i, j] = drone_time
                    
                    # Truck travel time  
                    truck_time = travel_time(distance, DEFAULT_TRUCK_SPEED)
                    self.time_matrix[VehicleType.TRUCK][i, j] = truck_time
        
        log_message(f"Built travel time matrices for {n_nodes} nodes")
    
    def _build_metro_lookup(self):
        """Build lookup tables for metro timetables."""
        self.metro_lookup = {}
        
        for entry in self.metro_timetables:
            key = f"{entry.line_id}_{entry.direction}_{entry.station_id}"
            
            if key not in self.metro_lookup:
                self.metro_lookup[key] = []
            
            self.metro_lookup[key].append(entry)
        
        # Sort by departure time for each lookup key
        for key in self.metro_lookup:
            self.metro_lookup[key].sort(key=lambda x: x.departure_time)
        
        log_message(f"Built metro lookup tables with {len(self.metro_lookup)} entries")
    
    def get_node_index(self, node_id: int) -> int:
        """Get matrix index for a node ID."""
        node_ids = sorted(self.nodes.keys())
        return node_ids.index(node_id)
    
    def get_travel_time(self, from_node: int, to_node: int, vehicle_type: VehicleType) -> int:
        """Get travel time between two nodes for a specific vehicle type."""
        from_idx = self.get_node_index(from_node)
        to_idx = self.get_node_index(to_node)
        
        if vehicle_type in self.time_matrix:
            return int(self.time_matrix[vehicle_type][from_idx, to_idx])
        
        return 0
    
    def get_distance(self, from_node: int, to_node: int) -> float:
        """Get distance between two nodes."""
        from_idx = self.get_node_index(from_node)
        to_idx = self.get_node_index(to_node)
        return self.distance_matrix[from_idx, to_idx]
    
    def get_metro_departures(self, line_id: str, direction: str, station_id: int, 
                           after_time: int) -> List[MetroTimetable]:
        """Get metro departures after a specific time."""
        key = f"{line_id}_{direction}_{station_id}"
        
        if key not in self.metro_lookup:
            return []
        
        departures = []
        for entry in self.metro_lookup[key]:
            if entry.departure_time >= after_time:
                departures.append(entry)
        
        return departures
