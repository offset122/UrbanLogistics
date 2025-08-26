"""
Algorithm 2: Unidirectional Label-setting Algorithm for Drones (ULA-D)
Basic drone routing with energy and capacity constraints.
"""

from typing import List, Dict, Any, Optional, Tuple
import heapq
from collections import defaultdict

from data_structures import DroneLabel, Column, ColumnType, VehicleType
from data_loader import DataLoader
from utils import log_message, calculate_reduced_cost
from config import DEFAULT_DRONE_CAPACITY, DEFAULT_DRONE_ENERGY

class ULADSolver:
    """
    Unidirectional Label-setting Algorithm for Drone routing.
    Solves Resource Constrained Shortest Path Problem (RCSPP) for drones.
    """
    
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.depot_node = 0  # Assume depot is node 0
        self.drone_capacity = DEFAULT_DRONE_CAPACITY
        self.drone_energy = DEFAULT_DRONE_ENERGY
        
    def solve_pricing(self, dual_values: Dict[str, float]) -> List[Column]:
        """
        Solve drone pricing problem using label-setting algorithm.
        
        Args:
            dual_values: Dual values from RMP solution
            
        Returns:
            List of columns with negative reduced cost
        """
        log_message("Starting ULA-D pricing algorithm")
        
        negative_cost_columns = []
        
        # Get available drones
        drones = [v for v in self.data_loader.vehicles.values() 
                 if v.type == VehicleType.DRONE]
        
        for drone in drones:
            columns = self._solve_for_drone(drone, dual_values)
            negative_cost_columns.extend(columns)
        
        log_message(f"ULA-D found {len(negative_cost_columns)} columns with negative reduced cost")
        return negative_cost_columns
    
    def _solve_for_drone(self, drone, dual_values: Dict[str, float]) -> List[Column]:
        """
        Solve pricing for a specific drone.
        
        Args:
            drone: Drone vehicle object
            dual_values: Dual values from RMP
            
        Returns:
            List of columns with negative reduced cost
        """
        # Initialize labels
        labels = defaultdict(list)  # node_id -> list of labels
        processed = set()  # (node, time, load, energy) tuples for processed states
        priority_queue = []  # Min-heap for label selection
        
        # Create initial label at depot
        initial_label = DroneLabel(
            node=self.depot_node,
            time=0,
            load=0.0,
            energy=drone.energy_capacity or self.drone_energy,
            cost=0.0,
            path=[self.depot_node]
        )
        
        labels[self.depot_node].append(initial_label)
        heapq.heappush(priority_queue, (0.0, 0, initial_label))  # (cost, time, label)
        
        negative_columns = []
        
        while priority_queue:
            current_cost, current_time, current_label = heapq.heappop(priority_queue)
            
            # Create state key for dominance
            state_key = (current_label.node, 
                        current_label.time // 15,  # Discretize time to 15-min intervals
                        int(current_label.load), 
                        int(current_label.energy))
            
            if state_key in processed:
                continue
            processed.add(state_key)
            
            # Check if we can return to depot
            if (current_label.node != self.depot_node and 
                len(current_label.path) > 1):
                
                return_column = self._try_return_to_depot(current_label, drone, dual_values)
                if return_column and return_column.reduced_cost < -1e-6:
                    negative_columns.append(return_column)
            
            # Extend to neighboring nodes
            self._extend_label(current_label, drone, dual_values, labels, priority_queue)
        
        return negative_columns
    
    def _extend_label(self, label: DroneLabel, drone, dual_values: Dict[str, float],
                     labels: Dict[int, List[DroneLabel]], priority_queue: List):
        """
        Extend a label to neighboring nodes.
        
        Args:
            label: Current label to extend
            drone: Drone vehicle object
            dual_values: Dual values from RMP
            labels: Dictionary of labels by node
            priority_queue: Priority queue for label processing
        """
        current_node = label.node
        
        # Try extending to all other nodes
        for next_node_id, next_node in self.data_loader.nodes.items():
            if next_node_id == current_node or next_node_id in label.path:
                continue  # Skip same node and already visited nodes
            
            # Calculate travel time, distance, and energy consumption
            travel_time = self.data_loader.get_travel_time(
                current_node, next_node_id, VehicleType.DRONE
            )
            travel_distance = self.data_loader.get_distance(current_node, next_node_id)
            
            # Energy consumption
            energy_consumption = travel_distance * (drone.energy_consumption_rate or 2.0)
            
            # Travel cost
            travel_cost = travel_distance * drone.cost_per_km + \
                         (travel_time / 60.0) * drone.cost_per_hour
            
            # Calculate arrival time
            arrival_time = label.time + travel_time + \
                          self.data_loader.nodes[current_node].service_time
            
            # Check time window constraints
            if (arrival_time < next_node.time_window_start or 
                arrival_time > next_node.time_window_end):
                continue
            
            # Check capacity constraints
            new_load = label.load + next_node.demand
            if new_load > drone.capacity:
                continue
            
            # Check energy constraints
            new_energy = label.energy - energy_consumption
            if new_energy < 0:
                continue
            
            # Check if enough energy to return to depot
            return_distance = self.data_loader.get_distance(next_node_id, self.depot_node)
            return_energy = return_distance * (drone.energy_consumption_rate or 2.0)
            if new_energy < return_energy:
                continue
            
            # Calculate cargo values based on dual values
            cargo_up, cargo_down = self._calculate_cargo_values(
                next_node_id, next_node.demand, dual_values
            )
            
            # Create new label
            new_label = DroneLabel(
                node=next_node_id,
                time=arrival_time + next_node.service_time,
                load=new_load,
                energy=new_energy,
                cost=label.cost + travel_cost,
                path=label.path + [next_node_id],
                cargo_up=label.cargo_up + cargo_up,
                cargo_down=label.cargo_down + cargo_down
            )
            
            # Check dominance
            if self._is_dominated(new_label, labels[next_node_id]):
                continue
            
            # Remove dominated labels
            labels[next_node_id] = [l for l in labels[next_node_id] 
                                   if not new_label.dominates(l)]
            
            # Add new label
            labels[next_node_id].append(new_label)
            heapq.heappush(priority_queue, (new_label.cost, new_label.time, new_label))
    
    def _calculate_cargo_values(self, node_id: int, demand: float, 
                               dual_values: Dict[str, float]) -> Tuple[float, float]:
        """
        Calculate cargo values for inventory constraints.
        
        Args:
            node_id: Node ID
            demand: Demand at the node
            dual_values: Dual values from RMP
            
        Returns:
            Tuple of (cargo_up, cargo_down) values
        """
        cargo_up = 0.0
        cargo_down = 0.0
        
        # Check if this is a metro station
        node = self.data_loader.nodes.get(node_id)
        if node and node.node_type == "metro_station":
            # Calculate cargo contribution based on dual values
            psi_up = dual_values.get(f"psi_up_{node_id}", 0.0)
            psi_down = dual_values.get(f"psi_down_{node_id}", 0.0)
            
            # Simple heuristic: allocate demand based on dual values
            if psi_up > psi_down:
                cargo_up = demand
            else:
                cargo_down = demand
        
        return cargo_up, cargo_down
    
    def _is_dominated(self, new_label: DroneLabel, existing_labels: List[DroneLabel]) -> bool:
        """
        Check if the new label is dominated by any existing label.
        
        Args:
            new_label: New label to check
            existing_labels: List of existing labels at the same node
            
        Returns:
            True if new label is dominated, False otherwise
        """
        for existing_label in existing_labels:
            if existing_label.dominates(new_label):
                return True
        return False
    
    def _try_return_to_depot(self, label: DroneLabel, drone, 
                            dual_values: Dict[str, float]) -> Optional[Column]:
        """
        Try to create a complete route by returning to depot.
        
        Args:
            label: Current label
            drone: Drone vehicle object
            dual_values: Dual values from RMP
            
        Returns:
            Column object if route is feasible, None otherwise
        """
        current_node = label.node
        
        # Calculate return travel
        return_time = self.data_loader.get_travel_time(
            current_node, self.depot_node, VehicleType.DRONE
        )
        return_distance = self.data_loader.get_distance(current_node, self.depot_node)
        return_energy = return_distance * (drone.energy_consumption_rate or 2.0)
        return_cost = return_distance * drone.cost_per_km + \
                     (return_time / 60.0) * drone.cost_per_hour
        
        # Check feasibility
        if label.energy < return_energy:
            return None
        
        total_time = label.time + return_time
        if total_time > 1440:  # 24 hours limit
            return None
        
        # Calculate total route cost
        total_cost = label.cost + return_cost
        
        # Create column details
        column_details = self._create_column_details(label, drone)
        
        # Calculate reduced cost
        reduced_cost = calculate_reduced_cost(total_cost, dual_values, column_details)
        
        # Create column
        column = Column(
            id=f"drone_{drone.id}_{len(label.path)}_{label.time}",
            type=ColumnType.DRONE_ROUTE,
            vehicle_type=VehicleType.DRONE,
            direct_cost=total_cost,
            reduced_cost=reduced_cost,
            details=column_details
        )
        
        return column
    
    def _create_column_details(self, label: DroneLabel, drone) -> Dict[str, Any]:
        """
        Create detailed information for the column.
        
        Args:
            label: Final label for the route
            drone: Drone vehicle object
            
        Returns:
            Dictionary with column details
        """
        details = {
            'a_ip': {},  # Demand coverage
            'delta_up': {},  # Inventory up
            'delta_down': {},  # Inventory down
            'resource_usage': {},  # Resource usage per time slice
            'route': label.path + [self.depot_node],  # Complete route
            'timing': [],  # Timing information
            'cargo': {
                'cargo_up': label.cargo_up,
                'cargo_down': label.cargo_down
            },
            'energy_profile': []  # Energy consumption profile
        }
        
        # Set demand coverage for served customers
        for node_id in label.path[1:]:  # Skip depot
            node = self.data_loader.nodes.get(node_id)
            if node and node.demand > 0:
                # Find corresponding demand
                for demand_id, demand in self.data_loader.demands.items():
                    if demand.destination == node_id:
                        details['a_ip'][demand_id] = 1.0
                        break
        
        # Set inventory coefficients for metro stations
        for node_id in label.path[1:]:  # Skip depot
            node = self.data_loader.nodes.get(node_id)
            if node and node.node_type == "metro_station":
                details['delta_up'][node_id] = label.cargo_up
                details['delta_down'][node_id] = label.cargo_down
        
        # Calculate resource usage per time slice
        from utils import create_time_slices
        from config import RESOURCE_TIME_SLICE
        
        time_slices = create_time_slices(0, 1440, RESOURCE_TIME_SLICE)
        
        # Simple resource usage: 1 if drone is active during time slice
        route_start_time = 0
        route_end_time = label.time
        
        for slice_start, slice_end in time_slices:
            if (route_start_time < slice_end and route_end_time > slice_start):
                details['resource_usage'][f"{slice_start}_{slice_end}"] = 1.0
        
        # Create energy profile
        current_energy = drone.energy_capacity or self.drone_energy
        current_time = 0
        
        for i in range(len(label.path) - 1):
            from_node = label.path[i]
            to_node = label.path[i + 1]
            
            travel_distance = self.data_loader.get_distance(from_node, to_node)
            energy_consumption = travel_distance * (drone.energy_consumption_rate or 2.0)
            travel_time = self.data_loader.get_travel_time(
                from_node, to_node, VehicleType.DRONE
            )
            
            current_energy -= energy_consumption
            current_time += travel_time
            
            details['energy_profile'].append({
                'node': to_node,
                'time': current_time,
                'energy': current_energy,
                'load': sum(self.data_loader.nodes[n].demand for n in label.path[:i+2])
            })
        
        return details
