"""
Algorithm 5: Bidirectional Label-setting Algorithm for Drones (BLA-D)
Optimized drone routing with bidirectional search and advanced energy management.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
import heapq
from collections import defaultdict
import math

from data_structures import DroneLabel, Column, ColumnType, VehicleType
from data_loader import DataLoader
from utils import log_message, calculate_reduced_cost
from config import DEFAULT_DRONE_CAPACITY, DEFAULT_DRONE_ENERGY

class BLADSolver:
    """
    Bidirectional Label-setting Algorithm for Drone routing.
    Enhanced version with bidirectional search, energy optimization, and advanced dominance.
    """
    
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.depot_node = 0
        self.drone_capacity = DEFAULT_DRONE_CAPACITY
        self.drone_energy = DEFAULT_DRONE_ENERGY
        
    def solve_pricing(self, dual_values: Dict[str, float]) -> List[Column]:
        """
        Solve drone pricing problem using bidirectional label-setting algorithm.
        
        Args:
            dual_values: Dual values from RMP solution
            
        Returns:
            List of columns with negative reduced cost
        """
        log_message("Starting BLA-D pricing algorithm")
        
        negative_cost_columns = []
        
        # Get available drones
        drones = [v for v in self.data_loader.vehicles.values() 
                 if v.type == VehicleType.DRONE]
        
        for drone in drones:
            columns = self._solve_for_drone_bidirectional(drone, dual_values)
            negative_cost_columns.extend(columns)
        
        log_message(f"BLA-D found {len(negative_cost_columns)} columns with negative reduced cost")
        return negative_cost_columns
    
    def _solve_for_drone_bidirectional(self, drone, dual_values: Dict[str, float]) -> List[Column]:
        """
        Solve pricing for a specific drone using bidirectional search.
        
        Args:
            drone: Drone vehicle object
            dual_values: Dual values from RMP
            
        Returns:
            List of columns with negative reduced cost
        """
        # Forward search from depot
        forward_labels = self._forward_search(drone, dual_values)
        
        # Backward search to depot
        backward_labels = self._backward_search(drone, dual_values)
        
        # Join forward and backward labels
        negative_columns = self._join_labels(forward_labels, backward_labels, drone, dual_values)
        
        # Also try direct returns for forward labels
        direct_columns = self._try_direct_returns(forward_labels, drone, dual_values)
        negative_columns.extend(direct_columns)
        
        return negative_columns
    
    def _forward_search(self, drone, dual_values: Dict[str, float]) -> Dict[int, List[DroneLabel]]:
        """
        Forward search from depot with energy-aware extensions.
        
        Args:
            drone: Drone vehicle object
            dual_values: Dual values from RMP
            
        Returns:
            Dictionary of forward labels by node
        """
        labels = defaultdict(list)
        processed = set()
        priority_queue = []
        
        # Initial label at depot
        initial_label = DroneLabel(
            node=self.depot_node,
            time=0,
            load=0.0,
            energy=drone.energy_capacity or self.drone_energy,
            cost=0.0,
            path=[self.depot_node]
        )
        
        labels[self.depot_node].append(initial_label)
        heapq.heappush(priority_queue, (0.0, 0, 0, initial_label))  # (cost, -energy, time, label)
        
        while priority_queue:
            current_cost, neg_energy, current_time, current_label = heapq.heappop(priority_queue)
            
            # Enhanced state key with energy discretization
            state_key = (current_label.node, 
                        current_label.time // 15,  # 15-min intervals
                        int(current_label.load), 
                        int(current_label.energy // 5))  # 5-unit energy intervals
            
            if state_key in processed:
                continue
            processed.add(state_key)
            
            # Extend to neighboring nodes
            self._extend_forward_label(current_label, drone, dual_values, labels, priority_queue)
        
        return dict(labels)
    
    def _backward_search(self, drone, dual_values: Dict[str, float]) -> Dict[int, List[DroneLabel]]:
        """
        Backward search to depot with energy constraints.
        
        Args:
            drone: Drone vehicle object
            dual_values: Dual values from RMP
            
        Returns:
            Dictionary of backward labels by node
        """
        labels = defaultdict(list)
        processed = set()
        priority_queue = []
        
        # Create initial backward labels at customer nodes
        for node_id, node in self.data_loader.nodes.items():
            if node_id != self.depot_node and node.node_type == "customer":
                # Calculate minimum energy needed to reach this node from depot
                depot_distance = self.data_loader.get_distance(self.depot_node, node_id)
                min_energy_needed = depot_distance * (drone.energy_consumption_rate or 2.0)
                
                # Only create label if reachable
                max_energy = drone.energy_capacity or self.drone_energy
                if min_energy_needed < max_energy * 0.8:  # Leave 20% margin
                    
                    # Start with energy after serving this customer
                    remaining_energy = max_energy - min_energy_needed
                    
                    initial_label = DroneLabel(
                        node=node_id,
                        time=node.time_window_end,  # Start from latest possible time
                        load=0.0,  # Backward: start empty
                        energy=remaining_energy,
                        cost=0.0,
                        path=[node_id]
                    )
                    
                    labels[node_id].append(initial_label)
                    heapq.heappush(priority_queue, (0.0, -remaining_energy, -node.time_window_end, initial_label))
        
        while priority_queue:
            current_cost, neg_energy, neg_time, current_label = heapq.heappop(priority_queue)
            
            # State key for backward search
            state_key = (current_label.node, 
                        current_label.time // 15,
                        int(current_label.load), 
                        int(current_label.energy // 5))
            
            if state_key in processed:
                continue
            processed.add(state_key)
            
            # Extend backward
            self._extend_backward_label(current_label, drone, dual_values, labels, priority_queue)
        
        return dict(labels)
    
    def _extend_forward_label(self, label: DroneLabel, drone, dual_values: Dict[str, float],
                             labels: Dict[int, List[DroneLabel]], priority_queue: List):
        """
        Extend a forward label with advanced energy management.
        
        Args:
            label: Current label to extend
            drone: Drone vehicle object
            dual_values: Dual values from RMP
            labels: Dictionary of labels by node
            priority_queue: Priority queue for label processing
        """
        current_node = label.node
        
        # Limit path length for efficiency
        if len(label.path) > 5:
            return
        
        for next_node_id, next_node in self.data_loader.nodes.items():
            if (next_node_id == current_node or 
                next_node_id in label.path or
                next_node_id == self.depot_node):  # Don't return to depot in forward
                continue
            
            # Calculate travel parameters
            travel_time = self.data_loader.get_travel_time(
                current_node, next_node_id, VehicleType.DRONE
            )
            travel_distance = self.data_loader.get_distance(current_node, next_node_id)
            
            # Energy consumption with efficiency factor based on load
            base_consumption = travel_distance * (drone.energy_consumption_rate or 2.0)
            load_factor = 1.0 + (label.load / drone.capacity) * 0.2  # Up to 20% more consumption when loaded
            energy_consumption = base_consumption * load_factor
            
            # Travel cost with time-of-day factor
            base_cost = travel_distance * drone.cost_per_km + (travel_time / 60.0) * drone.cost_per_hour
            time_factor = self._get_time_cost_factor(label.time)
            travel_cost = base_cost * time_factor
            
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
            
            # Check if enough energy to return to depot (with safety margin)
            return_distance = self.data_loader.get_distance(next_node_id, self.depot_node)
            return_energy_needed = return_distance * (drone.energy_consumption_rate or 2.0) * 1.1  # 10% safety
            if new_energy < return_energy_needed:
                continue
            
            # Calculate cargo values with enhanced heuristics
            cargo_up, cargo_down = self._calculate_enhanced_cargo_values(
                next_node_id, next_node.demand, dual_values, arrival_time
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
            
            # Enhanced dominance check
            if self._is_dominated_enhanced(new_label, labels[next_node_id], forward=True):
                continue
            
            # Remove dominated labels
            labels[next_node_id] = [l for l in labels[next_node_id] 
                                   if not self._dominates_enhanced(new_label, l)]
            
            # Add new label
            labels[next_node_id].append(new_label)
            priority_key = (new_label.cost, -new_label.energy, new_label.time, new_label)
            heapq.heappush(priority_queue, priority_key)
    
    def _extend_backward_label(self, label: DroneLabel, drone, dual_values: Dict[str, float],
                              labels: Dict[int, List[DroneLabel]], priority_queue: List):
        """
        Extend a backward label.
        
        Args:
            label: Current label to extend
            drone: Drone vehicle object
            dual_values: Dual values from RMP
            labels: Dictionary of labels by node
            priority_queue: Priority queue for label processing
        """
        current_node = label.node
        
        # Limit path length
        if len(label.path) > 5:
            return
        
        for prev_node_id, prev_node in self.data_loader.nodes.items():
            if (prev_node_id == current_node or 
                prev_node_id in label.path):
                continue
            
            # Calculate backward travel
            travel_time = self.data_loader.get_travel_time(
                prev_node_id, current_node, VehicleType.DRONE
            )
            travel_distance = self.data_loader.get_distance(prev_node_id, current_node)
            
            # Energy and cost calculations
            base_consumption = travel_distance * (drone.energy_consumption_rate or 2.0)
            load_factor = 1.0 + (label.load / drone.capacity) * 0.2
            energy_consumption = base_consumption * load_factor
            
            base_cost = travel_distance * drone.cost_per_km + (travel_time / 60.0) * drone.cost_per_hour
            
            # Backward time calculation
            departure_time = label.time - travel_time - prev_node.service_time
            
            # Check constraints
            if (departure_time < prev_node.time_window_start or 
                departure_time > prev_node.time_window_end):
                continue
            
            # Backward capacity (we're adding demand we need to carry)
            new_load = label.load + prev_node.demand
            if new_load > drone.capacity:
                continue
            
            # Energy check (we need energy to reach current from prev)
            required_energy = label.energy + energy_consumption
            max_energy = drone.energy_capacity or self.drone_energy
            if required_energy > max_energy:
                continue
            
            # Calculate cargo values
            cargo_up, cargo_down = self._calculate_enhanced_cargo_values(
                prev_node_id, prev_node.demand, dual_values, departure_time
            )
            
            # Create new backward label
            new_label = DroneLabel(
                node=prev_node_id,
                time=departure_time,
                load=new_load,
                energy=required_energy,  # Energy needed at this node
                cost=label.cost + base_cost,
                path=[prev_node_id] + label.path,
                cargo_up=label.cargo_up + cargo_up,
                cargo_down=label.cargo_down + cargo_down
            )
            
            # Dominance check for backward
            if self._is_dominated_enhanced(new_label, labels[prev_node_id], forward=False):
                continue
            
            # Remove dominated
            labels[prev_node_id] = [l for l in labels[prev_node_id] 
                                   if not self._dominates_enhanced(new_label, l)]
            
            # Add new label
            labels[prev_node_id].append(new_label)
            priority_key = (new_label.cost, -new_label.energy, -new_label.time, new_label)
            heapq.heappush(priority_queue, priority_key)
    
    def _get_time_cost_factor(self, time_minutes: int) -> float:
        """
        Get time-of-day cost factor for energy consumption.
        
        Args:
            time_minutes: Time in minutes from midnight
            
        Returns:
            Cost factor (1.0 = normal, >1.0 = more expensive)
        """
        hour = time_minutes // 60
        
        # Peak hours (7-9 AM, 5-7 PM) have higher energy consumption due to weather/traffic
        if (7 <= hour <= 9) or (17 <= hour <= 19):
            return 1.3
        # Night hours (10 PM - 6 AM) have reduced consumption
        elif hour >= 22 or hour <= 6:
            return 0.8
        # Normal hours
        else:
            return 1.0
    
    def _calculate_enhanced_cargo_values(self, node_id: int, demand: float, 
                                       dual_values: Dict[str, float],
                                       time: int) -> Tuple[float, float]:
        """
        Calculate cargo values with time-dependent factors.
        
        Args:
            node_id: Node ID
            demand: Demand at the node
            dual_values: Dual values from RMP
            time: Current time in minutes
            
        Returns:
            Tuple of (cargo_up, cargo_down) values
        """
        cargo_up = 0.0
        cargo_down = 0.0
        
        node = self.data_loader.nodes.get(node_id)
        if node and node.node_type == "metro_station":
            psi_up = dual_values.get(f"psi_up_{node_id}", 0.0)
            psi_down = dual_values.get(f"psi_down_{node_id}", 0.0)
            
            # Time-based allocation factor
            hour = time // 60
            if 7 <= hour <= 10:  # Morning rush - more up traffic
                up_factor = 1.5
                down_factor = 0.7
            elif 17 <= hour <= 20:  # Evening rush - more down traffic
                up_factor = 0.7
                down_factor = 1.5
            else:
                up_factor = 1.0
                down_factor = 1.0
            
            adjusted_psi_up = psi_up * up_factor
            adjusted_psi_down = psi_down * down_factor
            
            if adjusted_psi_up > adjusted_psi_down:
                cargo_up = demand
            else:
                cargo_down = demand
        
        return cargo_up, cargo_down
    
    def _is_dominated_enhanced(self, new_label: DroneLabel, existing_labels: List[DroneLabel], 
                              forward: bool = True) -> bool:
        """
        Enhanced dominance check with epsilon tolerance.
        
        Args:
            new_label: New label to check
            existing_labels: Existing labels
            forward: True for forward search, False for backward
            
        Returns:
            True if dominated
        """
        eps_time = 5  # 5 minutes tolerance
        eps_load = 1.0  # 1 kg tolerance
        eps_energy = 5.0  # 5 energy units tolerance
        eps_cost = 0.1  # Cost tolerance
        
        for existing_label in existing_labels:
            if forward:
                # Forward dominance
                if (existing_label.time <= new_label.time + eps_time and
                    existing_label.load <= new_label.load + eps_load and
                    existing_label.energy >= new_label.energy - eps_energy and
                    existing_label.cost <= new_label.cost + eps_cost and
                    existing_label.cargo_up >= new_label.cargo_up - eps_load and
                    existing_label.cargo_down >= new_label.cargo_down - eps_load):
                    return True
            else:
                # Backward dominance: earlier time and higher energy are better
                if (existing_label.time >= new_label.time - eps_time and
                    existing_label.load <= new_label.load + eps_load and
                    existing_label.energy <= new_label.energy + eps_energy and
                    existing_label.cost <= new_label.cost + eps_cost):
                    return True
        
        return False
    
    def _dominates_enhanced(self, label1: DroneLabel, label2: DroneLabel) -> bool:
        """
        Check if label1 dominates label2 with enhanced criteria.
        
        Args:
            label1: First label
            label2: Second label
            
        Returns:
            True if label1 dominates label2
        """
        return (label1.time <= label2.time and
                label1.load <= label2.load and
                label1.energy >= label2.energy and
                label1.cost <= label2.cost and
                label1.cargo_up >= label2.cargo_up and
                label1.cargo_down >= label2.cargo_down and
                (label1.time < label2.time or label1.load < label2.load or 
                 label1.energy > label2.energy or label1.cost < label2.cost))
    
    def _join_labels(self, forward_labels: Dict[int, List[DroneLabel]], 
                    backward_labels: Dict[int, List[DroneLabel]], 
                    drone, dual_values: Dict[str, float]) -> List[Column]:
        """
        Join forward and backward labels to create complete routes.
        
        Args:
            forward_labels: Forward labels by node
            backward_labels: Backward labels by node
            drone: Drone vehicle object
            dual_values: Dual values from RMP
            
        Returns:
            List of columns with negative reduced cost
        """
        negative_columns = []
        
        # Try to join at common nodes
        for node_id in forward_labels:
            if node_id in backward_labels:
                for f_label in forward_labels[node_id]:
                    for b_label in backward_labels[node_id]:
                        column = self._try_join_at_node(f_label, b_label, drone, dual_values)
                        if column and column.reduced_cost < -1e-6:
                            negative_columns.append(column)
        
        return negative_columns
    
    def _try_join_at_node(self, forward_label: DroneLabel, backward_label: DroneLabel,
                         drone, dual_values: Dict[str, float]) -> Optional[Column]:
        """
        Try to join forward and backward labels at a common node.
        
        Args:
            forward_label: Forward label
            backward_label: Backward label
            drone: Drone vehicle object
            dual_values: Dual values from RMP
            
        Returns:
            Column if joining is feasible, None otherwise
        """
        # Check time feasibility
        if forward_label.time > backward_label.time:
            return None
        
        # Check capacity feasibility
        total_load = forward_label.load + backward_label.load
        if total_load > drone.capacity:
            return None
        
        # Check energy feasibility
        # Forward label has energy remaining, backward label has energy needed
        if forward_label.energy < backward_label.energy:
            return None
        
        # Calculate return cost
        last_node = backward_label.path[-1]
        return_distance = self.data_loader.get_distance(last_node, self.depot_node)
        return_energy = return_distance * (drone.energy_consumption_rate or 2.0)
        return_cost = return_distance * drone.cost_per_km
        
        # Check if enough energy for return
        if forward_label.energy - backward_label.energy < return_energy:
            return None
        
        # Create complete route
        complete_path = forward_label.path + backward_label.path[1:]  # Avoid duplicate
        total_cost = forward_label.cost + backward_label.cost + return_cost
        total_cargo_up = forward_label.cargo_up + backward_label.cargo_up
        total_cargo_down = forward_label.cargo_down + backward_label.cargo_down
        
        # Create column details
        column_details = self._create_joined_column_details(
            complete_path, total_cargo_up, total_cargo_down, drone
        )
        
        # Calculate reduced cost
        reduced_cost = calculate_reduced_cost(total_cost, dual_values, column_details)
        
        # Create column
        column = Column(
            id=f"drone_bid_{drone.id}_{len(complete_path)}_{forward_label.time}",
            type=ColumnType.DRONE_ROUTE,
            vehicle_type=VehicleType.DRONE,
            direct_cost=total_cost,
            reduced_cost=reduced_cost,
            details=column_details
        )
        
        return column
    
    def _try_direct_returns(self, forward_labels: Dict[int, List[DroneLabel]], 
                           drone, dual_values: Dict[str, float]) -> List[Column]:
        """
        Try direct return to depot for forward labels.
        
        Args:
            forward_labels: Forward labels by node
            drone: Drone vehicle object
            dual_values: Dual values from RMP
            
        Returns:
            List of columns with negative reduced cost
        """
        negative_columns = []
        
        for node_id, labels_list in forward_labels.items():
            if node_id != self.depot_node:
                for label in labels_list:
                    column = self._try_return_to_depot(label, drone, dual_values)
                    if column and column.reduced_cost < -1e-6:
                        negative_columns.append(column)
        
        return negative_columns
    
    def _try_return_to_depot(self, label: DroneLabel, drone, 
                            dual_values: Dict[str, float]) -> Optional[Column]:
        """
        Try to create a route by returning to depot.
        
        Args:
            label: Current label
            drone: Drone vehicle object
            dual_values: Dual values from RMP
            
        Returns:
            Column if feasible, None otherwise
        """
        current_node = label.node
        
        # Calculate return parameters
        return_time = self.data_loader.get_travel_time(
            current_node, self.depot_node, VehicleType.DRONE
        )
        return_distance = self.data_loader.get_distance(current_node, self.depot_node)
        return_energy = return_distance * (drone.energy_consumption_rate or 2.0)
        return_cost = return_distance * drone.cost_per_km + (return_time / 60.0) * drone.cost_per_hour
        
        # Check feasibility
        if label.energy < return_energy:
            return None
        
        total_time = label.time + return_time
        if total_time > 1440:
            return None
        
        total_cost = label.cost + return_cost
        
        # Create column details
        column_details = self._create_column_details(label, drone)
        
        # Calculate reduced cost
        reduced_cost = calculate_reduced_cost(total_cost, dual_values, column_details)
        
        # Create column
        column = Column(
            id=f"drone_bid_{drone.id}_{len(label.path)}_{label.time}",
            type=ColumnType.DRONE_ROUTE,
            vehicle_type=VehicleType.DRONE,
            direct_cost=total_cost,
            reduced_cost=reduced_cost,
            details=column_details
        )
        
        return column
    
    def _create_column_details(self, label: DroneLabel, drone) -> Dict[str, Any]:
        """Create detailed column information."""
        details = {
            'a_ip': {},
            'delta_up': {},
            'delta_down': {},
            'resource_usage': {},
            'route': label.path + [self.depot_node],
            'timing': [],
            'cargo': {
                'cargo_up': label.cargo_up,
                'cargo_down': label.cargo_down
            },
            'energy_profile': []
        }
        
        # Set demand coverage
        for node_id in label.path[1:]:
            node = self.data_loader.nodes.get(node_id)
            if node and node.demand > 0:
                for demand_id, demand in self.data_loader.demands.items():
                    if demand.destination == node_id:
                        details['a_ip'][demand_id] = 1.0
                        break
        
        # Set inventory coefficients
        for node_id in label.path[1:]:
            node = self.data_loader.nodes.get(node_id)
            if node and node.node_type == "metro_station":
                details['delta_up'][node_id] = label.cargo_up
                details['delta_down'][node_id] = label.cargo_down
        
        # Set resource usage
        from utils import create_time_slices
        from config import RESOURCE_TIME_SLICE
        
        time_slices = create_time_slices(0, 1440, RESOURCE_TIME_SLICE)
        for slice_start, slice_end in time_slices:
            if 0 < slice_end and label.time > slice_start:
                details['resource_usage'][f"{slice_start}_{slice_end}"] = 1.0
        
        # Create energy profile
        current_energy = drone.energy_capacity or self.drone_energy
        current_time = 0
        
        for i in range(len(label.path) - 1):
            from_node = label.path[i]
            to_node = label.path[i + 1]
            
            travel_distance = self.data_loader.get_distance(from_node, to_node)
            energy_consumption = travel_distance * (drone.energy_consumption_rate or 2.0)
            travel_time = self.data_loader.get_travel_time(from_node, to_node, VehicleType.DRONE)
            
            current_energy -= energy_consumption
            current_time += travel_time
            
            details['energy_profile'].append({
                'node': to_node,
                'time': current_time,
                'energy': current_energy,
                'load': sum(self.data_loader.nodes[n].demand for n in label.path[:i+2] if n != 0)
            })
        
        return details
    
    def _create_joined_column_details(self, complete_path: List[int], 
                                     total_cargo_up: float, total_cargo_down: float,
                                     drone) -> Dict[str, Any]:
        """Create column details for joined route."""
        details = {
            'a_ip': {},
            'delta_up': {},
            'delta_down': {},
            'resource_usage': {},
            'route': complete_path + [self.depot_node],
            'timing': [],
            'cargo': {
                'cargo_up': total_cargo_up,
                'cargo_down': total_cargo_down
            },
            'energy_profile': []
        }
        
        # Set demand coverage
        for node_id in complete_path[1:]:
            node = self.data_loader.nodes.get(node_id)
            if node and node.demand > 0:
                for demand_id, demand in self.data_loader.demands.items():
                    if demand.destination == node_id:
                        details['a_ip'][demand_id] = 1.0
                        break
        
        # Set inventory coefficients
        for node_id in complete_path[1:]:
            node = self.data_loader.nodes.get(node_id)
            if node and node.node_type == "metro_station":
                details['delta_up'][node_id] = total_cargo_up
                details['delta_down'][node_id] = total_cargo_down
        
        # Set resource usage
        from utils import create_time_slices
        from config import RESOURCE_TIME_SLICE
        
        time_slices = create_time_slices(0, 1440, RESOURCE_TIME_SLICE)
        for slice_start, slice_end in time_slices:
            details['resource_usage'][f"{slice_start}_{slice_end}"] = 1.0
        
        return details
