"""
Algorithm 4: Bidirectional Label-setting Algorithm for Trucks (BLA-T)
Optimized truck routing with bidirectional search.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
import heapq
from collections import defaultdict

from data_structures import TruckLabel, Column, ColumnType, VehicleType
from data_loader import DataLoader
from utils import log_message, calculate_reduced_cost
from config import DEFAULT_TRUCK_CAPACITY

class BLATSolver:
    """
    Bidirectional Label-setting Algorithm for Truck routing.
    Enhanced version with bidirectional search for better performance.
    """
    
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.depot_node = 0
        self.truck_capacity = DEFAULT_TRUCK_CAPACITY
        
    def solve_pricing(self, dual_values: Dict[str, float]) -> List[Column]:
        """
        Solve truck pricing problem using bidirectional label-setting algorithm.
        
        Args:
            dual_values: Dual values from RMP solution
            
        Returns:
            List of columns with negative reduced cost
        """
        log_message("Starting BLA-T pricing algorithm")
        
        negative_cost_columns = []
        
        # Get available trucks
        trucks = [v for v in self.data_loader.vehicles.values() 
                 if v.type == VehicleType.TRUCK]
        
        for truck in trucks:
            columns = self._solve_for_truck_bidirectional(truck, dual_values)
            negative_cost_columns.extend(columns)
        
        log_message(f"BLA-T found {len(negative_cost_columns)} columns with negative reduced cost")
        return negative_cost_columns
    
    def _solve_for_truck_bidirectional(self, truck, dual_values: Dict[str, float]) -> List[Column]:
        """
        Solve pricing for a specific truck using bidirectional search.
        
        Args:
            truck: Truck vehicle object
            dual_values: Dual values from RMP
            
        Returns:
            List of columns with negative reduced cost
        """
        # Forward search from depot
        forward_labels = self._forward_search(truck, dual_values)
        
        # Backward search to depot
        backward_labels = self._backward_search(truck, dual_values)
        
        # Join forward and backward labels to form complete routes
        negative_columns = self._join_labels(forward_labels, backward_labels, truck, dual_values)
        
        return negative_columns
    
    def _forward_search(self, truck, dual_values: Dict[str, float]) -> Dict[int, List[TruckLabel]]:
        """
        Forward search from depot.
        
        Args:
            truck: Truck vehicle object
            dual_values: Dual values from RMP
            
        Returns:
            Dictionary of forward labels by node
        """
        labels = defaultdict(list)
        processed = set()
        priority_queue = []
        
        # Initial label at depot
        initial_label = TruckLabel(
            node=self.depot_node,
            time=0,
            load=0.0,
            cost=0.0,
            path=[self.depot_node]
        )
        
        labels[self.depot_node].append(initial_label)
        heapq.heappush(priority_queue, (0.0, 0, initial_label))
        
        while priority_queue:
            current_cost, current_time, current_label = heapq.heappop(priority_queue)
            
            # State representation for processed check
            state_key = (current_label.node, current_label.time // 30, int(current_label.load))
            if state_key in processed:
                continue
            processed.add(state_key)
            
            # Extend to neighboring nodes
            self._extend_forward_label(current_label, truck, dual_values, labels, priority_queue)
        
        return dict(labels)
    
    def _backward_search(self, truck, dual_values: Dict[str, float]) -> Dict[int, List[TruckLabel]]:
        """
        Backward search to depot.
        
        Args:
            truck: Truck vehicle object
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
                # Start from end of time window
                initial_label = TruckLabel(
                    node=node_id,
                    time=node.time_window_end,
                    load=0.0,  # Backward: start with empty load
                    cost=0.0,
                    path=[node_id]
                )
                
                labels[node_id].append(initial_label)
                heapq.heappush(priority_queue, (0.0, -node.time_window_end, initial_label))
        
        while priority_queue:
            current_cost, neg_time, current_label = heapq.heappop(priority_queue)
            
            # State representation
            state_key = (current_label.node, current_label.time // 30, int(current_label.load))
            if state_key in processed:
                continue
            processed.add(state_key)
            
            # Extend backward
            self._extend_backward_label(current_label, truck, dual_values, labels, priority_queue)
        
        return dict(labels)
    
    def _extend_forward_label(self, label: TruckLabel, truck, dual_values: Dict[str, float],
                             labels: Dict[int, List[TruckLabel]], priority_queue: List):
        """
        Extend a forward label.
        
        Args:
            label: Current label to extend
            truck: Truck vehicle object
            dual_values: Dual values from RMP
            labels: Dictionary of labels by node
            priority_queue: Priority queue for label processing
        """
        current_node = label.node
        
        # Limit search depth
        if len(label.path) > 6:  # Max 6 nodes in path
            return
        
        for next_node_id, next_node in self.data_loader.nodes.items():
            if (next_node_id == current_node or 
                next_node_id in label.path or
                next_node_id == self.depot_node):  # Don't return to depot in forward search
                continue
            
            # Calculate extension
            travel_time = self.data_loader.get_travel_time(
                current_node, next_node_id, VehicleType.TRUCK
            )
            travel_distance = self.data_loader.get_distance(current_node, next_node_id)
            travel_cost = travel_distance * truck.cost_per_km + \
                         (travel_time / 60.0) * truck.cost_per_hour
            
            arrival_time = label.time + travel_time + \
                          self.data_loader.nodes[current_node].service_time
            
            # Check constraints
            if (arrival_time < next_node.time_window_start or 
                arrival_time > next_node.time_window_end):
                continue
                
            new_load = label.load + next_node.demand
            if new_load > truck.capacity:
                continue
            
            # Calculate cargo values
            cargo_up, cargo_down = self._calculate_cargo_values(
                next_node_id, next_node.demand, dual_values
            )
            
            # Create new label
            new_label = TruckLabel(
                node=next_node_id,
                time=arrival_time + next_node.service_time,
                load=new_load,
                cost=label.cost + travel_cost,
                path=label.path + [next_node_id],
                cargo_up=label.cargo_up + cargo_up,
                cargo_down=label.cargo_down + cargo_down
            )
            
            # Dominance check
            if self._is_dominated(new_label, labels[next_node_id], forward=True):
                continue
            
            # Remove dominated
            labels[next_node_id] = [l for l in labels[next_node_id] 
                                   if not new_label.dominates(l)]
            
            # Add new label
            labels[next_node_id].append(new_label)
            heapq.heappush(priority_queue, (new_label.cost, new_label.time, new_label))
    
    def _extend_backward_label(self, label: TruckLabel, truck, dual_values: Dict[str, float],
                              labels: Dict[int, List[TruckLabel]], priority_queue: List):
        """
        Extend a backward label.
        
        Args:
            label: Current label to extend
            truck: Truck vehicle object
            dual_values: Dual values from RMP
            labels: Dictionary of labels by node
            priority_queue: Priority queue for label processing
        """
        current_node = label.node
        
        # Limit search depth
        if len(label.path) > 6:
            return
        
        for prev_node_id, prev_node in self.data_loader.nodes.items():
            if (prev_node_id == current_node or 
                prev_node_id in label.path):
                continue
            
            # Calculate backward extension
            travel_time = self.data_loader.get_travel_time(
                prev_node_id, current_node, VehicleType.TRUCK
            )
            travel_distance = self.data_loader.get_distance(prev_node_id, current_node)
            travel_cost = travel_distance * truck.cost_per_km + \
                         (travel_time / 60.0) * truck.cost_per_hour
            
            # Backward time calculation
            departure_time = label.time - travel_time - prev_node.service_time
            
            # Check time windows
            if (departure_time < prev_node.time_window_start or 
                departure_time > prev_node.time_window_end):
                continue
            
            # Backward load calculation (we're "removing" demand)
            new_load = label.load + prev_node.demand  # In backward, we add demand
            if new_load > truck.capacity:
                continue
            
            # Calculate cargo values
            cargo_up, cargo_down = self._calculate_cargo_values(
                prev_node_id, prev_node.demand, dual_values
            )
            
            # Create new backward label
            new_label = TruckLabel(
                node=prev_node_id,
                time=departure_time,
                load=new_load,
                cost=label.cost + travel_cost,
                path=[prev_node_id] + label.path,
                cargo_up=label.cargo_up + cargo_up,
                cargo_down=label.cargo_down + cargo_down
            )
            
            # Dominance check
            if self._is_dominated(new_label, labels[prev_node_id], forward=False):
                continue
            
            # Remove dominated
            labels[prev_node_id] = [l for l in labels[prev_node_id] 
                                   if not new_label.dominates(l)]
            
            # Add new label
            labels[prev_node_id].append(new_label)
            heapq.heappush(priority_queue, (new_label.cost, -new_label.time, new_label))
    
    def _join_labels(self, forward_labels: Dict[int, List[TruckLabel]], 
                    backward_labels: Dict[int, List[TruckLabel]], 
                    truck, dual_values: Dict[str, float]) -> List[Column]:
        """
        Join forward and backward labels to create complete routes.
        
        Args:
            forward_labels: Forward labels by node
            backward_labels: Backward labels by node
            truck: Truck vehicle object
            dual_values: Dual values from RMP
            
        Returns:
            List of columns with negative reduced cost
        """
        negative_columns = []
        
        # Try to join labels at common nodes
        for node_id in forward_labels:
            if node_id in backward_labels:
                for f_label in forward_labels[node_id]:
                    for b_label in backward_labels[node_id]:
                        column = self._try_join_at_node(f_label, b_label, truck, dual_values)
                        if column and column.reduced_cost < -1e-6:
                            negative_columns.append(column)
        
        # Also try direct return to depot for forward labels
        for node_id, labels_list in forward_labels.items():
            if node_id != self.depot_node:
                for label in labels_list:
                    column = self._try_return_to_depot(label, truck, dual_values)
                    if column and column.reduced_cost < -1e-6:
                        negative_columns.append(column)
        
        return negative_columns
    
    def _try_join_at_node(self, forward_label: TruckLabel, backward_label: TruckLabel,
                         truck, dual_values: Dict[str, float]) -> Optional[Column]:
        """
        Try to join forward and backward labels at a common node.
        
        Args:
            forward_label: Forward label
            backward_label: Backward label
            truck: Truck vehicle object
            dual_values: Dual values from RMP
            
        Returns:
            Column if joining is feasible, None otherwise
        """
        # Check if joining is feasible
        if forward_label.time > backward_label.time:
            return None
        
        # Check capacity
        total_load = forward_label.load + backward_label.load
        if total_load > truck.capacity:
            return None
        
        # Calculate return to depot cost
        return_time = self.data_loader.get_travel_time(
            backward_label.path[-1], self.depot_node, VehicleType.TRUCK
        )
        return_distance = self.data_loader.get_distance(
            backward_label.path[-1], self.depot_node
        )
        return_cost = return_distance * truck.cost_per_km + \
                     (return_time / 60.0) * truck.cost_per_hour
        
        # Create complete route
        complete_path = forward_label.path + backward_label.path[1:]  # Avoid duplicate node
        total_cost = forward_label.cost + backward_label.cost + return_cost
        total_cargo_up = forward_label.cargo_up + backward_label.cargo_up
        total_cargo_down = forward_label.cargo_down + backward_label.cargo_down
        
        # Create column details
        column_details = self._create_joined_column_details(
            complete_path, total_cargo_up, total_cargo_down, truck
        )
        
        # Calculate reduced cost
        reduced_cost = calculate_reduced_cost(total_cost, dual_values, column_details)
        
        # Create column
        column = Column(
            id=f"truck_bid_{truck.id}_{len(complete_path)}_{forward_label.time}",
            type=ColumnType.TRUCK_ROUTE,
            vehicle_type=VehicleType.TRUCK,
            direct_cost=total_cost,
            reduced_cost=reduced_cost,
            details=column_details
        )
        
        return column
    
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
        
        node = self.data_loader.nodes.get(node_id)
        if node and node.node_type == "metro_station":
            psi_up = dual_values.get(f"psi_up_{node_id}", 0.0)
            psi_down = dual_values.get(f"psi_down_{node_id}", 0.0)
            
            if psi_up > psi_down:
                cargo_up = demand
            else:
                cargo_down = demand
        
        return cargo_up, cargo_down
    
    def _is_dominated(self, new_label: TruckLabel, existing_labels: List[TruckLabel], 
                     forward: bool = True) -> bool:
        """
        Check dominance with direction-specific logic.
        
        Args:
            new_label: New label to check
            existing_labels: Existing labels
            forward: True for forward search, False for backward
            
        Returns:
            True if dominated
        """
        for existing_label in existing_labels:
            if forward:
                if existing_label.dominates(new_label):
                    return True
            else:
                # Backward dominance: earlier time is better
                if (existing_label.time >= new_label.time and
                    existing_label.load <= new_label.load and
                    existing_label.cost <= new_label.cost):
                    return True
        return False
    
    def _try_return_to_depot(self, label: TruckLabel, truck, 
                            dual_values: Dict[str, float]) -> Optional[Column]:
        """
        Try to create a route by returning to depot.
        
        Args:
            label: Current label
            truck: Truck vehicle object
            dual_values: Dual values from RMP
            
        Returns:
            Column if feasible, None otherwise
        """
        current_node = label.node
        
        return_time = self.data_loader.get_travel_time(
            current_node, self.depot_node, VehicleType.TRUCK
        )
        return_distance = self.data_loader.get_distance(current_node, self.depot_node)
        return_cost = return_distance * truck.cost_per_km + \
                     (return_time / 60.0) * truck.cost_per_hour
        
        total_time = label.time + return_time
        if total_time > 1440:
            return None
        
        total_cost = label.cost + return_cost
        
        column_details = self._create_column_details(label, truck)
        reduced_cost = calculate_reduced_cost(total_cost, dual_values, column_details)
        
        column = Column(
            id=f"truck_bid_{truck.id}_{len(label.path)}_{label.time}",
            type=ColumnType.TRUCK_ROUTE,
            vehicle_type=VehicleType.TRUCK,
            direct_cost=total_cost,
            reduced_cost=reduced_cost,
            details=column_details
        )
        
        return column
    
    def _create_column_details(self, label: TruckLabel, truck) -> Dict[str, Any]:
        """Create column details from a label."""
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
            }
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
        
        return details
    
    def _create_joined_column_details(self, complete_path: List[int], 
                                     total_cargo_up: float, total_cargo_down: float,
                                     truck) -> Dict[str, Any]:
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
            }
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
        
        # Set resource usage (simplified)
        from utils import create_time_slices
        from config import RESOURCE_TIME_SLICE
        
        time_slices = create_time_slices(0, 1440, RESOURCE_TIME_SLICE)
        for slice_start, slice_end in time_slices:
            details['resource_usage'][f"{slice_start}_{slice_end}"] = 1.0
        
        return details
