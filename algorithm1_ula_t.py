"""
Algorithm 1: Unidirectional Label-setting Algorithm for Trucks (ULA-T)
Basic truck routing with resource constraints.
"""

from typing import List, Dict, Any, Optional, Tuple
import heapq
from collections import defaultdict

from data_structures import TruckLabel, Column, ColumnType, VehicleType
from data_loader import DataLoader
from utils import log_message, calculate_reduced_cost
from config import DEFAULT_TRUCK_CAPACITY

class ULATSolver:
    """
    Unidirectional Label-setting Algorithm for Truck routing.
    Solves Resource Constrained Shortest Path Problem (RCSPP) for trucks.
    """
    
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.depot_node = 0  # Assume depot is node 0
        self.truck_capacity = DEFAULT_TRUCK_CAPACITY
        
    def solve_pricing(self, dual_values: Dict[str, float]) -> List[Column]:
        """
        Solve truck pricing problem using label-setting algorithm.
        
        Args:
            dual_values: Dual values from RMP solution
            
        Returns:
            List of columns with negative reduced cost
        """
        log_message("Starting ULA-T pricing algorithm")
        
        negative_cost_columns = []
        
        # Get available trucks
        trucks = [v for v in self.data_loader.vehicles.values() 
                 if v.type == VehicleType.TRUCK]
        
        for truck in trucks:
            columns = self._solve_for_truck(truck, dual_values)
            negative_cost_columns.extend(columns)
        
        log_message(f"ULA-T found {len(negative_cost_columns)} columns with negative reduced cost")
        return negative_cost_columns
    
    def _solve_for_truck(self, truck, dual_values: Dict[str, float]) -> List[Column]:
        """
        Solve pricing for a specific truck.
        
        Args:
            truck: Truck vehicle object
            dual_values: Dual values from RMP
            
        Returns:
            List of columns with negative reduced cost
        """
        # Initialize labels
        labels = defaultdict(list)  # node_id -> list of labels
        processed = set()  # (node, time) pairs
        priority_queue = []  # Min-heap for label selection
        
        # Create initial label at depot
        initial_label = TruckLabel(
            node=self.depot_node,
            time=0,
            load=0.0,
            cost=0.0,
            path=[self.depot_node]
        )
        
        labels[self.depot_node].append(initial_label)
        heapq.heappush(priority_queue, (0.0, 0, initial_label))  # (cost, time, label)
        
        negative_columns = []
        
        while priority_queue:
            current_cost, current_time, current_label = heapq.heappop(priority_queue)
            
            # Skip if already processed
            state_key = (current_label.node, current_label.time)
            if state_key in processed:
                continue
            processed.add(state_key)
            
            # Check if we can return to depot
            if (current_label.node != self.depot_node and 
                len(current_label.path) > 1):
                
                return_column = self._try_return_to_depot(current_label, truck, dual_values)
                if return_column and return_column.reduced_cost < -1e-6:
                    negative_columns.append(return_column)
            
            # Extend to neighboring nodes
            self._extend_label(current_label, truck, dual_values, labels, priority_queue)
        
        return negative_columns
    
    def _extend_label(self, label: TruckLabel, truck, dual_values: Dict[str, float],
                     labels: Dict[int, List[TruckLabel]], priority_queue: List):
        """
        Extend a label to neighboring nodes.
        
        Args:
            label: Current label to extend
            truck: Truck vehicle object
            dual_values: Dual values from RMP
            labels: Dictionary of labels by node
            priority_queue: Priority queue for label processing
        """
        current_node = label.node
        
        # Try extending to all other nodes
        for next_node_id, next_node in self.data_loader.nodes.items():
            if next_node_id == current_node or next_node_id in label.path:
                continue  # Skip same node and already visited nodes
            
            # Calculate travel time and cost
            travel_time = self.data_loader.get_travel_time(
                current_node, next_node_id, VehicleType.TRUCK
            )
            travel_distance = self.data_loader.get_distance(current_node, next_node_id)
            travel_cost = travel_distance * truck.cost_per_km + \
                         (travel_time / 60.0) * truck.cost_per_hour
            
            # Calculate arrival time
            arrival_time = label.time + travel_time + \
                          self.data_loader.nodes[current_node].service_time
            
            # Check time window constraints
            if (arrival_time < next_node.time_window_start or 
                arrival_time > next_node.time_window_end):
                continue
            
            # Check capacity constraints
            new_load = label.load + next_node.demand
            if new_load > truck.capacity:
                continue
            
            # Calculate cargo values based on dual values
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
    
    def _is_dominated(self, new_label: TruckLabel, existing_labels: List[TruckLabel]) -> bool:
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
    
    def _try_return_to_depot(self, label: TruckLabel, truck, 
                            dual_values: Dict[str, float]) -> Optional[Column]:
        """
        Try to create a complete route by returning to depot.
        
        Args:
            label: Current label
            truck: Truck vehicle object
            dual_values: Dual values from RMP
            
        Returns:
            Column object if route is feasible, None otherwise
        """
        current_node = label.node
        
        # Calculate return travel
        return_time = self.data_loader.get_travel_time(
            current_node, self.depot_node, VehicleType.TRUCK
        )
        return_distance = self.data_loader.get_distance(current_node, self.depot_node)
        return_cost = return_distance * truck.cost_per_km + \
                     (return_time / 60.0) * truck.cost_per_hour
        
        # Check if return is feasible
        total_time = label.time + return_time
        if total_time > 1440:  # 24 hours limit
            return None
        
        # Calculate total route cost
        total_cost = label.cost + return_cost
        
        # Create column details
        column_details = self._create_column_details(label, truck)
        
        # Calculate reduced cost
        reduced_cost = calculate_reduced_cost(total_cost, dual_values, column_details)
        
        # Create column
        column = Column(
            id=f"truck_{truck.id}_{len(label.path)}_{label.time}",
            type=ColumnType.TRUCK_ROUTE,
            vehicle_type=VehicleType.TRUCK,
            direct_cost=total_cost,
            reduced_cost=reduced_cost,
            details=column_details
        )
        
        return column
    
    def _create_column_details(self, label: TruckLabel, truck) -> Dict[str, Any]:
        """
        Create detailed information for the column.
        
        Args:
            label: Final label for the route
            truck: Truck vehicle object
            
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
            }
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
        
        # Simple resource usage: 1 if truck is active during time slice
        route_start_time = 0
        route_end_time = label.time
        
        for slice_start, slice_end in time_slices:
            if (route_start_time < slice_end and route_end_time > slice_start):
                details['resource_usage'][f"{slice_start}_{slice_end}"] = 1.0
        
        return details
