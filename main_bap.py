"""
Algorithm 7: Branch-and-Price Main Loop
Complete Branch-and-Price solver coordinating RMP and pricing algorithms.
"""

import time
import os
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
import pandas as pd

from data_loader import DataLoader
from data_structures import (
    Column, BapNode, SolutionStats, ColumnType, VehicleType
)
from rmp_solver import RMPSolver
from algorithm1_ula_t import ULATSolver
from algorithm2_ula_d import ULADSolver
from algorithm3_sela_m import SELAMSolver
from algorithm4_bla_t import BLATSolver
from algorithm5_bla_d import BLADSolver
from algorithm6_bala_m import BALAMSolver
from excel_writer import ExcelWriter
from utils import log_message
from config import (
    COLUMN_GENERATION_TOLERANCE, BRANCHING_TOLERANCE, MAX_ITERATIONS,
    DATA_DIR, OUTPUT_DIR
)

class BranchAndPriceSolver:
    """
    Main Branch-and-Price solver for urban logistics optimization.
    """
    
    def __init__(self, use_optimized: bool = True):
        self.use_optimized = use_optimized
        self.data_loader = DataLoader()
        self.rmp_solver = RMPSolver(self.data_loader)
        
        # Initialize pricing solvers
        if use_optimized:
            self.truck_solver = BLATSolver(self.data_loader)
            self.drone_solver = BLADSolver(self.data_loader)
            self.metro_solver = BALAMSolver(self.data_loader)
        else:
            self.truck_solver = ULATSolver(self.data_loader)
            self.drone_solver = ULADSolver(self.data_loader)
            self.metro_solver = SELAMSolver(self.data_loader)
        
        # Solution tracking
        self.columns: List[Column] = []
        self.bap_tree: List[BapNode] = []
        self.best_integer_solution: Optional[Dict[str, Any]] = None
        self.best_bound: float = float('inf')
        self.iteration_log: List[Dict[str, Any]] = []
        
        # Statistics
        self.stats = SolutionStats()
        self.start_time: float = 0.0
        
    def solve(self) -> bool:
        """
        Main solve method implementing the complete Branch-and-Price algorithm.
        
        Returns:
            True if solved successfully, False otherwise
        """
        log_message("Starting Branch-and-Price solver")
        self.start_time = time.time()
        
        try:
            # Load data
            if not self.data_loader.load_all_data():
                log_message("Failed to load data", "ERROR")
                return False
            
            # Initialize with empty columns
            self._initialize_columns()
            
            # Create root node
            root_node = BapNode(id=0, parent_id=None, depth=0)
            self.bap_tree.append(root_node)
            
            # Branch-and-Price tree processing
            node_queue = deque([root_node])
            
            while node_queue and len(self.iteration_log) < MAX_ITERATIONS:
                current_node = node_queue.popleft()
                
                log_message(f"Processing node {current_node.id} at depth {current_node.depth}")
                
                # Solve node using column generation
                node_result = self._solve_node(current_node)
                
                if not node_result['feasible']:
                    log_message(f"Node {current_node.id} is infeasible")
                    current_node.is_feasible = False
                    continue
                
                current_node.lower_bound = node_result['objective_value']
                
                # Update best bound
                if current_node.lower_bound < self.best_bound:
                    self.best_bound = current_node.lower_bound
                
                # Check if solution is integer
                if node_result['is_integer']:
                    log_message(f"Integer solution found at node {current_node.id}")
                    current_node.is_integer = True
                    
                    # Update best integer solution
                    if (not self.best_integer_solution or 
                        node_result['objective_value'] < self.best_integer_solution['objective_value']):
                        self.best_integer_solution = node_result
                        log_message(f"New best integer solution: {node_result['objective_value']:.2f}")
                
                else:
                    # Branch on fractional solution
                    child_nodes = self._branch_node(current_node, node_result)
                    
                    # Add child nodes to queue (best-first search)
                    for child in child_nodes:
                        if child.is_feasible:
                            node_queue.append(child)
                    
                    # Sort queue by lower bound (best-first)
                    node_queue = deque(sorted(node_queue, key=lambda n: n.lower_bound))
                
                # Prune nodes with bound worse than best integer solution
                if self.best_integer_solution:
                    node_queue = deque([n for n in node_queue 
                                      if n.lower_bound < self.best_integer_solution['objective_value']])
                
                # Log iteration
                self._log_iteration(current_node, node_result)
            
            # Finalize solution
            self._finalize_solution()
            
            log_message("Branch-and-Price solver completed")
            return True
            
        except Exception as e:
            log_message(f"Error in Branch-and-Price solver: {str(e)}", "ERROR")
            return False
    
    def _initialize_columns(self):
        """Initialize with basic feasible columns."""
        log_message("Initializing basic columns")
        
        # Create simple single-demand columns for each vehicle type
        vehicle_types = {VehicleType.TRUCK, VehicleType.DRONE}
        
        for vehicle_type in vehicle_types:
            for demand_id, demand in self.data_loader.demands.items():
                if demand.origin == 0:  # From depot
                    column = self._create_simple_column(demand, vehicle_type)
                    if column:
                        self.columns.append(column)
        
        log_message(f"Initialized {len(self.columns)} basic columns")
    
    def _create_simple_column(self, demand, vehicle_type: VehicleType) -> Optional[Column]:
        """Create a simple column serving one demand."""
        try:
            # Simple direct route: depot -> customer -> depot
            route = [demand.origin, demand.destination, demand.origin]
            
            # Calculate basic cost
            distance = self.data_loader.get_distance(demand.origin, demand.destination)
            cost = distance * 2.0  # Round trip
            
            # Create column details
            details = {
                'a_ip': {demand.id: 1.0},
                'delta_up': {},
                'delta_down': {},
                'resource_usage': {'0_1440': 1.0},  # Uses resource for entire day
                'route': route,
                'timing': [],
                'cargo': {'cargo_up': 0.0, 'cargo_down': 0.0}
            }
            
            column_type = ColumnType.TRUCK_ROUTE if vehicle_type == VehicleType.TRUCK else ColumnType.DRONE_ROUTE
            
            column = Column(
                id=f"init_{vehicle_type.value}_{demand.id}",
                type=column_type,
                vehicle_type=vehicle_type,
                direct_cost=cost,
                details=details
            )
            
            return column
            
        except Exception as e:
            log_message(f"Error creating simple column: {str(e)}", "ERROR")
            return None
    
    def _solve_node(self, node: BapNode) -> Dict[str, Any]:
        """
        Solve a single node using column generation.
        
        Args:
            node: Branch-and-price node to solve
            
        Returns:
            Dictionary with node solution information
        """
        log_message(f"Starting column generation for node {node.id}")
        
        iteration = 0
        while iteration < MAX_ITERATIONS:
            iteration += 1
            
            # Solve RMP
            if not self.rmp_solver.build_model(self.columns, node):
                return {'feasible': False}
            
            if not self.rmp_solver.solve():
                return {'feasible': False}
            
            # Get RMP solution
            rmp_solution = self.rmp_solver.get_solution()
            dual_values = rmp_solution['dual_values']
            
            # Solve pricing problems
            new_columns = []
            
            # Truck pricing
            truck_columns = self.truck_solver.solve_pricing(dual_values)
            new_columns.extend(truck_columns)
            
            # Drone pricing
            drone_columns = self.drone_solver.solve_pricing(dual_values)
            new_columns.extend(drone_columns)
            
            # Metro pricing
            metro_columns = self.metro_solver.solve_pricing(dual_values)
            new_columns.extend(metro_columns)
            
            log_message(f"Iteration {iteration}: Found {len(new_columns)} new columns")
            
            # Check termination condition
            min_reduced_cost = min([col.reduced_cost for col in new_columns], 
                                 default=0.0)
            
            if min_reduced_cost >= -COLUMN_GENERATION_TOLERANCE:
                log_message(f"Column generation converged after {iteration} iterations")
                break
            
            # Add new columns
            self.columns.extend(new_columns)
            
            # Clean up old RMP model
            self.rmp_solver.dispose()
        
        # Get final solution
        final_solution = self.rmp_solver.get_solution()
        final_solution['is_integer'] = self.rmp_solver.is_integer_solution(BRANCHING_TOLERANCE)
        final_solution['feasible'] = True
        
        return final_solution
    
    def _branch_node(self, parent_node: BapNode, solution: Dict[str, Any]) -> List[BapNode]:
        """
        Branch on a fractional solution.
        
        Args:
            parent_node: Parent node
            solution: Fractional solution
            
        Returns:
            List of child nodes
        """
        # Find fractional variable for branching
        fractional_var = self.rmp_solver.get_fractional_variable()
        
        if not fractional_var:
            log_message("No fractional variable found for branching", "WARNING")
            return []
        
        var_name, var_value = fractional_var
        log_message(f"Branching on variable {var_name} = {var_value:.3f}")
        
        # Create child nodes
        child_nodes = []
        
        # Child 1: variable = 0
        child1 = BapNode(
            id=len(self.bap_tree),
            parent_id=parent_node.id,
            depth=parent_node.depth + 1,
            constraints=parent_node.constraints.copy()
        )
        child1.add_branching_constraint(var_name, 0.0, "=0")
        self.bap_tree.append(child1)
        child_nodes.append(child1)
        
        # Child 2: variable = 1
        child2 = BapNode(
            id=len(self.bap_tree),
            parent_id=parent_node.id,
            depth=parent_node.depth + 1,
            constraints=parent_node.constraints.copy()
        )
        child2.add_branching_constraint(var_name, 1.0, "=1")
        self.bap_tree.append(child2)
        child_nodes.append(child2)
        
        log_message(f"Created {len(child_nodes)} child nodes")
        return child_nodes
    
    def _log_iteration(self, node: BapNode, result: Dict[str, Any]):
        """Log iteration information."""
        log_entry = {
            'iteration': len(self.iteration_log) + 1,
            'node_id': node.id,
            'node_depth': node.depth,
            'lower_bound': result.get('objective_value', float('inf')),
            'best_bound': self.best_bound,
            'best_integer': self.best_integer_solution['objective_value'] if self.best_integer_solution else float('inf'),
            'gap': self._calculate_gap(),
            'num_columns': len(self.columns),
            'is_integer': result.get('is_integer', False),
            'runtime': time.time() - self.start_time
        }
        
        self.iteration_log.append(log_entry)
        
        log_message(f"Iteration {log_entry['iteration']}: "
                   f"LB={log_entry['lower_bound']:.2f}, "
                   f"Gap={log_entry['gap']:.2f}%, "
                   f"Cols={log_entry['num_columns']}")
    
    def _calculate_gap(self) -> float:
        """Calculate optimality gap."""
        if not self.best_integer_solution:
            return float('inf')
        
        best_integer = self.best_integer_solution['objective_value']
        if abs(best_integer) < 1e-10:
            return 0.0
        
        return ((best_integer - self.best_bound) / abs(best_integer)) * 100.0
    
    def _finalize_solution(self):
        """Finalize solution and update statistics."""
        self.stats.total_runtime = time.time() - self.start_time
        self.stats.num_iterations = len(self.iteration_log)
        
        if self.best_integer_solution:
            self.stats.total_cost = self.best_integer_solution['objective_value']
            self.stats.final_gap = self._calculate_gap()
            
            # Count routes by type
            for var_name, value in self.best_integer_solution['variable_values'].items():
                if value > 0.5:  # Variable is selected
                    if 'drone' in var_name:
                        self.stats.num_drone_routes += 1
                    elif 'truck' in var_name:
                        self.stats.num_truck_routes += 1
                    elif 'metro' in var_name:
                        self.stats.num_metro_schedules += 1
        
        # Calculate demand served
        total_demand = sum(d.quantity for d in self.data_loader.demands.values())
        self.stats.total_demand_served = total_demand  # Assume all served for now
        
        log_message(f"Final statistics: Cost={self.stats.total_cost:.2f}, "
                   f"Gap={self.stats.final_gap:.2f}%, "
                   f"Runtime={self.stats.total_runtime:.2f}s")
    
    def export_results(self, output_dir: str = OUTPUT_DIR):
        """Export results to Excel files."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        log_message("Exporting results to Excel")
        
        writer = ExcelWriter(self.data_loader)
        
        try:
            # Export all required outputs
            writer.export_solution_summary(self.stats, output_dir)
            writer.export_bounds_log(self.iteration_log, output_dir)
            writer.export_runtime_comparison(self._get_runtime_comparison(), output_dir)
            
            if self.best_integer_solution:
                # Extract route information from solution
                routes_data = self._extract_routes_from_solution()
                
                writer.export_truck_routes(routes_data['truck_routes'], output_dir)
                writer.export_drone_routes(routes_data['drone_routes'], output_dir)
                writer.export_metro_timetables(routes_data['metro_schedules'], output_dir)
                writer.export_metro_shipments(routes_data['metro_shipments'], output_dir)
            
            log_message("Results exported successfully")
            
        except Exception as e:
            log_message(f"Error exporting results: {str(e)}", "ERROR")
    
    def _extract_routes_from_solution(self) -> Dict[str, List[Dict[str, Any]]]:
        """Extract route information from the best solution."""
        routes_data = {
            'truck_routes': [],
            'drone_routes': [],
            'metro_schedules': [],
            'metro_shipments': []
        }
        
        if not self.best_integer_solution:
            return routes_data
        
        # Extract selected columns from solution
        selected_columns = []
        for var_name, value in self.best_integer_solution['variable_values'].items():
            if value > 0.5:  # Variable is selected
                # Find corresponding column
                column_id = var_name.replace('lambda_', '')
                for column in self.columns:
                    if column.id == column_id:
                        selected_columns.append(column)
                        break
        
        # Process each selected column
        for column in selected_columns:
            if column.vehicle_type == VehicleType.TRUCK:
                routes_data['truck_routes'].append(self._extract_truck_route(column))
            elif column.vehicle_type == VehicleType.DRONE:
                routes_data['drone_routes'].append(self._extract_drone_route(column))
            elif column.vehicle_type == VehicleType.METRO:
                metro_schedule = self._extract_metro_schedule(column)
                routes_data['metro_schedules'].append(metro_schedule)
                routes_data['metro_shipments'].extend(self._extract_metro_shipments(column))
        
        return routes_data
    
    def _extract_truck_route(self, column: Column) -> Dict[str, Any]:
        """Extract truck route information."""
        return {
            'route_id': column.id,
            'vehicle_type': 'Truck',
            'route': column.details['route'],
            'total_cost': column.direct_cost,
            'cargo_up': column.details['cargo']['cargo_up'],
            'cargo_down': column.details['cargo']['cargo_down'],
            'demands_served': list(column.details['a_ip'].keys())
        }
    
    def _extract_drone_route(self, column: Column) -> Dict[str, Any]:
        """Extract drone route information."""
        return {
            'route_id': column.id,
            'vehicle_type': 'Drone',
            'route': column.details['route'],
            'total_cost': column.direct_cost,
            'energy_profile': column.details.get('energy_profile', []),
            'cargo_up': column.details['cargo']['cargo_up'],
            'cargo_down': column.details['cargo']['cargo_down'],
            'demands_served': list(column.details['a_ip'].keys())
        }
    
    def _extract_metro_schedule(self, column: Column) -> Dict[str, Any]:
        """Extract metro schedule information."""
        return {
            'schedule_id': column.id,
            'line_id': column.details.get('line_id', 'unknown'),
            'direction': column.details.get('direction', 'unknown'),
            'schedule': column.details['timing'],
            'total_cost': column.direct_cost,
            'stations': column.details['route']
        }
    
    def _extract_metro_shipments(self, column: Column) -> List[Dict[str, Any]]:
        """Extract metro shipment information."""
        shipments = []
        
        for station_id, cargo_up in column.details['delta_up'].items():
            if cargo_up > 0:
                shipments.append({
                    'schedule_id': column.id,
                    'station_id': station_id,
                    'direction': 'up',
                    'cargo_amount': cargo_up
                })
        
        for station_id, cargo_down in column.details['delta_down'].items():
            if cargo_down > 0:
                shipments.append({
                    'schedule_id': column.id,
                    'station_id': station_id,
                    'direction': 'down',
                    'cargo_amount': cargo_down
                })
        
        return shipments
    
    def _get_runtime_comparison(self) -> Dict[str, Any]:
        """Get runtime comparison between basic and optimized algorithms."""
        algorithm_type = "Optimized" if self.use_optimized else "Basic"
        
        return {
            'algorithm_type': algorithm_type,
            'total_runtime': self.stats.total_runtime,
            'iterations': self.stats.num_iterations,
            'final_cost': self.stats.total_cost,
            'final_gap': self.stats.final_gap,
            'columns_generated': len(self.columns)
        }

def main():
    """Main function to run the Branch-and-Price solver."""
    # Ensure data and output directories exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Run basic algorithms
    log_message("Running basic algorithms (1-3)")
    basic_solver = BranchAndPriceSolver(use_optimized=False)
    if basic_solver.solve():
        basic_solver.export_results(os.path.join(OUTPUT_DIR, "basic"))
    
    # Run optimized algorithms
    log_message("Running optimized algorithms (4-6)")
    optimized_solver = BranchAndPriceSolver(use_optimized=True)
    if optimized_solver.solve():
        optimized_solver.export_results(os.path.join(OUTPUT_DIR, "optimized"))
    
    # Create comparison report
    comparison_data = [
        basic_solver._get_runtime_comparison(),
        optimized_solver._get_runtime_comparison()
    ]
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_excel(os.path.join(OUTPUT_DIR, "algorithm_comparison.xlsx"), index=False)
    
    log_message("Branch-and-Price solver completed successfully")

if __name__ == "__main__":
    main()
