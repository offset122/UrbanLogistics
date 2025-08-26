"""
Restricted Master Problem (RMP) solver using Gurobi.
Integrated with optimized Gurobi configuration for better performance.
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from data_structures import Column, ColumnType, VehicleType, BapNode, SolutionStats
from data_loader import DataLoader
from gurobi_config import gurobi_config
from utils import log_message
from config import RESOURCE_TIME_SLICE, COLUMN_GENERATION_TOLERANCE

class RMPSolver:
    """
    Gurobi-based solver for the Restricted Master Problem.
    """
    
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.model: Optional[gp.Model] = None
        self.variables: Dict[str, gp.Var] = {}
        self.constraints: Dict[str, gp.Constr] = {}
        self.dual_values: Dict[str, float] = {}
        self.objective_value: float = 0.0
        
    def build_model(self, columns: List[Column], bap_node: Optional[BapNode] = None) -> bool:
        """
        Build the RMP model with given columns.
        
        Args:
            columns: List of columns (routes/schedules)
            bap_node: Current branch-and-price node (for branching constraints)
            
        Returns:
            True if model built successfully, False otherwise
        """
        try:
            # Create optimized Gurobi model for RMP
            self.model = gurobi_config.create_model("RMP", "RMP", silent=True)
            if self.model is None:
                log_message("Failed to create Gurobi model - license issues", "ERROR")
                return False
            
            # Create variables for each column
            self.variables = {}
            for column in columns:
                var_name = f"lambda_{column.id}"
                var = self.model.addVar(
                    lb=0.0,
                    ub=1.0 if column.type == ColumnType.METRO_SCHEDULE else GRB.INFINITY,
                    obj=column.direct_cost,
                    vtype=GRB.CONTINUOUS,
                    name=var_name
                )
                self.variables[var_name] = var
            
            # Add demand coverage constraints
            self._add_demand_constraints(columns)
            
            # Add resource capacity constraints
            self._add_resource_constraints(columns)
            
            # Add inventory balance constraints
            self._add_inventory_constraints(columns)
            
            # Add branching constraints if applicable
            if bap_node:
                self._add_branching_constraints(bap_node)
            
            self.model.update()
            log_message(f"Built RMP model with {len(columns)} columns")
            return True
            
        except Exception as e:
            log_message(f"Error building RMP model: {str(e)}", "ERROR")
            return False
    
    def _add_demand_constraints(self, columns: List[Column]):
        """Add demand coverage constraints."""
        self.constraints['demand'] = {}
        
        for demand_id in self.data_loader.demands:
            constraint_name = f"demand_{demand_id}"
            
            # Each demand must be served exactly once
            expr = gp.LinExpr()
            for column in columns:
                var_name = f"lambda_{column.id}"
                if var_name in self.variables:
                    # Check if this column serves this demand
                    coverage = column.details.get('a_ip', {}).get(demand_id, 0)
                    if coverage > 0:
                        expr.addTerms(coverage, self.variables[var_name])
            
            constraint = self.model.addConstr(
                expr == 1.0,
                name=constraint_name
            )
            self.constraints['demand'][demand_id] = constraint
    
    def _add_resource_constraints(self, columns: List[Column]):
        """Add resource capacity constraints."""
        self.constraints['resource'] = {}
        
        # Get time slices for resource constraints
        from utils import create_time_slices
        time_slices = create_time_slices(0, 1440, RESOURCE_TIME_SLICE)
        
        # Vehicle type to resource mapping
        resource_types = {
            VehicleType.DRONE: 'drone',
            VehicleType.TRUCK: 'truck'
        }
        
        for vehicle_type, resource_name in resource_types.items():
            # Count available resources of this type
            available_resources = sum(1 for v in self.data_loader.vehicles.values() 
                                    if v.type == vehicle_type)
            
            for slice_start, slice_end in time_slices:
                constraint_name = f"resource_{resource_name}_{slice_start}_{slice_end}"
                
                expr = gp.LinExpr()
                for column in columns:
                    if column.vehicle_type == vehicle_type:
                        var_name = f"lambda_{column.id}"
                        if var_name in self.variables:
                            # Check resource usage in this time slice
                            usage = column.details.get('resource_usage', {}).get(
                                f"{slice_start}_{slice_end}", 0
                            )
                            if usage > 0:
                                expr.addTerms(usage, self.variables[var_name])
                
                constraint = self.model.addConstr(
                    expr <= available_resources,
                    name=constraint_name
                )
                self.constraints['resource'][constraint_name] = constraint
        
        # Add pilot capacity constraints (for drones)
        available_pilots = sum(1 for v in self.data_loader.vehicles.values() 
                             if v.type == VehicleType.DRONE)
        
        for slice_start, slice_end in time_slices:
            constraint_name = f"resource_pilot_{slice_start}_{slice_end}"
            
            expr = gp.LinExpr()
            for column in columns:
                if column.vehicle_type == VehicleType.DRONE:
                    var_name = f"lambda_{column.id}"
                    if var_name in self.variables:
                        usage = column.details.get('resource_usage', {}).get(
                            f"{slice_start}_{slice_end}", 0
                        )
                        if usage > 0:
                            expr.addTerms(usage, self.variables[var_name])
            
            constraint = self.model.addConstr(
                expr <= available_pilots,
                name=constraint_name
            )
            self.constraints['resource'][constraint_name] = constraint
    
    def _add_inventory_constraints(self, columns: List[Column]):
        """Add inventory balance constraints."""
        self.constraints['inventory'] = {}
        
        # Find metro stations
        metro_stations = [node_id for node_id, node in self.data_loader.nodes.items() 
                         if node.node_type == "metro_station"]
        
        for station_id in metro_stations:
            for direction in ['up', 'down']:
                constraint_name = f"inventory_{direction}_{station_id}"
                
                expr = gp.LinExpr()
                for column in columns:
                    var_name = f"lambda_{column.id}"
                    if var_name in self.variables:
                        # Get inventory coefficient for this direction
                        delta_key = f'delta_{direction}'
                        inventory_coeff = column.details.get(delta_key, {}).get(station_id, 0)
                        
                        if inventory_coeff != 0:
                            expr.addTerms(inventory_coeff, self.variables[var_name])
                
                # Inventory balance: inflow - outflow <= capacity
                metro_capacity = self.data_loader.parameters.get('metro_capacity', 5000.0)
                constraint = self.model.addConstr(
                    expr <= metro_capacity,
                    name=constraint_name
                )
                self.constraints['inventory'][constraint_name] = constraint
    
    def _add_branching_constraints(self, bap_node: BapNode):
        """Add branching constraints from the branch-and-price node."""
        if not bap_node.constraints:
            return
        
        for constraint_data in bap_node.constraints:
            if constraint_data.get('type') == 'branching':
                variable = constraint_data.get('variable')
                value = constraint_data.get('value', 0)
                direction = constraint_data.get('direction', '=0')
                
                if variable in self.variables:
                    if direction == '=0':
                        # Variable must be 0
                        self.model.addConstr(
                            self.variables[variable] == 0,
                            name=f"branch_{variable}_eq_0"
                        )
                    elif direction == '=1':
                        # Variable must be 1
                        self.model.addConstr(
                            self.variables[variable] == 1,
                            name=f"branch_{variable}_eq_1"
                        )
    
    def solve(self) -> bool:
        """
        Solve the RMP and extract dual values.
        
        Returns:
            True if solved successfully, False otherwise
        """
        if not self.model:
            log_message("No model to solve", "ERROR")
            return False
        
        try:
            self.model.optimize()
            
            if self.model.status == GRB.OPTIMAL:
                self.objective_value = self.model.objVal
                self._extract_dual_values()
                log_message(f"RMP solved optimally, objective: {self.objective_value:.2f}")
                return True
            else:
                log_message(f"RMP solve failed with status: {self.model.status}", "ERROR")
                return False
                
        except Exception as e:
            log_message(f"Error solving RMP: {str(e)}", "ERROR")
            return False
    
    def _extract_dual_values(self):
        """Extract dual values from the solved model."""
        self.dual_values = {}
        
        try:
            # Extract demand constraint duals
            for demand_id, constraint in self.constraints.get('demand', {}).items():
                self.dual_values[f"demand_{demand_id}"] = constraint.pi
            
            # Extract resource constraint duals
            for constraint_name, constraint in self.constraints.get('resource', {}).items():
                self.dual_values[constraint_name] = constraint.pi
            
            # Extract inventory constraint duals (these are the important ones for pricing)
            for constraint_name, constraint in self.constraints.get('inventory', {}).items():
                self.dual_values[constraint_name] = constraint.pi
                
                # Also store with simplified names for pricing algorithms
                if 'inventory_up_' in constraint_name:
                    station_id = constraint_name.split('_')[-1]
                    self.dual_values[f"psi_up_{station_id}"] = constraint.pi
                elif 'inventory_down_' in constraint_name:
                    station_id = constraint_name.split('_')[-1]
                    self.dual_values[f"psi_down_{station_id}"] = constraint.pi
                    
        except Exception as e:
            log_message(f"Error extracting dual values: {str(e)}", "ERROR")
    
    def get_solution(self) -> Dict[str, Any]:
        """
        Get the current solution values.
        
        Returns:
            Dictionary with solution information
        """
        if not self.model or self.model.status != GRB.OPTIMAL:
            return {}
        
        solution = {
            'objective_value': self.objective_value,
            'variable_values': {},
            'dual_values': self.dual_values.copy()
        }
        
        # Get variable values
        for var_name, var in self.variables.items():
            solution['variable_values'][var_name] = var.x
        
        return solution
    
    def is_integer_solution(self, tolerance: float = 1e-6) -> bool:
        """
        Check if the current solution is integer.
        
        Args:
            tolerance: Tolerance for integrality check
            
        Returns:
            True if solution is integer, False otherwise
        """
        if not self.model or self.model.status != GRB.OPTIMAL:
            return False
        
        for var in self.variables.values():
            if abs(var.x - round(var.x)) > tolerance:
                return False
        
        return True
    
    def get_fractional_variable(self) -> Optional[Tuple[str, float]]:
        """
        Get a fractional variable for branching.
        
        Returns:
            Tuple of (variable_name, value) or None if all integer
        """
        if not self.model or self.model.status != GRB.OPTIMAL:
            return None
        
        # Find the most fractional variable (closest to 0.5)
        best_var = None
        best_fractionality = 0.0
        
        for var_name, var in self.variables.items():
            value = var.x
            fractionality = min(value - int(value), int(value) + 1 - value)
            
            if fractionality > best_fractionality:
                best_fractionality = fractionality
                best_var = (var_name, value)
        
        return best_var if best_fractionality > 1e-6 else None
    
    def warm_start(self, solution: Dict[str, float]):
        """
        Set warm start values for variables.
        
        Args:
            solution: Dictionary mapping variable names to values
        """
        if not self.model:
            return
        
        for var_name, value in solution.items():
            if var_name in self.variables:
                self.variables[var_name].start = value
    
    def dispose(self):
        """Clean up the model."""
        if self.model:
            self.model.dispose()
            self.model = None
