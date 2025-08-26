"""
Gurobi optimizer configuration and license management.
Provides centralized configuration for Gurobi optimization parameters.
"""

import os
import gurobipy as gp
from gurobipy import GRB
from typing import Dict, Any, Optional
from utils import log_message

class GurobiConfig:
    """
    Centralized Gurobi configuration and optimization parameters.
    """
    
    def __init__(self):
        self.license_configured = False
        self.optimization_params = self._get_default_params()
        self._check_license()
        
    def _check_license(self):
        """Check if Gurobi license is properly configured."""
        try:
            # Try to create a simple model to test license
            test_model = gp.Model("license_test")
            test_model.setParam('OutputFlag', 0)
            x = test_model.addVar(name="test_var")
            test_model.setObjective(x, GRB.MINIMIZE)
            test_model.optimize()
            test_model.dispose()
            
            self.license_configured = True
            log_message("Gurobi license validated successfully")
            
        except Exception as e:
            log_message(f"Gurobi license validation failed: {str(e)}", "ERROR")
            log_message("Note: Academic users can get free license at https://www.gurobi.com/academia/", "INFO")
            self.license_configured = False
            
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default Gurobi optimization parameters."""
        return {
            # Output control
            'OutputFlag': 1,  # Enable output by default
            'LogToConsole': 1,  # Log to console
            
            # Algorithm selection
            'Method': 1,  # Use dual simplex for LP
            'Presolve': 2,  # Aggressive presolve
            
            # Tolerances
            'OptimalityTol': 1e-6,  # Optimality tolerance
            'FeasibilityTol': 1e-6,  # Feasibility tolerance
            'IntFeasTol': 1e-5,  # Integer feasibility tolerance
            'MIPGap': 1e-4,  # MIP gap tolerance
            'MIPGapAbs': 1e-10,  # Absolute MIP gap
            
            # Time limits
            'TimeLimit': 300,  # 5 minutes time limit
            
            # Threading
            'Threads': 0,  # Use all available cores
            
            # Memory management
            'NodefileStart': 0.5,  # Start using disk at 0.5GB
            'NodefileDir': './gurobi_nodefiles',  # Directory for node files
            
            # Cutting planes
            'Cuts': 2,  # Aggressive cuts
            'CutPasses': 5,  # Maximum cutting plane passes
            
            # Heuristics
            'Heuristics': 0.05,  # 5% time on heuristics
            
            # Branching
            'VarBranch': 3,  # Strong branching
            'BranchDir': 0,  # Branch down first
            
            # Symmetry
            'Symmetry': 2,  # Aggressive symmetry detection
            
            # Other parameters
            'Aggregate': 1,  # Enable aggregation
            'PrePasses': 5,  # Presolve passes
            'Crossover': 0,  # Disable crossover for faster LP solving
        }
        
    def get_optimized_params_for_problem_type(self, problem_type: str) -> Dict[str, Any]:
        """
        Get optimized parameters based on problem type.
        
        Args:
            problem_type: Type of problem ('LP', 'MIP', 'RMP', 'Pricing')
            
        Returns:
            Dictionary of optimized Gurobi parameters
        """
        params = self.optimization_params.copy()
        
        if problem_type.upper() == 'RMP':
            # Parameters optimized for Restricted Master Problem
            params.update({
                'Method': 1,  # Dual simplex best for RMP
                'Presolve': 1,  # Conservative presolve to preserve dual info
                'Crossover': 0,  # No crossover needed for dual values
                'ScaleFlag': 2,  # Aggressive scaling
                'NumericFocus': 0,  # Speed over numerical precision
                'OptimalityTol': 1e-6,
                'FeasibilityTol': 1e-6
            })
            
        elif problem_type.upper() == 'PRICING':
            # Parameters for pricing subproblems (usually small LPs)
            params.update({
                'Method': 1,  # Dual simplex
                'Presolve': 0,  # No presolve for small problems
                'Crossover': 0,  # No crossover
                'Threads': 1,  # Single thread for small problems
                'OutputFlag': 0,  # No output for pricing
                'OptimalityTol': 1e-8,  # Higher precision for pricing
                'FeasibilityTol': 1e-8
            })
            
        elif problem_type.upper() == 'MIP':
            # Parameters for Mixed Integer Programming
            params.update({
                'Method': -1,  # Automatic method selection
                'MIPFocus': 1,  # Focus on finding feasible solutions
                'ImproveStartTime': 60,  # Start improvement after 60s
                'ImproveStartGap': 0.1,  # Start improvement at 10% gap
                'Heuristics': 0.1,  # More time on heuristics
                'VarBranch': 3,  # Strong branching
                'NodeMethod': 1,  # Dual simplex at nodes
            })
            
        elif problem_type.upper() == 'LP':
            # Parameters for Linear Programming
            params.update({
                'Method': 1,  # Dual simplex
                'Presolve': 2,  # Aggressive presolve
                'Crossover': 1,  # Enable crossover for basic solution
                'ScaleFlag': 2,  # Aggressive scaling
                'Quad': 1,  # Use quad precision if needed
            })
            
        return params
        
    def create_model(self, name: str, problem_type: str = 'LP', 
                    silent: bool = False) -> Optional[gp.Model]:
        """
        Create a Gurobi model with optimized parameters.
        
        Args:
            name: Model name
            problem_type: Type of problem for parameter optimization
            silent: Whether to suppress output
            
        Returns:
            Configured Gurobi model or None if license issues
        """
        if not self.license_configured:
            log_message("Cannot create model: Gurobi license not configured", "ERROR")
            return None
            
        try:
            model = gp.Model(name)
            
            # Apply optimized parameters
            params = self.get_optimized_params_for_problem_type(problem_type)
            
            if silent:
                params['OutputFlag'] = 0
                params['LogToConsole'] = 0
                
            for param, value in params.items():
                try:
                    model.setParam(param, value)
                except Exception as e:
                    log_message(f"Warning: Could not set parameter {param}={value}: {str(e)}", "WARNING")
                    
            return model
            
        except Exception as e:
            log_message(f"Error creating Gurobi model: {str(e)}", "ERROR")
            return None
            
    def setup_environment_variables(self):
        """Setup Gurobi environment variables for optimal performance."""
        try:
            # Create node file directory
            nodefiles_dir = './gurobi_nodefiles'
            if not os.path.exists(nodefiles_dir):
                os.makedirs(nodefiles_dir)
                log_message(f"Created Gurobi nodefiles directory: {nodefiles_dir}")
                
            # Set environment variables for Gurobi
            env_vars = {
                'GRB_LICENSE_FILE': os.environ.get('GRB_LICENSE_FILE', ''),
                'GUROBI_HOME': os.environ.get('GUROBI_HOME', ''),
            }
            
            for var, value in env_vars.items():
                if value:
                    log_message(f"Gurobi environment: {var} = {value}")
                else:
                    log_message(f"Gurobi environment: {var} not set", "WARNING")
                    
        except Exception as e:
            log_message(f"Error setting up Gurobi environment: {str(e)}", "ERROR")
            
    def get_license_info(self) -> Dict[str, Any]:
        """Get information about the current Gurobi license."""
        info = {
            'license_configured': self.license_configured,
            'license_file': os.environ.get('GRB_LICENSE_FILE', 'Not set'),
            'gurobi_home': os.environ.get('GUROBI_HOME', 'Not set'),
        }
        
        if self.license_configured:
            try:
                # Get license details from a test model
                model = gp.Model("license_info")
                model.setParam('OutputFlag', 0)
                
                # Try to get version info
                info['gurobi_version'] = f"{gp.gurobi.version()[0]}.{gp.gurobi.version()[1]}.{gp.gurobi.version()[2]}"
                
                model.dispose()
            except Exception as e:
                info['error'] = str(e)
                
        return info
        
    def tune_parameters(self, model: gp.Model, time_limit: int = 60) -> Dict[str, Any]:
        """
        Use Gurobi's automatic parameter tuning.
        
        Args:
            model: Gurobi model to tune
            time_limit: Time limit for tuning in seconds
            
        Returns:
            Dictionary of tuned parameters
        """
        if not self.license_configured:
            return {}
            
        try:
            log_message(f"Starting Gurobi parameter tuning (time limit: {time_limit}s)")
            
            # Set tuning time limit
            model.setParam('TuneTimeLimit', time_limit)
            model.setParam('TuneTrials', 3)  # Number of different parameter sets to try
            
            # Run parameter tuning
            model.tune()
            
            # Get the best parameter set
            if model.tuneResultCount > 0:
                model.getTuneResult(0)  # Get best result
                
                # Extract tuned parameters
                tuned_params = {}
                for param in model.getParams():
                    if hasattr(param, 'getAttr'):
                        try:
                            name = param.getAttr('ParamName')
                            value = model.getParamInfo(name)[2]  # Current value
                            tuned_params[name] = value
                        except:
                            continue
                            
                log_message(f"Parameter tuning completed. Found {len(tuned_params)} tuned parameters")
                return tuned_params
            else:
                log_message("No tuning results found", "WARNING")
                return {}
                
        except Exception as e:
            log_message(f"Error during parameter tuning: {str(e)}", "ERROR")
            return {}
            
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get Gurobi performance statistics."""
        stats = {
            'license_configured': self.license_configured,
            'default_threads': self.optimization_params.get('Threads', 0),
            'time_limit': self.optimization_params.get('TimeLimit', 300),
            'mip_gap': self.optimization_params.get('MIPGap', 1e-4),
        }
        
        if self.license_configured:
            try:
                # Test model creation and solving speed
                import time
                start_time = time.time()
                
                test_model = gp.Model("perf_test")
                test_model.setParam('OutputFlag', 0)
                
                # Create a small test problem
                x = test_model.addVars(10, name="x")
                test_model.setObjective(gp.quicksum(x[i] for i in range(10)), GRB.MINIMIZE)
                test_model.addConstr(gp.quicksum(x[i] for i in range(10)) >= 1)
                
                test_model.optimize()
                test_model.dispose()
                
                creation_time = time.time() - start_time
                stats['model_creation_time'] = round(creation_time, 4)
                
            except Exception as e:
                stats['performance_error'] = str(e)
                
        return stats

# Global Gurobi configuration instance
gurobi_config = GurobiConfig()