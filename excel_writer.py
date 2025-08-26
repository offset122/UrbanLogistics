"""
Excel output writer for Branch-and-Price solver results.
Generates all required Excel outputs as specified in the requirements.
"""

import os
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime

from data_structures import SolutionStats
from utils import log_message, minutes_to_time
from config import OUTPUT_FILES

class ExcelWriter:
    """
    Handles generation of all Excel output files for the Branch-and-Price solver.
    """
    
    def __init__(self, data_loader):
        self.data_loader = data_loader
        
    def export_solution_summary(self, stats: SolutionStats, output_dir: str):
        """
        Export solution summary to Excel.
        
        Args:
            stats: Solution statistics
            output_dir: Output directory path
        """
        try:
            summary_data = stats.to_dict()
            
            # Add additional summary information
            summary_data.update({
                'Solver Type': 'Branch-and-Price',
                'Instance Name': 'Urban Logistics Problem',
                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Number of Nodes': len(self.data_loader.nodes),
                'Number of Demands': len(self.data_loader.demands),
                'Number of Vehicles': len(self.data_loader.vehicles),
                'Optimization Gap (%)': stats.final_gap,
                'Status': 'Optimal' if stats.final_gap < 1.0 else 'Near-Optimal'
            })
            
            # Convert to DataFrame
            summary_df = pd.DataFrame([summary_data])
            
            # Export to Excel
            output_path = os.path.join(output_dir, OUTPUT_FILES['summary'])
            summary_df.to_excel(output_path, index=False, sheet_name='Summary')
            
            log_message(f"Solution summary exported to {output_path}")
            
        except Exception as e:
            log_message(f"Error exporting solution summary: {str(e)}", "ERROR")
    
    def export_bounds_log(self, iteration_log: List[Dict[str, Any]], output_dir: str):
        """
        Export iteration bounds log to Excel.
        
        Args:
            iteration_log: List of iteration data
            output_dir: Output directory path
        """
        try:
            if not iteration_log:
                log_message("No iteration log data to export", "WARNING")
                return
            
            # Convert to DataFrame
            bounds_df = pd.DataFrame(iteration_log)
            
            # Add formatted columns
            bounds_df['Runtime (min)'] = bounds_df['runtime'] / 60.0
            bounds_df['Gap (%)'] = bounds_df['gap'].round(2)
            bounds_df['Lower Bound'] = bounds_df['lower_bound'].round(2)
            bounds_df['Best Integer'] = bounds_df['best_integer'].round(2)
            
            # Reorder columns for better readability
            column_order = [
                'iteration', 'node_id', 'node_depth', 'Lower Bound', 'Best Integer',
                'Gap (%)', 'num_columns', 'is_integer', 'Runtime (min)'
            ]
            bounds_df = bounds_df[column_order]
            
            # Export to Excel
            output_path = os.path.join(output_dir, OUTPUT_FILES['bounds_log'])
            
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                bounds_df.to_excel(writer, index=False, sheet_name='Bounds_Log')
                
                # Add a chart sheet if possible
                try:
                    workbook = writer.book
                    chart_sheet = workbook.create_sheet("Charts")
                    
                    # Create bounds evolution chart data
                    chart_data = bounds_df[['iteration', 'Lower Bound', 'Best Integer']].copy()
                    chart_data.to_excel(writer, index=False, sheet_name='Chart_Data')
                    
                except Exception:
                    pass  # Charts are optional
            
            log_message(f"Bounds log exported to {output_path}")
            
        except Exception as e:
            log_message(f"Error exporting bounds log: {str(e)}", "ERROR")
    
    def export_runtime_comparison(self, runtime_data: Dict[str, Any], output_dir: str):
        """
        Export runtime comparison between basic and optimized algorithms.
        
        Args:
            runtime_data: Runtime comparison data
            output_dir: Output directory path
        """
        try:
            # Create comparison DataFrame
            comparison_df = pd.DataFrame([runtime_data])
            
            # Add performance metrics
            comparison_df['Runtime (seconds)'] = comparison_df['total_runtime'].round(2)
            comparison_df['Runtime (minutes)'] = (comparison_df['total_runtime'] / 60.0).round(2)
            comparison_df['Cost'] = comparison_df['final_cost'].round(2)
            comparison_df['Gap (%)'] = comparison_df['final_gap'].round(2)
            comparison_df['Columns Generated'] = comparison_df['columns_generated']
            comparison_df['Iterations'] = comparison_df['iterations']
            
            # Calculate efficiency metrics
            if runtime_data['iterations'] > 0:
                comparison_df['Avg Time per Iteration (s)'] = (
                    comparison_df['total_runtime'] / comparison_df['iterations']
                ).round(3)
                comparison_df['Columns per Iteration'] = (
                    comparison_df['columns_generated'] / comparison_df['iterations']
                ).round(1)
            
            # Select relevant columns
            output_columns = [
                'algorithm_type', 'Runtime (seconds)', 'Runtime (minutes)', 
                'Iterations', 'Cost', 'Gap (%)', 'Columns Generated',
                'Avg Time per Iteration (s)', 'Columns per Iteration'
            ]
            comparison_df = comparison_df[output_columns]
            
            # Export to Excel
            output_path = os.path.join(output_dir, OUTPUT_FILES['runtime_comparison'])
            comparison_df.to_excel(output_path, index=False, sheet_name='Runtime_Comparison')
            
            log_message(f"Runtime comparison exported to {output_path}")
            
        except Exception as e:
            log_message(f"Error exporting runtime comparison: {str(e)}", "ERROR")
    
    def export_truck_routes(self, truck_routes: List[Dict[str, Any]], output_dir: str):
        """
        Export truck routes to Excel.
        
        Args:
            truck_routes: List of truck route data
            output_dir: Output directory path
        """
        try:
            if not truck_routes:
                log_message("No truck routes to export", "WARNING")
                # Create empty file
                empty_df = pd.DataFrame(columns=[
                    'Route ID', 'Vehicle Type', 'Route', 'Total Cost', 
                    'Cargo Up', 'Cargo Down', 'Demands Served'
                ])
                output_path = os.path.join(output_dir, OUTPUT_FILES['truck_routes'])
                empty_df.to_excel(output_path, index=False)
                return
            
            # Process truck routes data
            routes_data = []
            for route in truck_routes:
                route_data = {
                    'Route ID': route['route_id'],
                    'Vehicle Type': route['vehicle_type'],
                    'Route': ' -> '.join([str(node) for node in route['route']]),
                    'Total Cost': round(route['total_cost'], 2),
                    'Cargo Up (kg)': round(route['cargo_up'], 1),
                    'Cargo Down (kg)': round(route['cargo_down'], 1),
                    'Demands Served': ', '.join(route['demands_served']),
                    'Number of Stops': len(route['route']) - 1,  # Exclude depot returns
                    'Route Length': len(route['route'])
                }
                routes_data.append(route_data)
            
            # Create DataFrame
            routes_df = pd.DataFrame(routes_data)
            
            # Create detailed route breakdown
            detailed_data = []
            for route in truck_routes:
                for i, node_id in enumerate(route['route']):
                    node = self.data_loader.nodes.get(node_id)
                    detailed_data.append({
                        'Route ID': route['route_id'],
                        'Stop Number': i + 1,
                        'Node ID': node_id,
                        'Node Type': node.node_type if node else 'Unknown',
                        'Demand': node.demand if node else 0,
                        'Cumulative Load': sum(
                            self.data_loader.nodes.get(n, type('', (), {'demand': 0})()).demand 
                            for n in route['route'][:i+1] if n != 0
                        )
                    })
            
            detailed_df = pd.DataFrame(detailed_data)
            
            # Export to Excel with multiple sheets
            output_path = os.path.join(output_dir, OUTPUT_FILES['truck_routes'])
            
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                routes_df.to_excel(writer, index=False, sheet_name='Routes_Summary')
                detailed_df.to_excel(writer, index=False, sheet_name='Route_Details')
            
            log_message(f"Truck routes exported to {output_path}")
            
        except Exception as e:
            log_message(f"Error exporting truck routes: {str(e)}", "ERROR")
    
    def export_drone_routes(self, drone_routes: List[Dict[str, Any]], output_dir: str):
        """
        Export drone routes with energy profiles to Excel.
        
        Args:
            drone_routes: List of drone route data
            output_dir: Output directory path
        """
        try:
            if not drone_routes:
                log_message("No drone routes to export", "WARNING")
                # Create empty file
                empty_df = pd.DataFrame(columns=[
                    'Route ID', 'Vehicle Type', 'Route', 'Total Cost', 
                    'Energy Profile', 'Cargo Up', 'Cargo Down', 'Demands Served'
                ])
                output_path = os.path.join(output_dir, OUTPUT_FILES['drone_routes'])
                empty_df.to_excel(output_path, index=False)
                return
            
            # Process drone routes data
            routes_data = []
            energy_data = []
            
            for route in drone_routes:
                route_data = {
                    'Route ID': route['route_id'],
                    'Vehicle Type': route['vehicle_type'],
                    'Route': ' -> '.join([str(node) for node in route['route']]),
                    'Total Cost': round(route['total_cost'], 2),
                    'Cargo Up (kg)': round(route['cargo_up'], 1),
                    'Cargo Down (kg)': round(route['cargo_down'], 1),
                    'Demands Served': ', '.join(route['demands_served']),
                    'Number of Stops': len(route['route']) - 1,
                    'Energy Consumption': 'See Energy Profile sheet'
                }
                routes_data.append(route_data)
                
                # Process energy profile
                for energy_point in route.get('energy_profile', []):
                    energy_data.append({
                        'Route ID': route['route_id'],
                        'Node ID': energy_point['node'],
                        'Time': minutes_to_time(energy_point['time']),
                        'Time (minutes)': energy_point['time'],
                        'Energy Remaining': round(energy_point['energy'], 1),
                        'Load (kg)': round(energy_point['load'], 1),
                        'Energy %': round(energy_point['energy'] / 100.0 * 100, 1)  # Assuming 100 is max
                    })
            
            # Create DataFrames
            routes_df = pd.DataFrame(routes_data)
            energy_df = pd.DataFrame(energy_data)
            
            # Export to Excel with multiple sheets
            output_path = os.path.join(output_dir, OUTPUT_FILES['drone_routes'])
            
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                routes_df.to_excel(writer, index=False, sheet_name='Routes_Summary')
                if not energy_df.empty:
                    energy_df.to_excel(writer, index=False, sheet_name='Energy_Profile')
            
            log_message(f"Drone routes exported to {output_path}")
            
        except Exception as e:
            log_message(f"Error exporting drone routes: {str(e)}", "ERROR")
    
    def export_metro_timetables(self, metro_schedules: List[Dict[str, Any]], output_dir: str):
        """
        Export metro timetables to Excel.
        
        Args:
            metro_schedules: List of metro schedule data
            output_dir: Output directory path
        """
        try:
            if not metro_schedules:
                log_message("No metro schedules to export", "WARNING")
                # Create empty file with sample structure
                empty_df = pd.DataFrame(columns=[
                    'Schedule ID', 'Line ID', 'Direction', 'Station ID', 
                    'Departure Time', 'Arrival Time', 'Total Cost'
                ])
                output_path = os.path.join(output_dir, OUTPUT_FILES['metro_timetables'])
                empty_df.to_excel(output_path, index=False)
                return
            
            # Process metro schedules
            timetable_data = []
            summary_data = []
            
            for schedule in metro_schedules:
                summary_data.append({
                    'Schedule ID': schedule['schedule_id'],
                    'Line ID': schedule['line_id'],
                    'Direction': schedule['direction'],
                    'Total Stations': len(schedule['stations']),
                    'Total Cost': round(schedule['total_cost'], 2),
                    'Start Time': minutes_to_time(schedule['schedule'][0][1]) if schedule['schedule'] else 'N/A',
                    'End Time': minutes_to_time(schedule['schedule'][-1][1]) if schedule['schedule'] else 'N/A',
                    'Duration (min)': (schedule['schedule'][-1][1] - schedule['schedule'][0][1]) if len(schedule['schedule']) > 1 else 0
                })
                
                # Process detailed timetable
                for station_id, dep_time, arr_time in schedule.get('schedule', []):
                    timetable_data.append({
                        'Schedule ID': schedule['schedule_id'],
                        'Line ID': schedule['line_id'],
                        'Direction': schedule['direction'],
                        'Station ID': station_id,
                        'Arrival Time': minutes_to_time(arr_time),
                        'Departure Time': minutes_to_time(dep_time),
                        'Arrival (minutes)': arr_time,
                        'Departure (minutes)': dep_time,
                        'Dwell Time (min)': dep_time - arr_time
                    })
            
            # Create DataFrames
            summary_df = pd.DataFrame(summary_data)
            timetable_df = pd.DataFrame(timetable_data)
            
            # Group by line and direction for organized output
            up_schedules = timetable_df[timetable_df['Direction'] == 'up']
            down_schedules = timetable_df[timetable_df['Direction'] == 'down']
            
            # Export to Excel with multiple sheets
            output_path = os.path.join(output_dir, OUTPUT_FILES['metro_timetables'])
            
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                summary_df.to_excel(writer, index=False, sheet_name='Schedule_Summary')
                
                if not up_schedules.empty:
                    up_schedules.to_excel(writer, index=False, sheet_name='Up_Direction')
                
                if not down_schedules.empty:
                    down_schedules.to_excel(writer, index=False, sheet_name='Down_Direction')
                
                if not timetable_df.empty:
                    timetable_df.to_excel(writer, index=False, sheet_name='All_Schedules')
            
            log_message(f"Metro timetables exported to {output_path}")
            
        except Exception as e:
            log_message(f"Error exporting metro timetables: {str(e)}", "ERROR")
    
    def export_metro_shipments(self, metro_shipments: List[Dict[str, Any]], output_dir: str):
        """
        Export metro shipments (cargo per train) to Excel.
        
        Args:
            metro_shipments: List of metro shipment data
            output_dir: Output directory path
        """
        try:
            if not metro_shipments:
                log_message("No metro shipments to export", "WARNING")
                # Create empty file
                empty_df = pd.DataFrame(columns=[
                    'Schedule ID', 'Station ID', 'Direction', 'Cargo Amount (kg)'
                ])
                output_path = os.path.join(output_dir, OUTPUT_FILES['metro_shipments'])
                empty_df.to_excel(output_path, index=False)
                return
            
            # Process shipments data
            shipments_data = []
            for shipment in metro_shipments:
                shipment_data = {
                    'Schedule ID': shipment['schedule_id'],
                    'Station ID': shipment['station_id'],
                    'Direction': shipment['direction'],
                    'Cargo Amount (kg)': round(shipment['cargo_amount'], 1),
                    'Station Name': f"Station_{shipment['station_id']}"  # Could be enhanced with real names
                }
                shipments_data.append(shipment_data)
            
            # Create DataFrame
            shipments_df = pd.DataFrame(shipments_data)
            
            # Create summary by direction
            up_shipments = shipments_df[shipments_df['Direction'] == 'up']
            down_shipments = shipments_df[shipments_df['Direction'] == 'down']
            
            # Calculate totals by station
            station_totals = shipments_df.groupby(['Station ID', 'Direction'])['Cargo Amount (kg)'].sum().reset_index()
            station_totals = station_totals.pivot(index='Station ID', columns='Direction', values='Cargo Amount (kg)').fillna(0)
            station_totals['Total Cargo'] = station_totals.sum(axis=1)
            station_totals = station_totals.reset_index()
            
            # Export to Excel with multiple sheets
            output_path = os.path.join(output_dir, OUTPUT_FILES['metro_shipments'])
            
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                shipments_df.to_excel(writer, index=False, sheet_name='All_Shipments')
                
                if not up_shipments.empty:
                    up_summary = up_shipments.groupby('Station ID')['Cargo Amount (kg)'].sum().reset_index()
                    up_summary.to_excel(writer, index=False, sheet_name='Up_Direction_Summary')
                
                if not down_shipments.empty:
                    down_summary = down_shipments.groupby('Station ID')['Cargo Amount (kg)'].sum().reset_index()
                    down_summary.to_excel(writer, index=False, sheet_name='Down_Direction_Summary')
                
                station_totals.to_excel(writer, index=False, sheet_name='Station_Totals')
            
            log_message(f"Metro shipments exported to {output_path}")
            
        except Exception as e:
            log_message(f"Error exporting metro shipments: {str(e)}", "ERROR")
    
    def export_all_results(self, results_data: Dict[str, Any], output_dir: str):
        """
        Export all results in a single call.
        
        Args:
            results_data: Dictionary containing all results data
            output_dir: Output directory path
        """
        log_message(f"Exporting all results to {output_dir}")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Export each type of result
        if 'stats' in results_data:
            self.export_solution_summary(results_data['stats'], output_dir)
        
        if 'iteration_log' in results_data:
            self.export_bounds_log(results_data['iteration_log'], output_dir)
        
        if 'runtime_comparison' in results_data:
            self.export_runtime_comparison(results_data['runtime_comparison'], output_dir)
        
        if 'truck_routes' in results_data:
            self.export_truck_routes(results_data['truck_routes'], output_dir)
        
        if 'drone_routes' in results_data:
            self.export_drone_routes(results_data['drone_routes'], output_dir)
        
        if 'metro_schedules' in results_data:
            self.export_metro_timetables(results_data['metro_schedules'], output_dir)
        
        if 'metro_shipments' in results_data:
            self.export_metro_shipments(results_data['metro_shipments'], output_dir)
        
        log_message("All results exported successfully")
