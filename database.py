"""
SQLite database integration for real-time data storage and retrieval.
Handles dynamic updates to nodes, demands, vehicles, and optimization results.
"""

import sqlite3
import os
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import threading
from contextlib import contextmanager

from data_structures import NetworkNode, Vehicle, Demand, VehicleType, SolutionStats
from utils import log_message, time_to_minutes, minutes_to_time

class DatabaseManager:
    """
    SQLite database manager for Branch-and-Price solver real-time data.
    """
    
    def __init__(self, db_path: str = "logistics.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._initialize_database()
        
    def _initialize_database(self):
        """Initialize database with required tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Create nodes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    id INTEGER PRIMARY KEY,
                    x_coord REAL NOT NULL,
                    y_coord REAL NOT NULL,
                    node_type TEXT NOT NULL,
                    demand REAL DEFAULT 0.0,
                    time_window_start INTEGER DEFAULT 0,
                    time_window_end INTEGER DEFAULT 1440,
                    service_time INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create vehicles table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS vehicles (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    capacity REAL NOT NULL,
                    speed REAL NOT NULL,
                    cost_per_km REAL DEFAULT 0.0,
                    cost_per_hour REAL DEFAULT 0.0,
                    energy_capacity REAL NULL,
                    energy_consumption_rate REAL NULL,
                    status TEXT DEFAULT 'available',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create demands table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS demands (
                    id TEXT PRIMARY KEY,
                    origin INTEGER NOT NULL,
                    destination INTEGER NOT NULL,
                    quantity REAL NOT NULL,
                    time_window_start INTEGER NOT NULL,
                    time_window_end INTEGER NOT NULL,
                    priority INTEGER DEFAULT 1,
                    status TEXT DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (origin) REFERENCES nodes (id),
                    FOREIGN KEY (destination) REFERENCES nodes (id)
                )
            """)
            
            # Create metro timetable table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metro_timetable (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    line_id TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    station_id INTEGER NOT NULL,
                    arrival_time INTEGER NOT NULL,
                    departure_time INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (station_id) REFERENCES nodes (id)
                )
            """)
            
            # Create optimization results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS optimization_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT UNIQUE NOT NULL,
                    algorithm_type TEXT NOT NULL,
                    total_cost REAL NOT NULL,
                    total_runtime REAL NOT NULL,
                    num_iterations INTEGER NOT NULL,
                    final_gap REAL NOT NULL,
                    num_drone_routes INTEGER DEFAULT 0,
                    num_truck_routes INTEGER DEFAULT 0,
                    num_metro_schedules INTEGER DEFAULT 0,
                    solution_data TEXT, -- JSON
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create real-time metrics table for monitoring
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS real_time_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_type TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metric_data TEXT, -- JSON for additional data
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes (node_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_vehicles_type ON vehicles (type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_demands_status ON demands (status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metro_line_direction ON metro_timetable (line_id, direction)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON real_time_metrics (timestamp)")
            
            conn.commit()
            log_message("Database initialized successfully")
            
    @contextmanager
    def _get_connection(self):
        """Get database connection with proper locking."""
        with self.lock:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            try:
                yield conn
            finally:
                conn.close()
                
    def insert_node(self, node: NetworkNode) -> bool:
        """Insert or update a network node."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO nodes 
                    (id, x_coord, y_coord, node_type, demand, time_window_start, 
                     time_window_end, service_time, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    node.id, node.x_coord, node.y_coord, node.node_type,
                    node.demand, node.time_window_start, node.time_window_end,
                    node.service_time
                ))
                conn.commit()
                return True
        except Exception as e:
            log_message(f"Error inserting node: {str(e)}", "ERROR")
            return False
            
    def insert_vehicle(self, vehicle: Vehicle) -> bool:
        """Insert or update a vehicle."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO vehicles 
                    (id, type, capacity, speed, cost_per_km, cost_per_hour,
                     energy_capacity, energy_consumption_rate, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    vehicle.id, vehicle.type.value, vehicle.capacity, vehicle.speed,
                    vehicle.cost_per_km, vehicle.cost_per_hour,
                    vehicle.energy_capacity, vehicle.energy_consumption_rate
                ))
                conn.commit()
                return True
        except Exception as e:
            log_message(f"Error inserting vehicle: {str(e)}", "ERROR")
            return False
            
    def insert_demand(self, demand: Demand) -> bool:
        """Insert or update a demand."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO demands 
                    (id, origin, destination, quantity, time_window_start, 
                     time_window_end, priority, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    demand.id, demand.origin, demand.destination, demand.quantity,
                    demand.time_window_start, demand.time_window_end, demand.priority
                ))
                conn.commit()
                return True
        except Exception as e:
            log_message(f"Error inserting demand: {str(e)}", "ERROR")
            return False
            
    def get_nodes(self) -> List[NetworkNode]:
        """Retrieve all nodes from database."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM nodes ORDER BY id")
                rows = cursor.fetchall()
                
                nodes = []
                for row in rows:
                    node = NetworkNode(
                        id=row['id'],
                        x_coord=row['x_coord'],
                        y_coord=row['y_coord'],
                        node_type=row['node_type'],
                        demand=row['demand'],
                        time_window_start=row['time_window_start'],
                        time_window_end=row['time_window_end'],
                        service_time=row['service_time']
                    )
                    nodes.append(node)
                
                return nodes
        except Exception as e:
            log_message(f"Error retrieving nodes: {str(e)}", "ERROR")
            return []
            
    def get_vehicles(self) -> List[Vehicle]:
        """Retrieve all vehicles from database."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM vehicles ORDER BY id")
                rows = cursor.fetchall()
                
                vehicles = []
                for row in rows:
                    vehicle_type = VehicleType.TRUCK
                    if row['type'] == 'drone':
                        vehicle_type = VehicleType.DRONE
                    elif row['type'] == 'metro':
                        vehicle_type = VehicleType.METRO
                    
                    vehicle = Vehicle(
                        id=row['id'],
                        type=vehicle_type,
                        capacity=row['capacity'],
                        speed=row['speed'],
                        cost_per_km=row['cost_per_km'],
                        cost_per_hour=row['cost_per_hour'],
                        energy_capacity=row['energy_capacity'],
                        energy_consumption_rate=row['energy_consumption_rate']
                    )
                    vehicles.append(vehicle)
                
                return vehicles
        except Exception as e:
            log_message(f"Error retrieving vehicles: {str(e)}", "ERROR")
            return []
            
    def get_demands(self, status: Optional[str] = None) -> List[Demand]:
        """Retrieve demands from database."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                if status:
                    cursor.execute("SELECT * FROM demands WHERE status = ? ORDER BY created_at", (status,))
                else:
                    cursor.execute("SELECT * FROM demands ORDER BY created_at")
                
                rows = cursor.fetchall()
                
                demands = []
                for row in rows:
                    demand = Demand(
                        id=row['id'],
                        origin=row['origin'],
                        destination=row['destination'],
                        quantity=row['quantity'],
                        time_window_start=row['time_window_start'],
                        time_window_end=row['time_window_end'],
                        priority=row['priority']
                    )
                    demands.append(demand)
                
                return demands
        except Exception as e:
            log_message(f"Error retrieving demands: {str(e)}", "ERROR")
            return []
            
    def save_optimization_result(self, run_id: str, algorithm_type: str, 
                               stats: SolutionStats, solution_data: Dict[str, Any]) -> bool:
        """Save optimization results to database."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO optimization_results 
                    (run_id, algorithm_type, total_cost, total_runtime, num_iterations,
                     final_gap, num_drone_routes, num_truck_routes, num_metro_schedules,
                     solution_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_id, algorithm_type, stats.total_cost, stats.total_runtime,
                    stats.num_iterations, stats.final_gap, stats.num_drone_routes,
                    stats.num_truck_routes, stats.num_metro_schedules,
                    json.dumps(solution_data)
                ))
                conn.commit()
                log_message(f"Saved optimization result for run {run_id}")
                return True
        except Exception as e:
            log_message(f"Error saving optimization result: {str(e)}", "ERROR")
            return False
            
    def get_latest_optimization_results(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get latest optimization results."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM optimization_results 
                    ORDER BY created_at DESC LIMIT ?
                """, (limit,))
                rows = cursor.fetchall()
                
                results = []
                for row in rows:
                    result = dict(row)
                    if result['solution_data']:
                        result['solution_data'] = json.loads(result['solution_data'])
                    results.append(result)
                
                return results
        except Exception as e:
            log_message(f"Error retrieving optimization results: {str(e)}", "ERROR")
            return []
            
    def log_real_time_metric(self, metric_type: str, metric_value: float, 
                           metric_data: Optional[Dict[str, Any]] = None) -> bool:
        """Log real-time performance metrics."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO real_time_metrics (metric_type, metric_value, metric_data)
                    VALUES (?, ?, ?)
                """, (
                    metric_type, metric_value,
                    json.dumps(metric_data) if metric_data else None
                ))
                conn.commit()
                return True
        except Exception as e:
            log_message(f"Error logging metric: {str(e)}", "ERROR")
            return False
            
    def get_real_time_metrics(self, metric_type: str, 
                            since: Optional[datetime] = None,
                            limit: int = 100) -> List[Dict[str, Any]]:
        """Get real-time metrics."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                if since:
                    cursor.execute("""
                        SELECT * FROM real_time_metrics 
                        WHERE metric_type = ? AND timestamp >= ?
                        ORDER BY timestamp DESC LIMIT ?
                    """, (metric_type, since.isoformat(), limit))
                else:
                    cursor.execute("""
                        SELECT * FROM real_time_metrics 
                        WHERE metric_type = ?
                        ORDER BY timestamp DESC LIMIT ?
                    """, (metric_type, limit))
                
                rows = cursor.fetchall()
                
                metrics = []
                for row in rows:
                    metric = dict(row)
                    if metric['metric_data']:
                        metric['metric_data'] = json.loads(metric['metric_data'])
                    metrics.append(metric)
                
                return metrics
        except Exception as e:
            log_message(f"Error retrieving metrics: {str(e)}", "ERROR")
            return []
            
    def update_demand_status(self, demand_id: str, status: str) -> bool:
        """Update demand status (e.g., 'assigned', 'completed', 'failed')."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE demands SET status = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (status, demand_id))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            log_message(f"Error updating demand status: {str(e)}", "ERROR")
            return False
            
    def update_vehicle_status(self, vehicle_id: str, status: str) -> bool:
        """Update vehicle status (e.g., 'available', 'busy', 'maintenance')."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE vehicles SET status = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (status, vehicle_id))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            log_message(f"Error updating vehicle status: {str(e)}", "ERROR")
            return False
            
    def cleanup_old_metrics(self, days_to_keep: int = 30) -> bool:
        """Clean up old real-time metrics to prevent database bloat."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM real_time_metrics 
                    WHERE timestamp < ?
                """, (cutoff_date.isoformat(),))
                conn.commit()
                log_message(f"Cleaned up {cursor.rowcount} old metric records")
                return True
        except Exception as e:
            log_message(f"Error cleaning up metrics: {str(e)}", "ERROR")
            return False
            
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Count records in each table
                tables = ['nodes', 'vehicles', 'demands', 'metro_timetable', 
                         'optimization_results', 'real_time_metrics']
                
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
                    stats[f"{table}_count"] = cursor.fetchone()['count']
                
                # Get database file size
                if os.path.exists(self.db_path):
                    stats['db_size_mb'] = round(os.path.getsize(self.db_path) / (1024 * 1024), 2)
                else:
                    stats['db_size_mb'] = 0
                
                return stats
        except Exception as e:
            log_message(f"Error getting database stats: {str(e)}", "ERROR")
            return {}

# Global database instance
db_manager = DatabaseManager()