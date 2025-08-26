"""
Configuration constants for the Branch-and-Price solver.
"""

# Time resolution constants
INTERNAL_TIME_RESOLUTION = 1  # minutes
RESOURCE_TIME_SLICE = 15  # minutes

# Default values for missing parameters
DEFAULT_DRONE_CAPACITY = 10.0  # kg
DEFAULT_TRUCK_CAPACITY = 1000.0  # kg
DEFAULT_DRONE_ENERGY = 100.0  # battery units
DEFAULT_DRONE_SPEED = 50.0  # km/h
DEFAULT_TRUCK_SPEED = 30.0  # km/h

# Metro constants
METRO_MAX_CAPACITY = 5000.0  # kg per train

# Optimization parameters
COLUMN_GENERATION_TOLERANCE = 1e-6
BRANCHING_TOLERANCE = 1e-6
MAX_ITERATIONS = 1000

# Excel column mappings for header normalization
HEADER_MAPPINGS = {
    # Chinese to English mappings
    '节点ID': 'node_id',
    '需求': 'demand',
    '时间': 'time',
    '距离': 'distance',
    '容量': 'capacity',
    '坐标X': 'x_coord',
    '坐标Y': 'y_coord',
    '开始时间': 'start_time',
    '结束时间': 'end_time',
    '车辆类型': 'vehicle_type',
    '路线ID': 'route_id',
    '成本': 'cost',
    '能耗': 'energy_consumption',
    # Add more mappings as needed
}

# File paths for data and output
DATA_DIR = "data"
OUTPUT_DIR = "output"

# Required input files
INPUT_FILES = [
    "nodes.xlsx",
    "demands.xlsx", 
    "vehicles.xlsx",
    "metro_timetable.xlsx",
    "parameters.xlsx"
]

# Output file names
OUTPUT_FILES = {
    "metro_timetables": "metro_timetables.xlsx",
    "drone_routes": "drone_routes.xlsx",
    "truck_routes": "truck_routes.xlsx",
    "metro_shipments": "metro_shipments.xlsx",
    "bounds_log": "bounds_log.xlsx",
    "runtime_comparison": "runtime_comparison.xlsx",
    "summary": "solution_summary.xlsx"
}
