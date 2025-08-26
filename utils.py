"""
Utility functions for the Branch-and-Price solver.
"""

import math
from typing import Tuple, List, Dict, Any
import numpy as np

def time_to_minutes(time_str: str) -> int:
    """
    Convert time string (HH:MM or HH:MM:SS) to minutes from 00:00.
    
    Args:
        time_str: Time string in format "HH:MM" or "HH:MM:SS"
        
    Returns:
        Minutes from 00:00
    """
    if isinstance(time_str, (int, float)):
        return int(time_str)
    
    if isinstance(time_str, str):
        parts = time_str.split(':')
        if len(parts) >= 2:
            hours = int(parts[0])
            minutes = int(parts[1])
            return hours * 60 + minutes
    
    return 0

def minutes_to_time(minutes: int) -> str:
    """
    Convert minutes from 00:00 to time string HH:MM.
    
    Args:
        minutes: Minutes from 00:00
        
    Returns:
        Time string in format "HH:MM"
    """
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours:02d}:{mins:02d}"

def euclidean_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two coordinates.
    
    Args:
        coord1: (x, y) coordinates of first point
        coord2: (x, y) coordinates of second point
        
    Returns:
        Euclidean distance
    """
    return math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

def travel_time(distance: float, speed: float) -> int:
    """
    Calculate travel time in minutes given distance and speed.
    
    Args:
        distance: Distance in km
        speed: Speed in km/h
        
    Returns:
        Travel time in minutes
    """
    if speed <= 0:
        return float('inf')
    return int(math.ceil(distance / speed * 60))

def normalize_header(header: str) -> str:
    """
    Normalize Excel header from Chinese to English.
    
    Args:
        header: Original header string
        
    Returns:
        Normalized header string
    """
    from config import HEADER_MAPPINGS
    
    # Clean header string
    header = str(header).strip()
    
    # Check for direct mapping
    if header in HEADER_MAPPINGS:
        return HEADER_MAPPINGS[header]
    
    # Convert to lowercase and replace spaces/special chars
    normalized = header.lower().replace(' ', '_').replace('-', '_')
    
    # Remove special characters
    import re
    normalized = re.sub(r'[^\w]', '_', normalized)
    
    return normalized

def create_time_slices(start_time: int, end_time: int, slice_duration: int) -> List[Tuple[int, int]]:
    """
    Create time slices for resource constraints.
    
    Args:
        start_time: Start time in minutes
        end_time: End time in minutes  
        slice_duration: Duration of each slice in minutes
        
    Returns:
        List of (start, end) time slice tuples
    """
    slices = []
    current_time = start_time
    
    while current_time < end_time:
        slice_end = min(current_time + slice_duration, end_time)
        slices.append((current_time, slice_end))
        current_time = slice_end
    
    return slices

def dominance_check(label1: Dict[str, Any], label2: Dict[str, Any], 
                   resources: List[str]) -> bool:
    """
    Check if label1 dominates label2 based on resource consumption.
    
    Args:
        label1: First label to compare
        label2: Second label to compare
        resources: List of resource names to compare
        
    Returns:
        True if label1 dominates label2
    """
    # label1 dominates label2 if it's better or equal in all resources
    # and strictly better in at least one
    
    better_in_one = False
    
    for resource in resources:
        val1 = label1.get(resource, 0)
        val2 = label2.get(resource, 0)
        
        if val1 > val2:  # label1 is worse in this resource
            return False
        elif val1 < val2:  # label1 is better in this resource
            better_in_one = True
    
    return better_in_one

def calculate_reduced_cost(column_cost: float, dual_values: Dict[str, float], 
                         column_details: Dict[str, Any]) -> float:
    """
    Calculate reduced cost for a column given dual values.
    
    Args:
        column_cost: Direct cost of the column
        dual_values: Dictionary of dual values from RMP
        column_details: Column details including coefficients
        
    Returns:
        Reduced cost
    """
    reduced_cost = column_cost
    
    # Subtract dual values for constraints this column participates in
    for constraint, coefficient in column_details.items():
        if constraint in dual_values:
            reduced_cost -= dual_values[constraint] * coefficient
    
    return reduced_cost

def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Perform safe division with default value for zero denominator.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value if denominator is zero
        
    Returns:
        Division result or default value
    """
    if abs(denominator) < 1e-10:
        return default
    return numerator / denominator

def log_message(message: str, level: str = "INFO") -> None:
    """
    Log a message with timestamp and level.
    
    Args:
        message: Message to log
        level: Log level (INFO, WARNING, ERROR, DEBUG)
    """
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {level}: {message}")
