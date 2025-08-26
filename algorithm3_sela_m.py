"""
Algorithm 3: Single-arc Elementary Label-setting Algorithm for Metro (SELA-M)
Basic metro scheduling with stepwise timetable following.
"""

from typing import List, Dict, Any, Optional, Tuple
import heapq
from collections import defaultdict

from data_structures import MetroLabel, Column, ColumnType, VehicleType
from data_loader import DataLoader
from utils import log_message, calculate_reduced_cost
from config import METRO_MAX_CAPACITY

class SELAMSolver:
    """
    Single-arc Elementary Label-setting Algorithm for Metro scheduling.
    Solves Resource Constrained Shortest Path Problem (RCSPP) for metro.
    """
    
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.metro_capacity = METRO_MAX_CAPACITY
        
    def solve_pricing(self, dual_values: Dict[str, float]) -> List[Column]:
        """
        Solve metro pricing problem using label-setting algorithm.
        
        Args:
            dual_values: Dual values from RMP solution
            
        Returns:
            List of columns with negative reduced cost
        """
        log_message("Starting SELA-M pricing algorithm")
        
        negative_cost_columns = []
        
        # Find metro lines and directions
        metro_lines = self._get_metro_lines()
        
        for line_id, directions in metro_lines.items():
            for direction in directions:
                columns = self._solve_for_line_direction(line_id, direction, dual_values)
                negative_cost_columns.extend(columns)
        
        log_message(f"SELA-M found {len(negative_cost_columns)} columns with negative reduced cost")
        return negative_cost_columns
    
    def _get_metro_lines(self) -> Dict[str, List[str]]:
        """
        Get available metro lines and directions.
        
        Returns:
            Dictionary mapping line_id to list of directions
        """
        lines = defaultdict(set)
        
        for entry in self.data_loader.metro_timetables:
            lines[entry.line_id].add(entry.direction)
        
        return {line_id: list(directions) for line_id, directions in lines.items()}
    
    def _solve_for_line_direction(self, line_id: str, direction: str, 
                                 dual_values: Dict[str, float]) -> List[Column]:
        """
        Solve pricing for a specific metro line and direction.
        
        Args:
            line_id: Metro line ID
            direction: Direction ("up" or "down")
            dual_values: Dual values from RMP
            
        Returns:
            List of columns with negative reduced cost
        """
        # Get stations for this line and direction
        stations = self._get_line_stations(line_id, direction)
        
        if not stations:
            return []
        
        # Initialize labels
        labels = defaultdict(list)  # (station, time) -> list of labels
        processed = set()  # (station, time) pairs
        priority_queue = []  # Min-heap for label selection
        
        # Start from first station at first available departure
        first_station = stations[0]
        departures = self.data_loader.get_metro_departures(
            line_id, direction, first_station, 0
        )
        
        negative_columns = []
        
        for departure in departures[:10]:  # Limit to first 10 departures
            initial_label = MetroLabel(
                station=first_station,
                time=departure.departure_time,
                load=0.0,
                cost=0.0,
                schedule=[(first_station, departure.departure_time)]
            )
            
            labels[(first_station, departure.departure_time)].append(initial_label)
            heapq.heappush(priority_queue, (0.0, departure.departure_time, initial_label))
        
        while priority_queue:
            current_cost, current_time, current_label = heapq.heappop(priority_queue)
            
            # Skip if already processed
            state_key = (current_label.station, current_label.time)
            if state_key in processed:
                continue
            processed.add(state_key)
            
            # Check if we can create a complete schedule
            if len(current_label.schedule) > 1:  # At least visited one station
                complete_column = self._try_complete_schedule(
                    current_label, line_id, direction, dual_values
                )
                if complete_column and complete_column.reduced_cost < -1e-6:
                    negative_columns.append(complete_column)
            
            # Extend to next station
            self._extend_label(current_label, line_id, direction, stations, 
                             dual_values, labels, priority_queue)
        
        return negative_columns
    
    def _get_line_stations(self, line_id: str, direction: str) -> List[int]:
        """
        Get ordered list of stations for a metro line and direction.
        
        Args:
            line_id: Metro line ID
            direction: Direction ("up" or "down")
            
        Returns:
            List of station IDs in order
        """
        # Get all timetable entries for this line and direction
        entries = [entry for entry in self.data_loader.metro_timetables
                  if entry.line_id == line_id and entry.direction == direction]
        
        if not entries:
            return []
        
        # Group by departure time and get unique stations
        stations_by_time = defaultdict(set)
        for entry in entries:
            stations_by_time[entry.departure_time].add(entry.station_id)
        
        # Find a representative time with multiple stations to determine order
        station_set = set()
        for stations in stations_by_time.values():
            if len(stations) > len(station_set):
                station_set = stations
        
        # Return sorted stations (simple ordering)
        return sorted(list(station_set))
    
    def _extend_label(self, label: MetroLabel, line_id: str, direction: str,
                     stations: List[int], dual_values: Dict[str, float],
                     labels: Dict[Tuple[int, int], List[MetroLabel]], 
                     priority_queue: List):
        """
        Extend a label to the next station.
        
        Args:
            label: Current label to extend
            line_id: Metro line ID
            direction: Direction
            stations: List of stations in order
            dual_values: Dual values from RMP
            labels: Dictionary of labels by (station, time)
            priority_queue: Priority queue for label processing
        """
        current_station = label.station
        current_time = label.time
        
        # Find current station index
        try:
            current_index = stations.index(current_station)
        except ValueError:
            return  # Station not in line
        
        # Try to extend to next station
        if current_index + 1 < len(stations):
            next_station = stations[current_index + 1]
            
            # Find next departure from current station that goes to next station
            next_departures = self.data_loader.get_metro_departures(
                line_id, direction, current_station, current_time
            )
            
            for departure in next_departures[:5]:  # Limit to first 5 options
                # Check if this departure reaches next station
                next_arrivals = self._get_arrivals_at_station(
                    line_id, direction, next_station, departure.departure_time
                )
                
                for arrival in next_arrivals:
                    if arrival.arrival_time > departure.departure_time:
                        # Calculate cargo pickup/dropoff
                        cargo_change = self._calculate_cargo_change(
                            next_station, dual_values
                        )
                        
                        new_load = label.load + cargo_change
                        if new_load > self.metro_capacity:
                            continue
                        
                        # Calculate cost (simple time-based cost)
                        travel_time = arrival.arrival_time - departure.departure_time
                        travel_cost = travel_time * 0.1  # Simple cost model
                        
                        # Create new label
                        new_label = MetroLabel(
                            station=next_station,
                            time=arrival.arrival_time,
                            load=new_load,
                            cost=label.cost + travel_cost,
                            schedule=label.schedule + [(next_station, arrival.arrival_time)],
                            cargo_up=label.cargo_up + max(0, cargo_change),
                            cargo_down=label.cargo_down + max(0, -cargo_change)
                        )
                        
                        # Check dominance
                        state_key = (next_station, arrival.arrival_time)
                        if self._is_dominated(new_label, labels[state_key]):
                            continue
                        
                        # Remove dominated labels
                        labels[state_key] = [l for l in labels[state_key] 
                                           if not new_label.dominates(l)]
                        
                        # Add new label
                        labels[state_key].append(new_label)
                        heapq.heappush(priority_queue, 
                                     (new_label.cost, new_label.time, new_label))
                        break  # Take first valid arrival
    
    def _get_arrivals_at_station(self, line_id: str, direction: str, 
                               station_id: int, after_time: int) -> List[Any]:
        """
        Get arrivals at a station after a given time.
        
        Args:
            line_id: Metro line ID
            direction: Direction
            station_id: Station ID
            after_time: Time threshold
            
        Returns:
            List of arrival entries
        """
        # Simple implementation: use departure times as arrival times
        # In a real system, this would account for travel time between stations
        departures = self.data_loader.get_metro_departures(
            line_id, direction, station_id, after_time
        )
        
        # Convert departures to arrivals (add small travel time)
        arrivals = []
        for dep in departures:
            arrival = type('Arrival', (), {
                'station_id': station_id,
                'arrival_time': dep.departure_time + 5,  # 5 minutes travel time
                'line_id': line_id,
                'direction': direction
            })()
            arrivals.append(arrival)
        
        return arrivals
    
    def _calculate_cargo_change(self, station_id: int, 
                               dual_values: Dict[str, float]) -> float:
        """
        Calculate cargo change at a station based on dual values.
        
        Args:
            station_id: Station ID
            dual_values: Dual values from RMP
            
        Returns:
            Cargo change (positive for pickup, negative for dropoff)
        """
        # Simple heuristic based on dual values
        psi_up = dual_values.get(f"psi_up_{station_id}", 0.0)
        psi_down = dual_values.get(f"psi_down_{station_id}", 0.0)
        
        # Calculate net cargo change based on dual value difference
        cargo_change = (psi_up - psi_down) * 10.0  # Scale factor
        
        # Limit cargo change
        return max(-100.0, min(100.0, cargo_change))
    
    def _is_dominated(self, new_label: MetroLabel, 
                     existing_labels: List[MetroLabel]) -> bool:
        """
        Check if the new label is dominated by any existing label.
        
        Args:
            new_label: New label to check
            existing_labels: List of existing labels at the same state
            
        Returns:
            True if new label is dominated, False otherwise
        """
        for existing_label in existing_labels:
            if existing_label.dominates(new_label):
                return True
        return False
    
    def _try_complete_schedule(self, label: MetroLabel, line_id: str, 
                              direction: str, dual_values: Dict[str, float]) -> Optional[Column]:
        """
        Try to create a complete metro schedule.
        
        Args:
            label: Current label
            line_id: Metro line ID
            direction: Direction
            dual_values: Dual values from RMP
            
        Returns:
            Column object if schedule is feasible, None otherwise
        """
        # Simple completion: schedule is valid as-is
        total_cost = label.cost
        
        # Create column details
        column_details = self._create_column_details(label, line_id, direction)
        
        # Calculate reduced cost
        reduced_cost = calculate_reduced_cost(total_cost, dual_values, column_details)
        
        # Create column
        column = Column(
            id=f"metro_{line_id}_{direction}_{len(label.schedule)}_{label.time}",
            type=ColumnType.METRO_SCHEDULE,
            vehicle_type=VehicleType.METRO,
            direct_cost=total_cost,
            reduced_cost=reduced_cost,
            details=column_details
        )
        
        return column
    
    def _create_column_details(self, label: MetroLabel, line_id: str, 
                              direction: str) -> Dict[str, Any]:
        """
        Create detailed information for the column.
        
        Args:
            label: Final label for the schedule
            line_id: Metro line ID
            direction: Direction
            
        Returns:
            Dictionary with column details
        """
        details = {
            'a_ip': {},  # Demand coverage (none for metro)
            'delta_up': {},  # Inventory up
            'delta_down': {},  # Inventory down
            'resource_usage': {},  # Resource usage per time slice
            'route': [entry[0] for entry in label.schedule],  # Station sequence
            'timing': label.schedule,  # Complete timing information
            'cargo': {
                'cargo_up': label.cargo_up,
                'cargo_down': label.cargo_down
            },
            'line_id': line_id,
            'direction': direction
        }
        
        # Set inventory coefficients for all stations in schedule
        for station_id, _ in label.schedule:
            if direction == 'up':
                details['delta_up'][station_id] = label.cargo_up / len(label.schedule)
            else:
                details['delta_down'][station_id] = label.cargo_down / len(label.schedule)
        
        # Calculate resource usage per time slice
        from utils import create_time_slices
        from config import RESOURCE_TIME_SLICE
        
        time_slices = create_time_slices(0, 1440, RESOURCE_TIME_SLICE)
        
        # Metro uses 1 resource (train) for the entire schedule duration
        if label.schedule:
            schedule_start = label.schedule[0][1]
            schedule_end = label.schedule[-1][1]
            
            for slice_start, slice_end in time_slices:
                if (schedule_start < slice_end and schedule_end > slice_start):
                    details['resource_usage'][f"{slice_start}_{slice_end}"] = 1.0
        
        return details
