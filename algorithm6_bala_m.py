"""
Algorithm 6: Big-Arc Label-setting Algorithm for Metro (BALA-M)
Advanced metro scheduling with big-arc optimization and complex timetable handling.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
import heapq
from collections import defaultdict
import itertools

from data_structures import MetroBALabel, Column, ColumnType, VehicleType
from data_loader import DataLoader
from utils import log_message, calculate_reduced_cost
from config import METRO_MAX_CAPACITY

class BALAMSolver:
    """
    Big-Arc Label-setting Algorithm for Metro scheduling.
    Enhanced version with big-arc optimization for improved performance.
    """
    
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.metro_capacity = METRO_MAX_CAPACITY
        self.big_arcs_cache = {}  # Cache for big-arc computations
        
    def solve_pricing(self, dual_values: Dict[str, float]) -> List[Column]:
        """
        Solve metro pricing problem using big-arc label-setting algorithm.
        
        Args:
            dual_values: Dual values from RMP solution
            
        Returns:
            List of columns with negative reduced cost
        """
        log_message("Starting BALA-M pricing algorithm")
        
        negative_cost_columns = []
        
        # Build big-arcs for efficient computation
        self._build_big_arcs()
        
        # Find metro lines and directions
        metro_lines = self._get_metro_lines()
        
        for line_id, directions in metro_lines.items():
            for direction in directions:
                columns = self._solve_for_line_direction_big_arc(line_id, direction, dual_values)
                negative_cost_columns.extend(columns)
        
        log_message(f"BALA-M found {len(negative_cost_columns)} columns with negative reduced cost")
        return negative_cost_columns
    
    def _build_big_arcs(self):
        """
        Build big-arcs that represent multi-station movements efficiently.
        A big-arc represents a sequence of consecutive stations with predetermined timing.
        """
        log_message("Building big-arcs for metro optimization")
        
        self.big_arcs_cache = {}
        
        # Group timetable entries by line and direction
        timetable_groups = defaultdict(list)
        for entry in self.data_loader.metro_timetables:
            key = f"{entry.line_id}_{entry.direction}"
            timetable_groups[key].append(entry)
        
        # For each line-direction combination
        for line_direction, entries in timetable_groups.items():
            line_id, direction = line_direction.split('_', 1)
            
            # Sort by departure time
            entries.sort(key=lambda x: x.departure_time)
            
            # Group entries by departure time (same train)
            trains = defaultdict(list)
            for entry in entries:
                trains[entry.departure_time].append(entry)
            
            # Build big-arcs for each train
            big_arcs = []
            for departure_time, train_entries in trains.items():
                # Sort stations by some logical order (assuming station_id represents order)
                train_entries.sort(key=lambda x: x.station_id)
                
                # Create big-arcs for subsequences of stations
                for start_idx in range(len(train_entries)):
                    for end_idx in range(start_idx + 1, min(start_idx + 4, len(train_entries) + 1)):
                        # Create big-arc from start_idx to end_idx-1
                        arc_stations = train_entries[start_idx:end_idx]
                        
                        big_arc = {
                            'line_id': line_id,
                            'direction': direction,
                            'start_station': arc_stations[0].station_id,
                            'end_station': arc_stations[-1].station_id,
                            'start_time': arc_stations[0].departure_time,
                            'end_time': arc_stations[-1].arrival_time,
                            'stations': [s.station_id for s in arc_stations],
                            'timing': [(s.station_id, s.departure_time, s.arrival_time) for s in arc_stations],
                            'travel_time': arc_stations[-1].arrival_time - arc_stations[0].departure_time,
                            'cost': len(arc_stations) * 0.1  # Simple cost model
                        }
                        
                        big_arcs.append(big_arc)
            
            self.big_arcs_cache[line_direction] = big_arcs
        
        total_arcs = sum(len(arcs) for arcs in self.big_arcs_cache.values())
        log_message(f"Built {total_arcs} big-arcs for metro optimization")
    
    def _get_metro_lines(self) -> Dict[str, List[str]]:
        """Get available metro lines and directions."""
        lines = defaultdict(set)
        
        for entry in self.data_loader.metro_timetables:
            lines[entry.line_id].add(entry.direction)
        
        return {line_id: list(directions) for line_id, directions in lines.items()}
    
    def _solve_for_line_direction_big_arc(self, line_id: str, direction: str, 
                                         dual_values: Dict[str, float]) -> List[Column]:
        """
        Solve pricing for a specific metro line and direction using big-arcs.
        
        Args:
            line_id: Metro line ID
            direction: Direction ("up" or "down")
            dual_values: Dual values from RMP
            
        Returns:
            List of columns with negative reduced cost
        """
        line_direction = f"{line_id}_{direction}"
        big_arcs = self.big_arcs_cache.get(line_direction, [])
        
        if not big_arcs:
            return []
        
        # Get all stations for this line
        stations = self._get_line_stations_ordered(line_id, direction)
        
        if not stations:
            return []
        
        # Initialize labels for big-arc algorithm
        labels = defaultdict(list)  # (station, time) -> list of labels
        processed = set()
        priority_queue = []
        
        negative_columns = []
        
        # Create initial labels at first station using big-arcs that start there
        first_station = stations[0]
        starting_arcs = [arc for arc in big_arcs if arc['start_station'] == first_station]
        
        for arc in starting_arcs[:20]:  # Limit to first 20 arcs
            initial_label = MetroBALabel(
                station=arc['end_station'],
                time=arc['end_time'],
                load=0.0,
                cost=arc['cost'],
                big_arcs=[arc],
                cargo_up=0.0,
                cargo_down=0.0
            )
            
            # Calculate initial cargo based on dual values
            cargo_up, cargo_down = self._calculate_arc_cargo_values(arc, dual_values)
            initial_label.cargo_up = cargo_up
            initial_label.cargo_down = cargo_down
            
            state_key = (arc['end_station'], arc['end_time'])
            labels[state_key].append(initial_label)
            heapq.heappush(priority_queue, (arc['cost'], arc['end_time'], initial_label))
        
        # Main big-arc label extension loop
        while priority_queue:
            current_cost, current_time, current_label = heapq.heappop(priority_queue)
            
            state_key = (current_label.station, current_label.time)
            if state_key in processed:
                continue
            processed.add(state_key)
            
            # Check if we can create a complete schedule
            if len(current_label.big_arcs) >= 1:  # At least one arc
                complete_column = self._try_complete_big_arc_schedule(
                    current_label, line_id, direction, dual_values
                )
                if complete_column and complete_column.reduced_cost < -1e-6:
                    negative_columns.append(complete_column)
            
            # Limit schedule length
            if len(current_label.big_arcs) >= 5:
                continue
            
            # Extend using big-arcs
            self._extend_big_arc_label(current_label, big_arcs, dual_values, 
                                     labels, priority_queue)
        
        return negative_columns
    
    def _get_line_stations_ordered(self, line_id: str, direction: str) -> List[int]:
        """
        Get ordered list of stations for a metro line and direction.
        
        Args:
            line_id: Metro line ID
            direction: Direction ("up" or "down")
            
        Returns:
            List of station IDs in order
        """
        # Get all stations for this line and direction
        stations = set()
        for entry in self.data_loader.metro_timetables:
            if entry.line_id == line_id and entry.direction == direction:
                stations.add(entry.station_id)
        
        # Simple ordering by station ID
        # In a real system, this would respect the actual line topology
        ordered_stations = sorted(list(stations))
        
        if direction == "down":
            ordered_stations.reverse()
        
        return ordered_stations
    
    def _extend_big_arc_label(self, label: MetroBALabel, big_arcs: List[Dict[str, Any]],
                             dual_values: Dict[str, float],
                             labels: Dict[Tuple[int, int], List[MetroBALabel]], 
                             priority_queue: List):
        """
        Extend a label using big-arcs.
        
        Args:
            label: Current label to extend
            big_arcs: Available big-arcs
            dual_values: Dual values from RMP
            labels: Dictionary of labels by (station, time)
            priority_queue: Priority queue for label processing
        """
        current_station = label.station
        current_time = label.time
        
        # Find big-arcs that can extend from current position
        extendable_arcs = []
        for arc in big_arcs:
            # Check if arc starts from current station and after current time
            if (arc['start_station'] == current_station and 
                arc['start_time'] >= current_time):
                
                # Check if not already used (to avoid cycles)
                if not any(used_arc['start_station'] == arc['start_station'] and
                          used_arc['start_time'] == arc['start_time'] 
                          for used_arc in label.big_arcs):
                    extendable_arcs.append(arc)
        
        # Limit number of extensions for efficiency
        extendable_arcs = extendable_arcs[:10]
        
        for arc in extendable_arcs:
            # Calculate cargo change for this arc
            cargo_up_change, cargo_down_change = self._calculate_arc_cargo_values(arc, dual_values)
            
            # Check capacity constraints
            new_load = label.load + cargo_up_change + cargo_down_change
            if new_load > self.metro_capacity:
                continue
            
            # Calculate waiting time cost if there's a gap
            waiting_time = max(0, arc['start_time'] - current_time)
            waiting_cost = waiting_time * 0.01  # Small cost for waiting
            
            # Create new label
            new_label = MetroBALabel(
                station=arc['end_station'],
                time=arc['end_time'],
                load=new_load,
                cost=label.cost + arc['cost'] + waiting_cost,
                big_arcs=label.big_arcs + [arc],
                cargo_up=label.cargo_up + cargo_up_change,
                cargo_down=label.cargo_down + cargo_down_change
            )
            
            # Check dominance
            state_key = (arc['end_station'], arc['end_time'])
            if self._is_big_arc_dominated(new_label, labels[state_key]):
                continue
            
            # Remove dominated labels
            labels[state_key] = [l for l in labels[state_key] 
                               if not new_label.dominates(l)]
            
            # Add new label
            labels[state_key].append(new_label)
            heapq.heappush(priority_queue, (new_label.cost, new_label.time, new_label))
    
    def _calculate_arc_cargo_values(self, arc: Dict[str, Any], 
                                   dual_values: Dict[str, float]) -> Tuple[float, float]:
        """
        Calculate cargo values for a big-arc based on dual values.
        
        Args:
            arc: Big-arc information
            dual_values: Dual values from RMP
            
        Returns:
            Tuple of (cargo_up_change, cargo_down_change)
        """
        total_cargo_up = 0.0
        total_cargo_down = 0.0
        
        # Calculate cargo contribution for each station in the arc
        for station_id in arc['stations']:
            psi_up = dual_values.get(f"psi_up_{station_id}", 0.0)
            psi_down = dual_values.get(f"psi_down_{station_id}", 0.0)
            
            # Time-based demand factor
            hour = (arc['start_time'] + arc['end_time']) // 120  # Average time in hours
            if 7 <= hour <= 10:  # Morning rush
                if arc['direction'] == 'up':
                    demand_factor = 1.5
                else:
                    demand_factor = 0.5
            elif 17 <= hour <= 20:  # Evening rush
                if arc['direction'] == 'down':
                    demand_factor = 1.5
                else:
                    demand_factor = 0.5
            else:
                demand_factor = 1.0
            
            # Calculate cargo based on dual values and demand patterns
            station_cargo = (psi_up - psi_down) * demand_factor * 5.0  # Scale factor
            
            if station_cargo > 0:
                total_cargo_up += station_cargo
            else:
                total_cargo_down += abs(station_cargo)
        
        # Limit cargo per arc
        total_cargo_up = min(total_cargo_up, 100.0)
        total_cargo_down = min(total_cargo_down, 100.0)
        
        return total_cargo_up, total_cargo_down
    
    def _is_big_arc_dominated(self, new_label: MetroBALabel, 
                             existing_labels: List[MetroBALabel]) -> bool:
        """
        Check if the new big-arc label is dominated.
        
        Args:
            new_label: New label to check
            existing_labels: List of existing labels at the same state
            
        Returns:
            True if new label is dominated, False otherwise
        """
        for existing_label in existing_labels:
            if (existing_label.time <= new_label.time and
                existing_label.load <= new_label.load and
                existing_label.cost <= new_label.cost and
                existing_label.cargo_up >= new_label.cargo_up and
                existing_label.cargo_down >= new_label.cargo_down):
                return True
        
        return False
    
    def _try_complete_big_arc_schedule(self, label: MetroBALabel, line_id: str, 
                                      direction: str, dual_values: Dict[str, float]) -> Optional[Column]:
        """
        Try to create a complete metro schedule using big-arcs.
        
        Args:
            label: Current label
            line_id: Metro line ID
            direction: Direction
            dual_values: Dual values from RMP
            
        Returns:
            Column object if schedule is feasible, None otherwise
        """
        # Calculate total cost including any final costs
        total_cost = label.cost
        
        # Add completion cost (returning train to depot or maintenance)
        completion_cost = len(label.big_arcs) * 0.5  # Cost per arc used
        total_cost += completion_cost
        
        # Create column details
        column_details = self._create_big_arc_column_details(label, line_id, direction)
        
        # Calculate reduced cost
        reduced_cost = calculate_reduced_cost(total_cost, dual_values, column_details)
        
        # Create column
        column = Column(
            id=f"metro_ba_{line_id}_{direction}_{len(label.big_arcs)}_{label.time}",
            type=ColumnType.METRO_SCHEDULE,
            vehicle_type=VehicleType.METRO,
            direct_cost=total_cost,
            reduced_cost=reduced_cost,
            details=column_details
        )
        
        return column
    
    def _create_big_arc_column_details(self, label: MetroBALabel, line_id: str, 
                                      direction: str) -> Dict[str, Any]:
        """
        Create detailed information for the big-arc column.
        
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
            'route': [],  # Station sequence
            'timing': [],  # Complete timing information
            'cargo': {
                'cargo_up': label.cargo_up,
                'cargo_down': label.cargo_down
            },
            'line_id': line_id,
            'direction': direction,
            'big_arcs': label.big_arcs
        }
        
        # Collect all stations and timing from big-arcs
        all_stations = []
        all_timing = []
        
        for arc in label.big_arcs:
            for station_id, dep_time, arr_time in arc['timing']:
                if station_id not in [s[0] for s in all_timing]:  # Avoid duplicates
                    all_stations.append(station_id)
                    all_timing.append((station_id, dep_time, arr_time))
        
        details['route'] = all_stations
        details['timing'] = all_timing
        
        # Set inventory coefficients for all stations
        for station_id in all_stations:
            if direction == 'up':
                details['delta_up'][station_id] = label.cargo_up / max(1, len(all_stations))
            else:
                details['delta_down'][station_id] = label.cargo_down / max(1, len(all_stations))
        
        # Calculate resource usage per time slice
        from utils import create_time_slices
        from config import RESOURCE_TIME_SLICE
        
        time_slices = create_time_slices(0, 1440, RESOURCE_TIME_SLICE)
        
        # Metro uses 1 resource (train) for the entire schedule duration
        if label.big_arcs:
            schedule_start = min(arc['start_time'] for arc in label.big_arcs)
            schedule_end = max(arc['end_time'] for arc in label.big_arcs)
            
            for slice_start, slice_end in time_slices:
                if (schedule_start < slice_end and schedule_end > slice_start):
                    details['resource_usage'][f"{slice_start}_{slice_end}"] = 1.0
        
        return details
