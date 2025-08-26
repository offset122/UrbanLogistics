# Branch-and-Price for Urban Logistics

## Overview

This project implements a Branch-and-Price solver for last-mile delivery optimization using multiple vehicle types: drones, trucks, and metro transportation. The system solves a Resource Constrained Shortest Path Problem (RCSPP) through column generation, where the Restricted Master Problem (RMP) selects optimal routes and schedules while pricing subproblems generate new columns using various label-setting algorithms.

The solver addresses urban logistics challenges by coordinating different transportation modes to satisfy delivery demands while respecting vehicle capacities, energy constraints, and time windows. It includes both basic and optimized algorithms for each vehicle type, allowing for performance comparison and scalability testing.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Core Components

The system follows a modular architecture with clear separation between data handling, optimization algorithms, and output generation:

**Data Layer**: The `DataLoader` class handles all input processing, including Excel file parsing, header normalization (Chinese to English), time conversion to minutes, and construction of distance/time matrices. It builds unified data structures for nodes, vehicles, demands, and metro timetables.

**Algorithm Layer**: Six pricing algorithms are implemented in separate modules:
- Basic algorithms: ULA-T (trucks), ULA-D (drones), SELA-M (metro)  
- Optimized algorithms: BLA-T (bidirectional trucks), BLA-D (bidirectional drones), BALA-M (big-arc metro)

**Optimization Engine**: The `BranchAndPriceSolver` coordinates the main optimization loop, while `RMPSolver` handles the Gurobi-based master problem. The system uses column generation with dual values to guide the pricing subproblems.

**Output Layer**: `ExcelWriter` generates comprehensive Excel reports with solution statistics, route details, and performance metrics.

### Design Patterns

**Strategy Pattern**: Algorithm selection between basic and optimized solvers based on configuration flags, allowing runtime switching between performance levels.

**Factory Pattern**: Data structure creation through the loader, which builds appropriate objects based on input file content and vehicle types.

**Observer Pattern**: Solution tracking through statistics collection, enabling real-time monitoring of optimization progress.

### Data Processing Architecture

The system uses a two-stage time resolution approach: 1-minute internal resolution for precise calculations and 15-minute resource slices for constraint management. All input times are normalized to minutes from 00:00 for consistent processing.

Header normalization ensures compatibility with Chinese and English Excel files through a configurable mapping system. Distance and time matrices are pre-computed for efficient route evaluation.

### Constraint Handling

Resource constraints are managed through specialized label structures:
- `DroneLabel`: tracks time, load, energy consumption, and battery state
- `TruckLabel`: manages time, load, and capacity constraints  
- `MetroLabel`: handles timetable adherence and passenger capacity

The label-setting algorithms use dominance rules to prune suboptimal solutions while ensuring feasibility across all resource dimensions.

## External Dependencies

**Optimization Engine**: Gurobi Optimizer for solving the Restricted Master Problem, providing high-performance linear programming capabilities with dual value extraction for column generation.

**Data Processing**: 
- pandas for Excel file reading and data manipulation
- numpy for matrix operations and numerical computations
- openpyxl (via pandas) for Excel output generation

**Core Libraries**:
- heapq for priority queue implementation in label-setting algorithms
- collections.defaultdict for efficient data structure management
- typing for comprehensive type annotations
- dataclasses for structured data representation
- enum for vehicle and column type definitions

**Input Data Sources**: Five Excel files containing nodes, demands, vehicles, metro timetables, and system parameters. The system expects structured data with specific column headers that are normalized during loading.

**Output Generation**: Excel files for solution summaries, detailed routes, performance metrics, and optimization statistics. All outputs follow a standardized format for easy analysis and visualization.