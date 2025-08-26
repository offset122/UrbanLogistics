/**
 * Frontend JavaScript Application for Branch-and-Price Solver UI
 */

class BranchAndPriceApp {
    constructor() {
        this.ws = null;
        this.chart = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        
        this.init();
    }
    
    init() {
        this.setupWebSocket();
        this.setupEventListeners();
        this.setupTabs();
        this.initializeChart();
        this.loadInitialData();
    }
    
    setupWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}`;
        
        try {
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                console.log('WebSocket connected');
                this.updateConnectionStatus(true);
                this.reconnectAttempts = 0;
            };
            
            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleWebSocketMessage(data);
                } catch (e) {
                    console.error('Error parsing WebSocket message:', e);
                }
            };
            
            this.ws.onclose = () => {
                console.log('WebSocket disconnected');
                this.updateConnectionStatus(false);
                this.attemptReconnect();
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus(false);
            };
        } catch (e) {
            console.error('Failed to create WebSocket connection:', e);
            this.updateConnectionStatus(false);
        }
    }
    
    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`Attempting to reconnect... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
            setTimeout(() => this.setupWebSocket(), 2000 * this.reconnectAttempts);
        }
    }
    
    updateConnectionStatus(connected) {
        const statusIndicator = document.getElementById('connectionStatus');
        const statusText = document.getElementById('connectionText');
        
        if (connected) {
            statusIndicator.className = 'status-indicator status-active';
            statusText.textContent = 'Connected';
        } else {
            statusIndicator.className = 'status-indicator status-inactive';
            statusText.textContent = 'Disconnected';
        }
    }
    
    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'optimization_started':
                this.onOptimizationStarted(data.data);
                break;
            case 'optimization_log':
                this.addLogEntry(data.data);
                break;
            case 'optimization_completed':
                this.onOptimizationCompleted(data.data);
                break;
            case 'optimization_result':
                this.onOptimizationResult(data.data);
                break;
            case 'node_updated':
            case 'vehicle_updated':
            case 'demand_updated':
                this.loadDataManagement();
                break;
            default:
                console.log('Unknown message type:', data.type, data);
        }
    }
    
    setupEventListeners() {
        // Optimize button
        document.getElementById('optimizeBtn').addEventListener('click', () => {
            this.startOptimization();
        });
        
        // Data management button
        document.getElementById('dataManagementBtn').addEventListener('click', () => {
            this.showTab('data');
        });
        
        // Upload button
        document.getElementById('uploadBtn').addEventListener('click', () => {
            this.showModal('uploadModal');
        });
        
        // Results button
        document.getElementById('resultsBtn').addEventListener('click', () => {
            this.showTab('results');
            this.loadOptimizationResults();
        });
        
        // Refresh results button
        document.getElementById('refreshResults').addEventListener('click', () => {
            this.loadOptimizationResults();
        });
        
        // Node form
        document.getElementById('nodeForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.saveNode(e.target);
        });
        
        // Upload form
        document.getElementById('uploadForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.uploadFile(e.target);
        });
    }
    
    setupTabs() {
        const tabButtons = document.querySelectorAll('.tab-button');
        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                const tabName = button.dataset.tab;
                this.showTab(tabName);
            });
        });
    }
    
    showTab(tabName) {
        // Hide all tab contents
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.add('hidden');
        });
        
        // Remove active class from all tab buttons
        document.querySelectorAll('.tab-button').forEach(button => {
            button.classList.remove('active', 'border-blue-500', 'text-blue-600');
            button.classList.add('border-transparent', 'text-gray-500');
        });
        
        // Show selected tab content
        document.getElementById(tabName).classList.remove('hidden');
        
        // Activate selected tab button
        const activeButton = document.querySelector(`[data-tab="${tabName}"]`);
        activeButton.classList.add('active', 'border-blue-500', 'text-blue-600');
        activeButton.classList.remove('border-transparent', 'text-gray-500');
        
        // Load tab-specific data
        if (tabName === 'data') {
            this.loadDataManagement();
        } else if (tabName === 'results') {
            this.loadOptimizationResults();
        }
    }
    
    async loadInitialData() {
        try {
            const response = await fetch('/api/stats');
            const stats = await response.json();
            
            document.getElementById('nodesCount').textContent = stats.nodes_count || 0;
            document.getElementById('vehiclesCount').textContent = stats.vehicles_count || 0;
            document.getElementById('demandsCount').textContent = stats.demands_count || 0;
            document.getElementById('optimizationRuns').textContent = stats.results_count || 0;
        } catch (error) {
            console.error('Error loading initial data:', error);
        }
    }
    
    initializeChart() {
        const ctx = document.getElementById('performanceChart').getContext('2d');
        this.chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Total Cost',
                    data: [],
                    borderColor: 'rgb(59, 130, 246)',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    tension: 0.1
                }, {
                    label: 'Runtime (s)',
                    data: [],
                    borderColor: 'rgb(16, 185, 129)',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    yAxisID: 'y1',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Optimization Performance Over Time'
                    }
                },
                scales: {
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Total Cost'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Runtime (seconds)'
                        },
                        grid: {
                            drawOnChartArea: false,
                        },
                    }
                }
            }
        });
    }
    
    async startOptimization() {
        const button = document.getElementById('optimizeBtn');
        button.disabled = true;
        button.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Optimizing...';
        
        this.clearLogs();
        this.addLogEntry({
            message: 'Starting optimization process...',
            timestamp: new Date().toISOString()
        });
        
        try {
            const response = await fetch('/api/optimize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    algorithm_type: 'optimized'
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.addLogEntry({
                    message: 'Optimization completed successfully!',
                    timestamp: new Date().toISOString()
                });
            } else {
                this.addLogEntry({
                    message: `Optimization failed: ${result.error || 'Unknown error'}`,
                    timestamp: new Date().toISOString()
                });
            }
        } catch (error) {
            console.error('Error starting optimization:', error);
            this.addLogEntry({
                message: `Error starting optimization: ${error.message}`,
                timestamp: new Date().toISOString()
            });
        } finally {
            button.disabled = false;
            button.innerHTML = '<i class="fas fa-play mr-2"></i>Start Optimization';
        }
    }
    
    onOptimizationStarted(data) {
        this.addLogEntry({
            message: `Optimization started with ${data.algorithm_type} algorithms`,
            timestamp: data.timestamp
        });
    }
    
    onOptimizationCompleted(data) {
        const message = data.success ? 
            'Optimization completed successfully!' : 
            `Optimization failed with exit code ${data.exit_code}`;
            
        this.addLogEntry({
            message: message,
            timestamp: data.timestamp
        });
        
        if (data.success) {
            this.loadInitialData();
            this.loadOptimizationResults();
        }
    }
    
    onOptimizationResult(data) {
        // Update chart with new result
        if (this.chart) {
            const timestamp = new Date().toLocaleTimeString();
            this.chart.data.labels.push(timestamp);
            this.chart.data.datasets[0].data.push(data.total_cost);
            this.chart.data.datasets[1].data.push(data.total_runtime);
            
            // Keep only last 10 data points
            if (this.chart.data.labels.length > 10) {
                this.chart.data.labels.shift();
                this.chart.data.datasets[0].data.shift();
                this.chart.data.datasets[1].data.shift();
            }
            
            this.chart.update();
        }
    }
    
    addLogEntry(data) {
        const container = document.getElementById('liveLogContainer');
        const entry = document.createElement('div');
        entry.className = 'log-entry log-info';
        
        // Determine log level from message
        if (data.message.toLowerCase().includes('error')) {
            entry.className = 'log-entry log-error';
        } else if (data.message.toLowerCase().includes('warning')) {
            entry.className = 'log-entry log-warning';
        }
        
        const timestamp = new Date(data.timestamp).toLocaleTimeString();
        entry.innerHTML = `
            <div class="flex justify-between items-start">
                <span class="text-sm">${data.message}</span>
                <span class="text-xs text-gray-500 ml-2">${timestamp}</span>
            </div>
        `;
        
        // Remove placeholder text if it exists
        if (container.querySelector('p')) {
            container.innerHTML = '';
        }
        
        container.insertBefore(entry, container.firstChild);
        
        // Keep only last 50 entries
        while (container.children.length > 50) {
            container.removeChild(container.lastChild);
        }
    }
    
    clearLogs() {
        document.getElementById('liveLogContainer').innerHTML = '<p class="text-gray-500 text-sm">Waiting for optimization logs...</p>';
    }
    
    async loadOptimizationResults() {
        try {
            const response = await fetch('/api/results?limit=20');
            const results = await response.json();
            
            const tbody = document.getElementById('resultsTableBody');
            tbody.innerHTML = '';
            
            results.forEach(result => {
                const row = document.createElement('tr');
                row.className = 'hover:bg-gray-50';
                
                const totalRoutes = (result.num_truck_routes || 0) + (result.num_drone_routes || 0) + (result.num_metro_schedules || 0);
                
                row.innerHTML = `
                    <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">${result.run_id}</td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">${result.algorithm_type}</td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">${result.total_cost.toFixed(2)}</td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">${result.total_runtime.toFixed(2)}s</td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">${(result.final_gap * 100).toFixed(2)}%</td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                            ${totalRoutes} routes
                        </span>
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm font-medium">
                        <button onclick="viewResultDetails('${result.run_id}')" class="text-indigo-600 hover:text-indigo-900">
                            <i class="fas fa-eye mr-1"></i>View Details
                        </button>
                    </td>
                `;
                
                tbody.appendChild(row);
            });
        } catch (error) {
            console.error('Error loading optimization results:', error);
        }
    }
    
    async loadDataManagement() {
        try {
            // Load nodes
            const nodesResponse = await fetch('/api/nodes');
            const nodes = await nodesResponse.json();
            this.renderNodesList(nodes);
            
            // Load vehicles
            const vehiclesResponse = await fetch('/api/vehicles');
            const vehicles = await vehiclesResponse.json();
            this.renderVehiclesList(vehicles);
            
            // Load demands
            const demandsResponse = await fetch('/api/demands');
            const demands = await demandsResponse.json();
            this.renderDemandsList(demands);
            
        } catch (error) {
            console.error('Error loading data management:', error);
        }
    }
    
    renderNodesList(nodes) {
        const container = document.getElementById('nodesList');
        container.innerHTML = '';
        
        nodes.forEach(node => {
            const item = document.createElement('div');
            item.className = 'flex justify-between items-center bg-white p-2 rounded border';
            item.innerHTML = `
                <div>
                    <div class="text-sm font-medium">Node ${node.id}</div>
                    <div class="text-xs text-gray-500">${node.node_type} (${node.x_coord}, ${node.y_coord})</div>
                </div>
                <button onclick="editNode(${node.id})" class="text-blue-600 hover:text-blue-800">
                    <i class="fas fa-edit"></i>
                </button>
            `;
            container.appendChild(item);
        });
    }
    
    renderVehiclesList(vehicles) {
        const container = document.getElementById('vehiclesList');
        container.innerHTML = '';
        
        vehicles.forEach(vehicle => {
            const statusColor = vehicle.status === 'available' ? 'text-green-600' : 'text-red-600';
            const item = document.createElement('div');
            item.className = 'flex justify-between items-center bg-white p-2 rounded border';
            item.innerHTML = `
                <div>
                    <div class="text-sm font-medium">${vehicle.id}</div>
                    <div class="text-xs text-gray-500">${vehicle.type} - Cap: ${vehicle.capacity}</div>
                    <div class="text-xs ${statusColor}">${vehicle.status}</div>
                </div>
                <button onclick="editVehicle('${vehicle.id}')" class="text-blue-600 hover:text-blue-800">
                    <i class="fas fa-edit"></i>
                </button>
            `;
            container.appendChild(item);
        });
    }
    
    renderDemandsList(demands) {
        const container = document.getElementById('demandsList');
        container.innerHTML = '';
        
        demands.slice(0, 10).forEach(demand => {
            const statusColor = demand.status === 'pending' ? 'text-yellow-600' : 
                               demand.status === 'completed' ? 'text-green-600' : 'text-red-600';
            const item = document.createElement('div');
            item.className = 'flex justify-between items-center bg-white p-2 rounded border';
            item.innerHTML = `
                <div>
                    <div class="text-sm font-medium">${demand.id}</div>
                    <div class="text-xs text-gray-500">${demand.origin} → ${demand.destination}</div>
                    <div class="text-xs ${statusColor}">${demand.status} (${demand.quantity})</div>
                </div>
                <button onclick="editDemand('${demand.id}')" class="text-blue-600 hover:text-blue-800">
                    <i class="fas fa-edit"></i>
                </button>
            `;
            container.appendChild(item);
        });
    }
    
    async saveNode(form) {
        const formData = new FormData(form);
        const nodeData = Object.fromEntries(formData);
        
        try {
            const response = await fetch('/api/nodes', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(nodeData)
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.closeModal('nodeModal');
                this.loadDataManagement();
                this.loadInitialData();
                this.showNotification('Node saved successfully', 'success');
            } else {
                this.showNotification('Error saving node: ' + result.error, 'error');
            }
        } catch (error) {
            console.error('Error saving node:', error);
            this.showNotification('Error saving node', 'error');
        }
    }
    
    async uploadFile(form) {
        const formData = new FormData(form);
        const dataType = formData.get('dataType');
        
        try {
            const response = await fetch(`/api/upload/${dataType}`, {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.closeModal('uploadModal');
                this.showNotification(`${dataType} file uploaded successfully`, 'success');
                form.reset();
            } else {
                this.showNotification('Upload failed: ' + result.error, 'error');
            }
        } catch (error) {
            console.error('Error uploading file:', error);
            this.showNotification('Upload failed', 'error');
        }
    }
    
    showModal(modalId) {
        document.getElementById(modalId).classList.add('active');
    }
    
    closeModal(modalId) {
        document.getElementById(modalId).classList.remove('active');
    }
    
    showNotification(message, type = 'info') {
        // Simple notification system - could be enhanced with a proper notification library
        const notification = document.createElement('div');
        notification.className = `fixed top-4 right-4 px-4 py-2 rounded shadow-lg z-50 ${
            type === 'success' ? 'bg-green-500 text-white' :
            type === 'error' ? 'bg-red-500 text-white' :
            'bg-blue-500 text-white'
        }`;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 3000);
    }
}

// Global functions for modal operations
function showAddNodeModal() {
    document.getElementById('nodeForm').reset();
    app.showModal('nodeModal');
}

function showAddVehicleModal() {
    // Implementation for vehicle modal
    app.showNotification('Vehicle management coming soon', 'info');
}

function showAddDemandModal() {
    // Implementation for demand modal
    app.showNotification('Demand management coming soon', 'info');
}

function closeModal(modalId) {
    app.closeModal(modalId);
}

function viewResultDetails(runId) {
    // Implementation for viewing result details
    app.showNotification(`Viewing details for run ${runId}`, 'info');
}

function editNode(nodeId) {
    app.showNotification(`Edit node ${nodeId} - feature coming soon`, 'info');
}

function editVehicle(vehicleId) {
    app.showNotification(`Edit vehicle ${vehicleId} - feature coming soon`, 'info');
}

function editDemand(demandId) {
    app.showNotification(`Edit demand ${demandId} - feature coming soon`, 'info');
}

// Initialize the application
const app = new BranchAndPriceApp();