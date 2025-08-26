/**
 * Express Backend Server for Branch-and-Price Solver
 * Provides REST API endpoints and WebSocket real-time updates
 */

const express = require('express');
const cors = require('cors');
const sqlite3 = require('sqlite3').verbose();
const WebSocket = require('ws');
const http = require('http');
const bodyParser = require('body-parser');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

// Middleware
app.use(cors());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(express.static('public'));

// File upload configuration
const upload = multer({
    dest: 'uploads/',
    fileFilter: (req, file, cb) => {
        if (file.mimetype === 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet') {
            cb(null, true);
        } else {
            cb(new Error('Only Excel files are allowed'));
        }
    }
});

// Database connection
const db = new sqlite3.Database('./logistics.db', (err) => {
    if (err) {
        console.error('Error opening database:', err.message);
    } else {
        console.log('Connected to SQLite database');
    }
});

// WebSocket connections tracking
const clients = new Set();

wss.on('connection', (ws) => {
    clients.add(ws);
    console.log('New WebSocket client connected');
    
    // Send initial data
    ws.send(JSON.stringify({
        type: 'connection',
        message: 'Connected to Branch-and-Price solver'
    }));
    
    ws.on('close', () => {
        clients.delete(ws);
        console.log('WebSocket client disconnected');
    });
});

// Broadcast to all connected clients
function broadcast(data) {
    const message = JSON.stringify(data);
    clients.forEach((client) => {
        if (client.readyState === WebSocket.OPEN) {
            client.send(message);
        }
    });
}

// API Routes

// Get system status
app.get('/api/status', (req, res) => {
    db.get(`
        SELECT COUNT(*) as total_runs FROM optimization_results
    `, (err, result) => {
        if (err) {
            return res.status(500).json({ error: err.message });
        }
        
        db.get(`
            SELECT 
                COUNT(*) as nodes_count,
                (SELECT COUNT(*) FROM vehicles) as vehicles_count,
                (SELECT COUNT(*) FROM demands) as demands_count,
                (SELECT COUNT(*) FROM real_time_metrics) as metrics_count
            FROM nodes
        `, (err2, counts) => {
            if (err2) {
                return res.status(500).json({ error: err2.message });
            }
            
            res.json({
                status: 'active',
                database_connected: true,
                total_optimization_runs: result.total_runs,
                ...counts,
                timestamp: new Date().toISOString()
            });
        });
    });
});

// Get all optimization results
app.get('/api/results', (req, res) => {
    const limit = parseInt(req.query.limit) || 10;
    const offset = parseInt(req.query.offset) || 0;
    
    db.all(`
        SELECT 
            id, run_id, algorithm_type, total_cost, total_runtime,
            num_iterations, final_gap, num_drone_routes, num_truck_routes,
            num_metro_schedules, created_at
        FROM optimization_results 
        ORDER BY created_at DESC 
        LIMIT ? OFFSET ?
    `, [limit, offset], (err, rows) => {
        if (err) {
            return res.status(500).json({ error: err.message });
        }
        res.json(rows);
    });
});

// Get specific optimization result with full solution data
app.get('/api/results/:runId', (req, res) => {
    const runId = req.params.runId;
    
    db.get(`
        SELECT * FROM optimization_results WHERE run_id = ?
    `, [runId], (err, row) => {
        if (err) {
            return res.status(500).json({ error: err.message });
        }
        
        if (!row) {
            return res.status(404).json({ error: 'Result not found' });
        }
        
        // Parse solution data if it exists
        if (row.solution_data) {
            try {
                row.solution_data = JSON.parse(row.solution_data);
            } catch (e) {
                console.error('Error parsing solution data:', e);
            }
        }
        
        res.json(row);
    });
});

// Get real-time metrics
app.get('/api/metrics', (req, res) => {
    const metricType = req.query.type || 'optimization_completed';
    const limit = parseInt(req.query.limit) || 50;
    
    db.all(`
        SELECT * FROM real_time_metrics 
        WHERE metric_type = ? 
        ORDER BY timestamp DESC 
        LIMIT ?
    `, [metricType, limit], (err, rows) => {
        if (err) {
            return res.status(500).json({ error: err.message });
        }
        
        // Parse metric_data JSON
        const metrics = rows.map(row => ({
            ...row,
            metric_data: row.metric_data ? JSON.parse(row.metric_data) : null
        }));
        
        res.json(metrics);
    });
});

// Get all nodes
app.get('/api/nodes', (req, res) => {
    db.all(`
        SELECT * FROM nodes ORDER BY id
    `, (err, rows) => {
        if (err) {
            return res.status(500).json({ error: err.message });
        }
        res.json(rows);
    });
});

// Add or update node
app.post('/api/nodes', (req, res) => {
    const { id, x_coord, y_coord, node_type, demand, time_window_start, time_window_end, service_time } = req.body;
    
    db.run(`
        INSERT OR REPLACE INTO nodes 
        (id, x_coord, y_coord, node_type, demand, time_window_start, time_window_end, service_time, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    `, [id, x_coord, y_coord, node_type, demand || 0, time_window_start || 0, time_window_end || 1440, service_time || 0], 
    function(err) {
        if (err) {
            return res.status(500).json({ error: err.message });
        }
        
        broadcast({
            type: 'node_updated',
            data: { id, x_coord, y_coord, node_type }
        });
        
        res.json({ id: this.lastID, message: 'Node saved successfully' });
    });
});

// Get all vehicles
app.get('/api/vehicles', (req, res) => {
    db.all(`
        SELECT * FROM vehicles ORDER BY id
    `, (err, rows) => {
        if (err) {
            return res.status(500).json({ error: err.message });
        }
        res.json(rows);
    });
});

// Add or update vehicle
app.post('/api/vehicles', (req, res) => {
    const { id, type, capacity, speed, cost_per_km, cost_per_hour, energy_capacity, energy_consumption_rate, status } = req.body;
    
    db.run(`
        INSERT OR REPLACE INTO vehicles 
        (id, type, capacity, speed, cost_per_km, cost_per_hour, energy_capacity, energy_consumption_rate, status, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    `, [id, type, capacity, speed, cost_per_km || 0, cost_per_hour || 0, energy_capacity, energy_consumption_rate, status || 'available'], 
    function(err) {
        if (err) {
            return res.status(500).json({ error: err.message });
        }
        
        broadcast({
            type: 'vehicle_updated',
            data: { id, type, capacity, status }
        });
        
        res.json({ message: 'Vehicle saved successfully' });
    });
});

// Get all demands
app.get('/api/demands', (req, res) => {
    const status = req.query.status;
    let query = `SELECT * FROM demands`;
    let params = [];
    
    if (status) {
        query += ` WHERE status = ?`;
        params.push(status);
    }
    
    query += ` ORDER BY created_at DESC`;
    
    db.all(query, params, (err, rows) => {
        if (err) {
            return res.status(500).json({ error: err.message });
        }
        res.json(rows);
    });
});

// Add or update demand
app.post('/api/demands', (req, res) => {
    const { id, origin, destination, quantity, time_window_start, time_window_end, priority, status } = req.body;
    
    db.run(`
        INSERT OR REPLACE INTO demands 
        (id, origin, destination, quantity, time_window_start, time_window_end, priority, status, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    `, [id, origin, destination, quantity, time_window_start, time_window_end, priority || 1, status || 'pending'], 
    function(err) {
        if (err) {
            return res.status(500).json({ error: err.message });
        }
        
        broadcast({
            type: 'demand_updated',
            data: { id, origin, destination, quantity, status }
        });
        
        res.json({ message: 'Demand saved successfully' });
    });
});

// Update demand status
app.put('/api/demands/:id/status', (req, res) => {
    const demandId = req.params.id;
    const { status } = req.body;
    
    db.run(`
        UPDATE demands SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?
    `, [status, demandId], function(err) {
        if (err) {
            return res.status(500).json({ error: err.message });
        }
        
        if (this.changes === 0) {
            return res.status(404).json({ error: 'Demand not found' });
        }
        
        broadcast({
            type: 'demand_status_updated',
            data: { id: demandId, status }
        });
        
        res.json({ message: 'Demand status updated successfully' });
    });
});

// Run optimization
app.post('/api/optimize', (req, res) => {
    const { algorithm_type = 'optimized' } = req.body;
    
    console.log('Starting optimization with algorithm:', algorithm_type);
    
    broadcast({
        type: 'optimization_started',
        data: { algorithm_type, timestamp: new Date().toISOString() }
    });
    
    // Run the Python solver
    const pythonProcess = spawn('python', ['main_bap.py'], {
        env: { ...process.env, ALGORITHM_TYPE: algorithm_type }
    });
    
    let outputBuffer = '';
    let errorBuffer = '';
    
    pythonProcess.stdout.on('data', (data) => {
        const output = data.toString();
        outputBuffer += output;
        
        // Parse log messages and broadcast real-time updates
        const lines = output.split('\n').filter(line => line.trim());
        lines.forEach(line => {
            if (line.includes('INFO:') || line.includes('ERROR:') || line.includes('WARNING:')) {
                broadcast({
                    type: 'optimization_log',
                    data: { message: line, timestamp: new Date().toISOString() }
                });
            }
        });
    });
    
    pythonProcess.stderr.on('data', (data) => {
        const error = data.toString();
        errorBuffer += error;
        
        broadcast({
            type: 'optimization_error',
            data: { error: error, timestamp: new Date().toISOString() }
        });
    });
    
    pythonProcess.on('close', (code) => {
        const success = code === 0;
        
        broadcast({
            type: 'optimization_completed',
            data: { 
                success, 
                exit_code: code,
                timestamp: new Date().toISOString()
            }
        });
        
        if (success) {
            // Get the latest optimization result
            db.get(`
                SELECT * FROM optimization_results 
                ORDER BY created_at DESC 
                LIMIT 1
            `, (err, result) => {
                if (result) {
                    broadcast({
                        type: 'optimization_result',
                        data: result
                    });
                }
            });
        }
        
        res.json({
            success,
            exit_code: code,
            output: outputBuffer,
            error: success ? null : errorBuffer
        });
    });
});

// File upload for Excel data files
app.post('/api/upload/:dataType', upload.single('file'), (req, res) => {
    const dataType = req.params.dataType; // nodes, demands, vehicles, etc.
    const file = req.file;
    
    if (!file) {
        return res.status(400).json({ error: 'No file uploaded' });
    }
    
    // Move file to data directory with correct name
    const targetPath = path.join('./data', `${dataType}.xlsx`);
    
    // Ensure data directory exists
    if (!fs.existsSync('./data')) {
        fs.mkdirSync('./data');
    }
    
    fs.rename(file.path, targetPath, (err) => {
        if (err) {
            return res.status(500).json({ error: 'Failed to save file' });
        }
        
        broadcast({
            type: 'file_uploaded',
            data: { dataType, filename: `${dataType}.xlsx` }
        });
        
        res.json({ 
            message: `${dataType} file uploaded successfully`,
            filename: `${dataType}.xlsx`
        });
    });
});

// Get database statistics
app.get('/api/stats', (req, res) => {
    const queries = [
        "SELECT COUNT(*) as nodes_count FROM nodes",
        "SELECT COUNT(*) as vehicles_count FROM vehicles", 
        "SELECT COUNT(*) as demands_count FROM demands",
        "SELECT COUNT(*) as results_count FROM optimization_results",
        "SELECT COUNT(*) as metrics_count FROM real_time_metrics"
    ];
    
    Promise.all(queries.map(query => 
        new Promise((resolve, reject) => {
            db.get(query, (err, result) => {
                if (err) reject(err);
                else resolve(result);
            });
        })
    )).then(results => {
        const stats = Object.assign({}, ...results);
        
        // Add file sizes
        try {
            const dbPath = './logistics.db';
            if (fs.existsSync(dbPath)) {
                const dbStats = fs.statSync(dbPath);
                stats.db_size_mb = (dbStats.size / (1024 * 1024)).toFixed(2);
            }
        } catch (e) {
            stats.db_size_mb = 0;
        }
        
        res.json(stats);
    }).catch(err => {
        res.status(500).json({ error: err.message });
    });
});

// Health check endpoint
app.get('/api/health', (req, res) => {
    res.json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        database: 'connected',
        websocket_clients: clients.size
    });
});

// Serve the main HTML file
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Error handling middleware
app.use((err, req, res, next) => {
    console.error('Error:', err);
    res.status(500).json({ error: 'Internal server error' });
});

// Graceful shutdown
process.on('SIGTERM', () => {
    console.log('Shutting down server...');
    db.close((err) => {
        if (err) {
            console.error('Error closing database:', err.message);
        } else {
            console.log('Database connection closed');
        }
    });
    server.close(() => {
        console.log('Server closed');
        process.exit(0);
    });
});

const PORT = process.env.PORT || 5000;
server.listen(PORT, '0.0.0.0', () => {
    console.log(`Branch-and-Price Server running on port ${PORT}`);
    console.log(`WebSocket server ready for real-time updates`);
});