import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, Battery, Navigation, AlertTriangle, Settings, Cpu, Radio } from 'lucide-react';

// Project Sanjay Mk2 - Drone Simulation Visualization
// Demonstrates: MuJoCo Physics, Flight Controller State Machine, Multi-drone support

const DroneSimulationViz = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [time, setTime] = useState(0);
  const [selectedDrone, setSelectedDrone] = useState(0);
  const [viewMode, setViewMode] = useState('3d'); // '3d', 'topdown', 'stateMachine'
  const canvasRef = useRef(null);

  // Simulated drone states
  const [drones, setDrones] = useState([
    {
      id: 0,
      type: 'ALPHA',
      position: { x: 0, y: 0, z: 0.5 },
      velocity: { x: 0, y: 0, z: 0 },
      mode: 'IDLE',
      battery: 100,
      targetPosition: null,
      color: '#3b82f6',
      path: [],
      motorSpeeds: [0, 0, 0, 0]
    },
    {
      id: 1,
      type: 'BETA',
      position: { x: 20, y: 0, z: 0.5 },
      velocity: { x: 0, y: 0, z: 0 },
      mode: 'IDLE',
      battery: 95,
      targetPosition: null,
      color: '#10b981',
      path: [],
      motorSpeeds: [0, 0, 0, 0]
    },
    {
      id: 2,
      type: 'BETA',
      position: { x: -20, y: 0, z: 0.5 },
      velocity: { x: 0, y: 0, z: 0 },
      mode: 'IDLE',
      battery: 88,
      targetPosition: null,
      color: '#f59e0b',
      path: [],
      motorSpeeds: [0, 0, 0, 0]
    }
  ]);

  // Mission waypoints for demo
  const missionWaypoints = [
    { x: 0, y: 0, z: 25 },
    { x: 50, y: 0, z: 25 },
    { x: 50, y: 50, z: 25 },
    { x: 0, y: 50, z: 25 },
    { x: 0, y: 0, z: 25 },
    { x: 0, y: 0, z: 0.5 }
  ];

  const [currentWaypoint, setCurrentWaypoint] = useState(0);

  // State machine transitions
  const stateTransitions = {
    'IDLE': ['ARMING', 'EMERGENCY'],
    'ARMING': ['ARMED', 'IDLE', 'EMERGENCY'],
    'ARMED': ['TAKING_OFF', 'IDLE', 'EMERGENCY'],
    'TAKING_OFF': ['HOVERING', 'EMERGENCY'],
    'HOVERING': ['NAVIGATING', 'LANDING', 'EMERGENCY'],
    'NAVIGATING': ['HOVERING', 'LANDING', 'EMERGENCY'],
    'LANDING': ['LANDED', 'EMERGENCY'],
    'LANDED': ['IDLE', 'ARMING', 'EMERGENCY'],
    'EMERGENCY': ['LANDED', 'IDLE']
  };

  // Physics simulation
  useEffect(() => {
    if (!isRunning) return;

    const interval = setInterval(() => {
      setTime(t => t + 0.02);

      setDrones(prevDrones => prevDrones.map((drone, idx) => {
        if (idx !== 0) return drone; // Only animate first drone for demo

        let newDrone = { ...drone };
        const target = missionWaypoints[currentWaypoint];

        // State machine logic
        if (drone.mode === 'IDLE' && isRunning) {
          newDrone.mode = 'ARMING';
        } else if (drone.mode === 'ARMING') {
          newDrone.mode = 'ARMED';
          newDrone.motorSpeeds = [1000, 1000, 1000, 1000];
        } else if (drone.mode === 'ARMED') {
          newDrone.mode = 'TAKING_OFF';
          newDrone.targetPosition = { ...target };
        } else if (drone.mode === 'TAKING_OFF' || drone.mode === 'NAVIGATING') {
          // Simple physics simulation
          const dx = target.x - drone.position.x;
          const dy = target.y - drone.position.y;
          const dz = target.z - drone.position.z;
          const distance = Math.sqrt(dx*dx + dy*dy + dz*dz);

          if (distance < 2) {
            if (currentWaypoint < missionWaypoints.length - 1) {
              setCurrentWaypoint(c => c + 1);
            } else {
              newDrone.mode = 'LANDING';
            }
            newDrone.mode = currentWaypoint === missionWaypoints.length - 1 ? 'LANDING' : 'HOVERING';
          } else {
            // Move towards target
            const speed = 5; // m/s
            const vx = (dx / distance) * speed * 0.02;
            const vy = (dy / distance) * speed * 0.02;
            const vz = (dz / distance) * speed * 0.02;

            newDrone.position = {
              x: drone.position.x + vx,
              y: drone.position.y + vy,
              z: drone.position.z + vz
            };
            newDrone.velocity = { x: vx/0.02, y: vy/0.02, z: vz/0.02 };

            // Update motor speeds based on thrust needed
            const baseThrust = 9.81 * 1.5 / 4; // Hover thrust per motor
            const thrustNeeded = baseThrust + (vz > 0 ? 2 : vz < 0 ? -1 : 0);
            newDrone.motorSpeeds = [thrustNeeded, thrustNeeded, thrustNeeded, thrustNeeded].map(
              t => Math.max(0, Math.min(5, t)) * 1000
            );

            // Add to path
            newDrone.path = [...drone.path.slice(-200), { ...newDrone.position }];

            newDrone.mode = 'NAVIGATING';
          }
        } else if (drone.mode === 'HOVERING') {
          newDrone.mode = 'NAVIGATING';
        } else if (drone.mode === 'LANDING') {
          if (drone.position.z > 0.5) {
            newDrone.position.z -= 0.05;
          } else {
            newDrone.mode = 'LANDED';
            newDrone.motorSpeeds = [0, 0, 0, 0];
          }
        } else if (drone.mode === 'LANDED') {
          newDrone.mode = 'IDLE';
        }

        // Battery drain
        newDrone.battery = Math.max(0, drone.battery - 0.001);

        return newDrone;
      }));
    }, 20);

    return () => clearInterval(interval);
  }, [isRunning, currentWaypoint]);

  const reset = () => {
    setIsRunning(false);
    setTime(0);
    setCurrentWaypoint(0);
    setDrones(drones.map(d => ({
      ...d,
      position: d.id === 0 ? { x: 0, y: 0, z: 0.5 } : d.id === 1 ? { x: 20, y: 0, z: 0.5 } : { x: -20, y: 0, z: 0.5 },
      velocity: { x: 0, y: 0, z: 0 },
      mode: 'IDLE',
      battery: 100 - d.id * 5,
      targetPosition: null,
      path: [],
      motorSpeeds: [0, 0, 0, 0]
    })));
  };

  const getModeColor = (mode) => {
    const colors = {
      'IDLE': '#6b7280',
      'ARMING': '#f59e0b',
      'ARMED': '#f59e0b',
      'TAKING_OFF': '#3b82f6',
      'HOVERING': '#10b981',
      'NAVIGATING': '#8b5cf6',
      'LANDING': '#f97316',
      'LANDED': '#6b7280',
      'EMERGENCY': '#ef4444'
    };
    return colors[mode] || '#6b7280';
  };

  const selectedDroneData = drones[selectedDrone];

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-3xl font-bold text-blue-400">Project Sanjay Mk2</h1>
            <p className="text-gray-400">MuJoCo Drone Simulation Visualization</p>
          </div>
          <div className="flex items-center gap-4">
            <button
              onClick={() => setIsRunning(!isRunning)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all ${
                isRunning ? 'bg-yellow-600 hover:bg-yellow-700' : 'bg-green-600 hover:bg-green-700'
              }`}
            >
              {isRunning ? <Pause size={20} /> : <Play size={20} />}
              {isRunning ? 'Pause' : 'Start'}
            </button>
            <button
              onClick={reset}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-gray-700 hover:bg-gray-600 transition-all"
            >
              <RotateCcw size={20} />
              Reset
            </button>
          </div>
        </div>

        {/* View Mode Tabs */}
        <div className="flex gap-2 mb-4">
          {['3d', 'topdown', 'stateMachine'].map(mode => (
            <button
              key={mode}
              onClick={() => setViewMode(mode)}
              className={`px-4 py-2 rounded-lg transition-all ${
                viewMode === mode ? 'bg-blue-600' : 'bg-gray-700 hover:bg-gray-600'
              }`}
            >
              {mode === '3d' ? '3D View' : mode === 'topdown' ? 'Top Down' : 'State Machine'}
            </button>
          ))}
        </div>

        <div className="grid grid-cols-3 gap-6">
          {/* Main Visualization */}
          <div className="col-span-2 bg-gray-800 rounded-xl p-4">
            {viewMode === 'stateMachine' ? (
              <StateMachineDiagram
                currentMode={selectedDroneData.mode}
                transitions={stateTransitions}
                getModeColor={getModeColor}
              />
            ) : (
              <DroneVisualization
                drones={drones}
                waypoints={missionWaypoints}
                currentWaypoint={currentWaypoint}
                viewMode={viewMode}
                selectedDrone={selectedDrone}
                setSelectedDrone={setSelectedDrone}
              />
            )}
          </div>

          {/* Control Panel */}
          <div className="space-y-4">
            {/* Drone Selector */}
            <div className="bg-gray-800 rounded-xl p-4">
              <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <Cpu size={20} className="text-blue-400" />
                Drone Fleet
              </h3>
              <div className="space-y-2">
                {drones.map((drone, idx) => (
                  <button
                    key={drone.id}
                    onClick={() => setSelectedDrone(idx)}
                    className={`w-full p-3 rounded-lg text-left transition-all ${
                      selectedDrone === idx ? 'bg-blue-600' : 'bg-gray-700 hover:bg-gray-600'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <div
                          className="w-3 h-3 rounded-full"
                          style={{ backgroundColor: drone.color }}
                        />
                        <span>Drone {drone.id}</span>
                        <span className="text-xs text-gray-400">({drone.type})</span>
                      </div>
                      <span
                        className="text-xs px-2 py-1 rounded"
                        style={{ backgroundColor: getModeColor(drone.mode) }}
                      >
                        {drone.mode}
                      </span>
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Selected Drone Telemetry */}
            <div className="bg-gray-800 rounded-xl p-4">
              <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <Navigation size={20} className="text-green-400" />
                Telemetry - Drone {selectedDrone}
              </h3>
              <div className="space-y-3">
                <TelemetryRow
                  label="Position"
                  value={`(${selectedDroneData.position.x.toFixed(1)}, ${selectedDroneData.position.y.toFixed(1)}, ${selectedDroneData.position.z.toFixed(1)})`}
                />
                <TelemetryRow
                  label="Altitude"
                  value={`${selectedDroneData.position.z.toFixed(2)} m`}
                />
                <TelemetryRow
                  label="Velocity"
                  value={`${Math.sqrt(
                    selectedDroneData.velocity.x**2 +
                    selectedDroneData.velocity.y**2 +
                    selectedDroneData.velocity.z**2
                  ).toFixed(2)} m/s`}
                />
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Battery</span>
                  <div className="flex items-center gap-2">
                    <Battery
                      size={16}
                      className={selectedDroneData.battery < 20 ? 'text-red-500' : 'text-green-400'}
                    />
                    <span>{selectedDroneData.battery.toFixed(1)}%</span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Flight Mode</span>
                  <span
                    className="px-2 py-1 rounded text-sm"
                    style={{ backgroundColor: getModeColor(selectedDroneData.mode) }}
                  >
                    {selectedDroneData.mode}
                  </span>
                </div>
              </div>
            </div>

            {/* Motor Speeds */}
            <div className="bg-gray-800 rounded-xl p-4">
              <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <Settings size={20} className="text-yellow-400" />
                Motor Speeds
              </h3>
              <div className="grid grid-cols-2 gap-3">
                {['M1 (CW)', 'M2 (CCW)', 'M3 (CCW)', 'M4 (CW)'].map((label, idx) => (
                  <div key={label} className="bg-gray-700 rounded-lg p-2">
                    <div className="text-xs text-gray-400 mb-1">{label}</div>
                    <div className="text-lg font-mono">
                      {selectedDroneData.motorSpeeds[idx].toFixed(0)} rpm
                    </div>
                    <div className="w-full bg-gray-600 rounded-full h-1.5 mt-1">
                      <div
                        className="bg-blue-500 h-1.5 rounded-full transition-all"
                        style={{ width: `${Math.min(100, selectedDroneData.motorSpeeds[idx] / 50)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Mission Progress */}
            <div className="bg-gray-800 rounded-xl p-4">
              <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <Radio size={20} className="text-purple-400" />
                Mission Progress
              </h3>
              <div className="space-y-2">
                {missionWaypoints.map((wp, idx) => (
                  <div
                    key={idx}
                    className={`flex items-center gap-2 p-2 rounded ${
                      idx < currentWaypoint ? 'bg-green-900/30 text-green-400' :
                      idx === currentWaypoint ? 'bg-blue-900/30 text-blue-400' :
                      'bg-gray-700 text-gray-400'
                    }`}
                  >
                    <div className={`w-2 h-2 rounded-full ${
                      idx < currentWaypoint ? 'bg-green-400' :
                      idx === currentWaypoint ? 'bg-blue-400' : 'bg-gray-500'
                    }`} />
                    <span className="text-sm">
                      WP{idx + 1}: ({wp.x}, {wp.y}, {wp.z}m)
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Architecture Overview */}
        <div className="mt-6 bg-gray-800 rounded-xl p-6">
          <h3 className="text-xl font-semibold mb-4">System Architecture</h3>
          <div className="grid grid-cols-4 gap-4">
            <ArchBox
              title="MuJoCo Physics"
              items={['Quadrotor dynamics', 'Thrust & drag', 'Quaternion orientation', 'Ground collision']}
              color="blue"
            />
            <ArchBox
              title="Flight Controller"
              items={['State machine', 'Position control', 'Velocity limiting', 'Safety monitor']}
              color="green"
            />
            <ArchBox
              title="MAVSDK Interface"
              items={['MAVLink protocol', 'Telemetry stream', 'Command handling', 'Offboard mode']}
              color="yellow"
            />
            <ArchBox
              title="Swarm Layer"
              items={['Gossip protocol', 'State sync', 'Task allocation', 'Formation control']}
              color="purple"
            />
          </div>
        </div>

        {/* Time Display */}
        <div className="mt-4 text-center text-gray-400">
          Simulation Time: {time.toFixed(2)}s
        </div>
      </div>
    </div>
  );
};

// Telemetry Row Component
const TelemetryRow = ({ label, value }) => (
  <div className="flex items-center justify-between">
    <span className="text-gray-400">{label}</span>
    <span className="font-mono">{value}</span>
  </div>
);

// Architecture Box Component
const ArchBox = ({ title, items, color }) => {
  const colors = {
    blue: 'border-blue-500 bg-blue-900/20',
    green: 'border-green-500 bg-green-900/20',
    yellow: 'border-yellow-500 bg-yellow-900/20',
    purple: 'border-purple-500 bg-purple-900/20'
  };

  return (
    <div className={`border-l-4 rounded-r-lg p-4 ${colors[color]}`}>
      <h4 className="font-semibold mb-2">{title}</h4>
      <ul className="text-sm text-gray-400 space-y-1">
        {items.map((item, idx) => (
          <li key={idx}>• {item}</li>
        ))}
      </ul>
    </div>
  );
};

// State Machine Diagram Component
const StateMachineDiagram = ({ currentMode, transitions, getModeColor }) => {
  const states = [
    { id: 'IDLE', x: 100, y: 200 },
    { id: 'ARMING', x: 250, y: 200 },
    { id: 'ARMED', x: 400, y: 200 },
    { id: 'TAKING_OFF', x: 550, y: 200 },
    { id: 'HOVERING', x: 550, y: 80 },
    { id: 'NAVIGATING', x: 700, y: 80 },
    { id: 'LANDING', x: 400, y: 320 },
    { id: 'LANDED', x: 250, y: 320 },
    { id: 'EMERGENCY', x: 700, y: 320 }
  ];

  const arrows = [
    { from: 'IDLE', to: 'ARMING' },
    { from: 'ARMING', to: 'ARMED' },
    { from: 'ARMED', to: 'TAKING_OFF' },
    { from: 'TAKING_OFF', to: 'HOVERING' },
    { from: 'HOVERING', to: 'NAVIGATING' },
    { from: 'NAVIGATING', to: 'HOVERING' },
    { from: 'HOVERING', to: 'LANDING' },
    { from: 'LANDING', to: 'LANDED' },
    { from: 'LANDED', to: 'IDLE' }
  ];

  return (
    <div className="h-96 relative">
      <svg className="w-full h-full">
        {/* Draw arrows */}
        {arrows.map((arrow, idx) => {
          const from = states.find(s => s.id === arrow.from);
          const to = states.find(s => s.id === arrow.to);
          return (
            <g key={idx}>
              <line
                x1={from.x + 40}
                y1={from.y}
                x2={to.x - 40}
                y2={to.y}
                stroke="#4b5563"
                strokeWidth="2"
                markerEnd="url(#arrowhead)"
              />
            </g>
          );
        })}

        {/* Arrow marker */}
        <defs>
          <marker
            id="arrowhead"
            markerWidth="10"
            markerHeight="7"
            refX="9"
            refY="3.5"
            orient="auto"
          >
            <polygon points="0 0, 10 3.5, 0 7" fill="#4b5563" />
          </marker>
        </defs>

        {/* Draw states */}
        {states.map(state => (
          <g key={state.id}>
            <rect
              x={state.x - 50}
              y={state.y - 20}
              width="100"
              height="40"
              rx="8"
              fill={state.id === currentMode ? getModeColor(state.id) : '#374151'}
              stroke={state.id === currentMode ? '#fff' : '#4b5563'}
              strokeWidth={state.id === currentMode ? 3 : 1}
            />
            <text
              x={state.x}
              y={state.y + 5}
              textAnchor="middle"
              fill="white"
              fontSize="12"
              fontWeight={state.id === currentMode ? 'bold' : 'normal'}
            >
              {state.id}
            </text>
          </g>
        ))}
      </svg>

      <div className="absolute bottom-4 left-4 text-sm text-gray-400">
        Current State: <span className="font-bold text-white">{currentMode}</span>
      </div>
    </div>
  );
};

// Drone Visualization Component
const DroneVisualization = ({ drones, waypoints, currentWaypoint, viewMode, selectedDrone, setSelectedDrone }) => {
  const scale = viewMode === 'topdown' ? 4 : 6;
  const centerX = 400;
  const centerY = 200;

  return (
    <div className="h-96 relative overflow-hidden">
      <svg className="w-full h-full">
        {/* Grid */}
        <defs>
          <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#374151" strokeWidth="0.5"/>
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#grid)" />

        {/* Waypoints */}
        {waypoints.map((wp, idx) => (
          <g key={idx}>
            <circle
              cx={centerX + wp.x * scale}
              cy={centerY - wp.y * scale}
              r={idx === currentWaypoint ? 12 : 8}
              fill={idx < currentWaypoint ? '#10b981' : idx === currentWaypoint ? '#3b82f6' : '#4b5563'}
              stroke={idx === currentWaypoint ? '#fff' : 'none'}
              strokeWidth="2"
            />
            <text
              x={centerX + wp.x * scale}
              y={centerY - wp.y * scale + 4}
              textAnchor="middle"
              fill="white"
              fontSize="10"
            >
              {idx + 1}
            </text>
          </g>
        ))}

        {/* Waypoint connections */}
        {waypoints.slice(0, -1).map((wp, idx) => {
          const next = waypoints[idx + 1];
          return (
            <line
              key={idx}
              x1={centerX + wp.x * scale}
              y1={centerY - wp.y * scale}
              x2={centerX + next.x * scale}
              y2={centerY - next.y * scale}
              stroke={idx < currentWaypoint ? '#10b981' : '#4b5563'}
              strokeWidth="2"
              strokeDasharray={idx >= currentWaypoint ? '4,4' : 'none'}
            />
          );
        })}

        {/* Drone paths */}
        {drones.map(drone => (
          <polyline
            key={`path-${drone.id}`}
            points={drone.path.map(p =>
              `${centerX + p.x * scale},${centerY - p.y * scale}`
            ).join(' ')}
            fill="none"
            stroke={drone.color}
            strokeWidth="1"
            opacity="0.5"
          />
        ))}

        {/* Drones */}
        {drones.map((drone, idx) => {
          const x = centerX + drone.position.x * scale;
          const y = centerY - drone.position.y * scale;
          const size = selectedDrone === idx ? 16 : 12;

          return (
            <g
              key={drone.id}
              onClick={() => setSelectedDrone(idx)}
              style={{ cursor: 'pointer' }}
            >
              {/* Drone body */}
              <polygon
                points={`${x},${y-size} ${x+size},${y+size/2} ${x-size},${y+size/2}`}
                fill={drone.color}
                stroke={selectedDrone === idx ? '#fff' : 'none'}
                strokeWidth="2"
              />

              {/* Altitude indicator (3D view only) */}
              {viewMode === '3d' && drone.position.z > 1 && (
                <>
                  <line
                    x1={x}
                    y1={y}
                    x2={x}
                    y2={y + drone.position.z * 2}
                    stroke={drone.color}
                    strokeWidth="1"
                    strokeDasharray="2,2"
                    opacity="0.5"
                  />
                  <circle
                    cx={x}
                    cy={y + drone.position.z * 2}
                    r="3"
                    fill={drone.color}
                    opacity="0.3"
                  />
                </>
              )}

              {/* Drone label */}
              <text
                x={x}
                y={y - size - 5}
                textAnchor="middle"
                fill="white"
                fontSize="10"
              >
                D{drone.id} [{drone.position.z.toFixed(0)}m]
              </text>
            </g>
          );
        })}

        {/* Scale indicator */}
        <g>
          <line x1="20" y1="370" x2="60" y2="370" stroke="white" strokeWidth="2" />
          <text x="40" y="385" textAnchor="middle" fill="white" fontSize="10">
            10m
          </text>
        </g>

        {/* Compass */}
        <g transform="translate(750, 50)">
          <circle cx="0" cy="0" r="25" fill="#1f2937" stroke="#4b5563" />
          <text x="0" y="-10" textAnchor="middle" fill="#ef4444" fontSize="12" fontWeight="bold">N</text>
          <text x="10" y="4" textAnchor="middle" fill="white" fontSize="10">E</text>
          <text x="0" y="18" textAnchor="middle" fill="white" fontSize="10">S</text>
          <text x="-10" y="4" textAnchor="middle" fill="white" fontSize="10">W</text>
        </g>
      </svg>

      {/* Altitude scale (3D view) */}
      {viewMode === '3d' && (
        <div className="absolute right-4 top-4 bg-gray-900/80 rounded p-2">
          <div className="text-xs text-gray-400 mb-1">Altitude</div>
          <div className="h-32 w-4 bg-gradient-to-t from-green-800 via-yellow-600 to-red-600 rounded relative">
            {drones.map(drone => (
              <div
                key={drone.id}
                className="absolute w-6 h-1 -left-1"
                style={{
                  backgroundColor: drone.color,
                  bottom: `${Math.min(100, drone.position.z / 65 * 100)}%`
                }}
              />
            ))}
          </div>
          <div className="flex justify-between text-xs text-gray-400 mt-1">
            <span>0</span>
            <span>65m</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default DroneSimulationViz;
