<div align="center">
  <h1>🤖 IsaacMCP</h1>
  <p><strong>The Ultimate AI Copilot for Robotics Simulation</strong></p>
  <p>Seamlessly bridge LLMs (Claude, Cursor, Claude Code) with NVIDIA Isaac Sim using the Model Context Protocol (MCP).</p>
</div>

---

## 🚀 The Innovation: AI-Driven Simulation

**IsaacMCP** transforms robotics development from a manual, tedious workflow into a highly automated, self-healing, and dynamic process. By wrapping NVIDIA Isaac Sim in the Model Context Protocol (MCP), your favorite LLM can natively interact with, manipulate, and analyze your robotics environments in real-time. 

Say goodbye to hunting through logs or manually tuning physics parameters. IsaacMCP acts as your on-demand robotics engineering assistant, capable of diagnosing failures, writing fixes, orchestrating complex experiment campaigns, and learning from past mistakes.

### 🏆 Key Achievements & Capabilities

We have successfully engineered **54 specialized MCP tools** across **10 functional plugins**, delivering a true "Auto-Pilot" for Sim-to-Real workflows:

- 🧠 **Intelligent Diagnostics**: Cross-correlates telemetry, logs, and scene hierarchy to perform automated root-cause analysis of simulation failures.
- 💊 **Autonomous Fix Loop**: Self-healing simulations! When an error occurs, the AI synthesizes a fix, generates Kit API Python scripts, and injects them live to remediate the issue automatically.
- 🔬 **Experiment Engine**: Run massive batch simulations and parameter sweeps. Uses a blazing-fast async SQLite (`aiosqlite`) backend to record metrics such as success rates, durations, and parameter values.
- 🌪️ **Scenario Lab**: Procedurally generate randomized scenarios covering friction, gravity, obstacles, payloads, and lighting. Run automated **robustness testing campaigns** to validate your robot's resilience against worst-case environments.
- 📚 **Knowledge Memory**: A self-learning memory base that records known error patterns and tracks the statistical success rate of applied fixes—meaning the AI actually gets smarter over time.

---

## 🎯 Our Goal

Our mission is to drastically accelerate the Sim-to-Real pipeline by allowing developers and researchers to control, debug, and optimize robotic simulations using natural language and AI logic. We strive to provide an architecture where:
1. **Testing is Automated:** Let the AI spawn thousands of scenarios.
2. **Debugging is Instant:** AI reads the tracebacks and scene hierarchy for you.
3. **Safety is Guaranteed:** Read-only by default, with strict, explicit mutation gating.

---

## 🛠️ Tech Stack

IsaacMCP leverages a cutting-edge stack to guarantee low latency and high reliability:
- **Protocol:** Model Context Protocol (MCP)
- **Simulation Environment:** NVIDIA Isaac Sim (Omniverse Kit API, PhysX, Sensors)
- **Language:** Python 3.10+ (Fully Asynchronous)
- **Storage:** `aiosqlite` (Async SQLite) for deep experiment tracking; JSON for knowledge graphs.
- **Transports & Integrations:** WebSockets, SSH, HTTP/REST, ROS2 (via `rclpy`), Local Stdio, Remote HTTPS/SSE.
- **Testing:** `pytest` and `pytest-asyncio` with **100% core coverage**.

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.10 or higher.
- A running instance of NVIDIA Isaac Sim (or remote access to one).

### 1. Clone & Install
```bash
git clone https://github.com/your-org/isaac-mcp.git
cd isaac-mcp

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install the package with dev dependencies
pip install -U pip
pip install -e '.[dev]'
```

*(Optional)* If you are working with ROS2:
```bash
pip install -e '.[ros2]'
```

### 2. Configuration
IsaacMCP is configured via the `config/mcp_server.yaml` file. Here, you can define your Isaac Sim connection parameters, enable tools, and manage the knowledge base.

By default, **mutating actions** (like changing physics or applying fixes) are **DISABLED** for safety. To empower the AI to make changes, enable mutations via your environment:
```bash
export ISAAC_MCP_ENABLE_MUTATIONS=true
```

---

## 🕹️ Run Modes & Connector Setup

### Mode A: Local Execution (Claude Desktop / Cursor)
Run the server locally over `stdio`. This is the easiest way to get started.

**Claude Code CLI:**
```bash
claude mcp add --transport stdio --scope project isaac-sim -- .venv/bin/python -m isaac_mcp.server
```

**Cursor Setup (One-Click DeepLink):**
```bash
.venv/bin/python scripts/generate_cursor_deeplink.py \
  --name isaac-sim \
  --remote-url 'http://localhost:8000/mcp'
```

### Mode B: Remote Enterprise Deployment (Cloudflare & HTTPS)
Host IsaacMCP near your heavy GPU boxes while querying it securely from anywhere.

```bash
ISAAC_MCP_TRANSPORT=streamable-http \
ISAAC_MCP_HOST=0.0.0.1 \
ISAAC_MCP_PORT=8000 \
ISAAC_MCP_PUBLIC_BASE_URL='https://mcp.your-domain.com' \
.venv/bin/python -m isaac_mcp.server
```
*Note: Remote rollout supports robust OAuth bearer-token verification to keep your simulation endpoints secure.*

---

## 🧰 The Tool Arsenal

Once connected, your AI assistant will have access to 54 semantic tools. Here represents a fraction of what you can ask the AI to do:

- **"Analyze why the robot keeps falling over."** -> Triggers `analyze_simulation` + `get_diagnosis_history`.
- **"Run a parameter sweep on floor friction from 0.1 to 1.0."** -> Triggers `run_parameter_sweep` utilizing the Experiment Engine.
- **"Generate 50 randomized lighting and gravity scenarios and test robustness."** -> Triggers the Scenario Lab's `generate_scenario` and `run_robustness_test`.
- **"Has this physics error happened before?"** -> Triggers `query_knowledge_base` to retrieve statistical success rates of past fixes.
- **"Fix it."** -> Triggers `generate_fix` and `apply_fix_script` utilizing the Autonomous Fix Loop.

### Included MCP Resources:
Gain instant read access to active contexts:
- `isaac://logs/latest`
- `isaac://logs/errors`
- `isaac://sim/state`
- `isaac://sim/config`
- `isaac://scene/hierarchy`
- `isaac://ros2/status`

---

## 🔒 Security & Safety Defaults

Safety is a first-class citizen in IsaacMCP.
- **Read-Only by Default:** The server boots into read-only mode.
- **Gated Actions:** Destructive operations (like changing the USD stage or applying scripts) check a strict mutation gate (`ISAAC_MCP_ENABLE_MUTATIONS`). 
- **Tool Annotations:** All tools are heavily annotated with `readOnlyHint`, `destructiveHint`, and `idempotentHint` so the LLM intrinsically understands the weight of its actions.

---

## 🤔 Troubleshooting & Support

- **Tests Failing?** Run `.venv/bin/python -m pytest -v` to ensure your local environment is correctly configured.
- **Mutation Blocked?** Ensure the `ISAAC_MCP_ENABLE_MUTATIONS=true` environment variable is exported to the process running the server.
- **Resources Not Loading?** Check `config/mcp_server.yaml` to verify the IP and port mapping for your Omniverse/Kit APIs.

---

<div align="center">
  <p>Built with ❤️ for Robotics Engineers.</p>
</div>
