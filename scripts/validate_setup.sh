#!/bin/bash
# =============================================================================
# Validation Script for WSL2 + Isaac Sim ROS 2 Setup
# =============================================================================
#
# Run this script inside WSL2 to verify the setup is correct.
#
# Usage:
#   chmod +x scripts/validate_setup.sh
#   ./scripts/validate_setup.sh
#
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "WSL2 + Isaac Sim ROS 2 Validation Script"
echo "=========================================="

# Get Windows username
WINDOWS_USER=$(cmd.exe /c "echo %USERNAME%" 2>/dev/null | tr -d '\r' || echo "unknown")

# Check WSL2 networking mode
echo -e "\n${YELLOW}[1/8]${NC} Checking WSL2 networking mode..."
if [ -f "/mnt/c/Users/$WINDOWS_USER/.wslconfig" ]; then
    if grep -q "networkingMode=mirrored" "/mnt/c/Users/$WINDOWS_USER/.wslconfig" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} Mirrored networking is enabled"
    else
        echo -e "${RED}✗${NC} WARNING: Mirrored networking not configured in .wslconfig"
        echo "  Create C:\\Users\\$WINDOWS_USER\\.wslconfig with:"
        echo "  [wsl2]"
        echo "  networkingMode=mirrored"
        echo "  hostAddressLoopback=true"
    fi
else
    echo -e "${RED}✗${NC} .wslconfig not found"
fi

# Check Fast DDS profile exists
echo -e "\n${YELLOW}[2/8]${NC} Checking Fast DDS profile..."
if [ -f "./network/fastdds_profiles.xml" ]; then
    echo -e "${GREEN}✓${NC} Fast DDS profile found at ./network/fastdds_profiles.xml"
elif [ -f "/opt/config/fastdds_profiles.xml" ]; then
    echo -e "${GREEN}✓${NC} Fast DDS profile found at /opt/config/fastdds_profiles.xml"
else
    echo -e "${RED}✗${NC} Fast DDS profile not found"
    echo "  Expected at ./network/fastdds_profiles.xml or /opt/config/fastdds_profiles.xml"
fi

# Check environment variables
echo -e "\n${YELLOW}[3/8]${NC} Checking environment variables..."
echo "  ROS_DOMAIN_ID: ${ROS_DOMAIN_ID:-NOT SET}"
echo "  RMW_IMPLEMENTATION: ${RMW_IMPLEMENTATION:-NOT SET}"
echo "  FASTRTPS_DEFAULT_PROFILES_FILE: ${FASTRTPS_DEFAULT_PROFILES_FILE:-NOT SET}"
echo "  ROS_LOCALHOST_ONLY: ${ROS_LOCALHOST_ONLY:-NOT SET}"

if [ "$ROS_DOMAIN_ID" = "10" ]; then
    echo -e "${GREEN}✓${NC} ROS_DOMAIN_ID is set to 10"
else
    echo -e "${RED}✗${NC} ROS_DOMAIN_ID should be 10"
fi

if [ "$RMW_IMPLEMENTATION" = "rmw_fastrtps_cpp" ]; then
    echo -e "${GREEN}✓${NC} RMW_IMPLEMENTATION is set to rmw_fastrtps_cpp"
else
    echo -e "${RED}✗${NC} RMW_IMPLEMENTATION should be rmw_fastrtps_cpp"
fi

# Check network interfaces
echo -e "\n${YELLOW}[4/8]${NC} Checking network interfaces..."
if ip addr show lo | grep -q "127.0.0.1"; then
    echo -e "${GREEN}✓${NC} Loopback interface is active"
    ip addr show lo | grep "inet"
else
    echo -e "${RED}✗${NC} Loopback interface not found"
fi

# Check Docker
echo -e "\n${YELLOW}[5/8]${NC} Checking Docker..."
if command -v docker &> /dev/null; then
    if docker info &> /dev/null; then
        echo -e "${GREEN}✓${NC} Docker is running"
        docker version --format '  Client: {{.Client.Version}}'
        docker version --format '  Server: {{.Server.Version}}'
    else
        echo -e "${RED}✗${NC} Docker is not running"
        echo "  Start Docker Desktop and ensure WSL2 integration is enabled"
    fi
else
    echo -e "${RED}✗${NC} Docker is not installed"
fi

# Check Docker Compose
echo -e "\n${YELLOW}[6/8]${NC} Checking Docker Compose..."
if docker compose version &> /dev/null; then
    echo -e "${GREEN}✓${NC} Docker Compose is available"
    docker compose version
else
    echo -e "${RED}✗${NC} Docker Compose is not available"
fi

# Test ROS 2 connectivity (if available)
echo -e "\n${YELLOW}[7/8]${NC} Testing ROS 2 topic discovery..."
if command -v ros2 &> /dev/null; then
    echo "  Available topics:"
    ros2 topic list 2>/dev/null || echo "  (No topics yet - start containers first)"
else
    echo "  ROS 2 CLI not installed in WSL2 base"
    echo "  Install with: sudo apt install ros-humble-ros2cli"
fi

# Check Windows Isaac Sim connectivity
echo -e "\n${YELLOW}[8/8]${NC} Checking Windows Isaac Sim connectivity..."
if ping -c 1 127.0.0.1 > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Loopback is reachable (Windows <-> WSL2 connection)"
else
    echo -e "${RED}✗${NC} Loopback not reachable"
fi

echo -e "\n=========================================="
echo -e "${GREEN}Validation complete!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Ensure Isaac Sim is running on Windows with ROS 2 Bridge enabled"
echo "2. Run: docker compose --profile autonomy up -d"
echo "3. Test with: ros2 topic echo /chatter std_msgs/String"
echo ""
echo "If any checks failed, review the troubleshooting section in"
echo "docs/WSL2_ISAAC_SIM_ROS2_SETUP.md"
