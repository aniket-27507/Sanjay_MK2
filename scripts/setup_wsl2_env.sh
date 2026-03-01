#!/bin/bash
# WSL2 environment setup for ROS 2 + Isaac Sim
# Appends ROS 2 env vars and aliases to ~/.bashrc

MARKER="# === ROS 2 + Fast DDS Config for WSL2 <-> Isaac Sim ==="

if grep -q "$MARKER" ~/.bashrc 2>/dev/null; then
    echo "ROS 2 config already in ~/.bashrc - skipping"
else
    cat >> ~/.bashrc << 'EOF'

# === ROS 2 + Fast DDS Config for WSL2 <-> Isaac Sim ===
export ROS_DOMAIN_ID=10
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=/opt/config/fastdds_profiles.xml
export ROS_LOCALHOST_ONLY=0

# Useful aliases
alias rostop='ros2 topic list'
alias rosecho='ros2 topic echo'
alias rosrun='ros2 run'
alias roslaunch='ros2 launch'

# Docker Compose aliases
alias dc='docker compose'
alias dcup='docker compose --profile autonomy up -d'
alias dclog='docker compose logs -f'
alias dcdown='docker compose down'
# === END ROS 2 Config ===
EOF
    echo "ROS 2 config added to ~/.bashrc"
fi

# Source it
source ~/.bashrc

# Verify
echo ""
echo "Environment variables set:"
echo "  ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
echo "  RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION"
echo "  FASTRTPS_DEFAULT_PROFILES_FILE=$FASTRTPS_DEFAULT_PROFILES_FILE"
echo "  ROS_LOCALHOST_ONLY=$ROS_LOCALHOST_ONLY"
