#!/usr/bin/env bash
# =============================================================================
# Project Sanjay Mk2 - WSL2 Ubuntu Provisioning
# =============================================================================
#
# Usage:
#   bash scripts/setup_wsl2_env.sh --as-root
#   bash scripts/setup_wsl2_env.sh --as-user
#   bash scripts/setup_wsl2_env.sh             # best-effort mode
#
# =============================================================================

set -euo pipefail

MODE="all"
SKIP_VALIDATE=0

for arg in "$@"; do
    case "$arg" in
        --as-root) MODE="root" ;;
        --as-user) MODE="user" ;;
        --skip-validate) SKIP_VALIDATE=1 ;;
        *)
            echo "Unknown argument: $arg" >&2
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REQ_FILE="$PROJECT_ROOT/requirements.txt"
VENV_DIR="/opt/sanjay_venv"
ROS_KEYRING="/usr/share/keyrings/ros-archive-keyring.gpg"
ROS_LIST="/etc/apt/sources.list.d/ros2.list"
NVIDIA_KEYRING="/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
NVIDIA_LIST="/etc/apt/sources.list.d/nvidia-container-toolkit.list"
MAVSDK_SERVER_URL="https://github.com/mavlink/MAVSDK/releases/latest/download/mavsdk_server_linux-x64-musl"
MAVSDK_SERVER_PATH="/usr/local/bin/mavsdk_server"

MARKER_BEGIN="# === ROS 2 + Fast DDS Config for WSL2 <-> Isaac Sim ==="
MARKER_END="# === END ROS 2 Config ==="

log() {
    printf '[setup] %s\n' "$1"
}

require_root() {
    if [[ "${EUID}" -ne 0 ]]; then
        echo "This mode requires root. Re-run with sudo or via bootstrap_wsl2.ps1." >&2
        exit 1
    fi
}

detect_target_user() {
    if [[ -n "${WSL_TARGET_USER:-}" ]]; then
        echo "${WSL_TARGET_USER}"
        return
    fi
    if [[ -n "${SUDO_USER:-}" && "${SUDO_USER}" != "root" ]]; then
        echo "${SUDO_USER}"
        return
    fi
    local uid_user
    uid_user="$(getent passwd 1000 | cut -d: -f1 || true)"
    if [[ -n "$uid_user" ]]; then
        echo "$uid_user"
        return
    fi
    echo "root"
}

setup_apt_and_base_packages() {
    log "Installing base packages..."
    apt-get update
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        wget \
        git \
        gnupg2 \
        lsb-release \
        software-properties-common \
        build-essential \
        pkg-config \
        python3-pip \
        python3-venv \
        libgl1 \
        libglib2.0-0 \
        libglx-mesa0 \
        libgl1-mesa-dri \
        mesa-utils \
        x11-apps \
        unzip \
        net-tools

    if ! apt-cache show python3.11 >/dev/null 2>&1; then
        log "Adding deadsnakes PPA for Python 3.11..."
        add-apt-repository -y ppa:deadsnakes/ppa
        apt-get update
    fi

    apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3.11-dev
}

setup_ros2_humble() {
    log "Configuring ROS 2 apt repository..."
    if [[ ! -f "$ROS_KEYRING" ]]; then
        curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc \
            | gpg --dearmor -o "$ROS_KEYRING"
    fi

    local codename
    codename="$(. /etc/os-release && echo "$UBUNTU_CODENAME")"
    echo "deb [arch=$(dpkg --print-architecture) signed-by=${ROS_KEYRING}] http://packages.ros.org/ros2/ubuntu ${codename} main" \
        > "$ROS_LIST"

    apt-get update
    apt-get install -y --no-install-recommends \
        ros-humble-desktop \
        ros-humble-rmw-fastrtps-cpp \
        ros-humble-cv-bridge \
        ros-humble-sensor-msgs \
        ros-humble-nav-msgs \
        ros-humble-geometry-msgs \
        python3-colcon-common-extensions
}

setup_nvidia_container_toolkit() {
    log "Configuring NVIDIA Container Toolkit..."
    if [[ ! -f "$NVIDIA_KEYRING" ]]; then
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
            | gpg --dearmor -o "$NVIDIA_KEYRING"
    fi

    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed "s#deb https://#deb [signed-by=${NVIDIA_KEYRING}] https://#g" \
        > "$NVIDIA_LIST"

    apt-get update
    apt-get install -y --no-install-recommends nvidia-container-toolkit

    if command -v nvidia-ctk >/dev/null 2>&1; then
        if command -v docker >/dev/null 2>&1; then
            nvidia-ctk runtime configure --runtime=docker
        else
            log "Docker CLI not found yet; skipping nvidia-ctk runtime configure."
        fi
    fi
}

setup_project_venv_and_dependencies() {
    log "Creating project venv at ${VENV_DIR}..."
    python3.11 -m venv "$VENV_DIR"
    "${VENV_DIR}/bin/pip" install --upgrade pip setuptools wheel
    "${VENV_DIR}/bin/pip" install -r "$REQ_FILE"
}

install_mavsdk_server() {
    log "Installing MAVSDK server..."
    curl -fsSL "$MAVSDK_SERVER_URL" -o "$MAVSDK_SERVER_PATH"
    chmod +x "$MAVSDK_SERVER_PATH"
}

ensure_docker_group_membership() {
    local user_name="$1"
    if [[ "$user_name" == "root" ]]; then
        log "Default user resolved to root; skipping docker group membership."
        return
    fi

    if ! getent group docker >/dev/null 2>&1; then
        groupadd docker
    fi
    usermod -aG docker "$user_name"
    log "Added ${user_name} to docker group."
}

write_bashrc_config() {
    local user_name="$1"
    local home_dir
    home_dir="$(getent passwd "$user_name" | cut -d: -f6)"
    if [[ -z "$home_dir" ]]; then
        echo "Unable to resolve home directory for user '$user_name'." >&2
        exit 1
    fi
    local bashrc_path="${home_dir}/.bashrc"

    if [[ -f "$bashrc_path" ]] && grep -q "$MARKER_BEGIN" "$bashrc_path"; then
        sed -i "/${MARKER_BEGIN}/,/${MARKER_END}/d" "$bashrc_path"
    fi

    cat >> "$bashrc_path" <<EOF

${MARKER_BEGIN}
source /opt/ros/humble/setup.bash
if [ -f /opt/sanjay_venv/bin/activate ]; then
    source /opt/sanjay_venv/bin/activate
fi

export ROS_DOMAIN_ID=10
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=/opt/config/fastdds_profiles.xml
export ROS_LOCALHOST_ONLY=0

alias rostop='ros2 topic list'
alias rosecho='ros2 topic echo'
alias rosrun='ros2 run'
alias roslaunch='ros2 launch'
alias dc='docker compose'
alias dcup='docker compose --profile autonomy up -d'
alias dclog='docker compose logs -f'
alias dcdown='docker compose down'
${MARKER_END}
EOF
}

run_validation_if_available() {
    if [[ "$SKIP_VALIDATE" -eq 1 ]]; then
        log "Validation skipped by flag."
        return
    fi

    if [[ -f "$PROJECT_ROOT/scripts/validate_setup.sh" ]]; then
        log "Running setup validation..."
        bash "$PROJECT_ROOT/scripts/validate_setup.sh" || true
    fi
}

run_root_phase() {
    require_root
    log "Running root provisioning phase..."
    setup_apt_and_base_packages
    setup_ros2_humble
    setup_nvidia_container_toolkit
    setup_project_venv_and_dependencies
    install_mavsdk_server
    ensure_docker_group_membership "$(detect_target_user)"
    log "Root provisioning phase complete."
}

run_user_phase() {
    local user_name="${WSL_TARGET_USER:-$(id -un)}"
    log "Running user configuration phase for ${user_name}..."
    if [[ "${EUID}" -eq 0 ]]; then
        write_bashrc_config "$user_name"
    else
        if [[ "$user_name" != "$(id -un)" ]]; then
            echo "Cannot configure another user's .bashrc without root privileges." >&2
            exit 1
        fi
        local bashrc_path="${HOME}/.bashrc"
        if [[ -f "$bashrc_path" ]] && grep -q "$MARKER_BEGIN" "$bashrc_path"; then
            sed -i "/${MARKER_BEGIN}/,/${MARKER_END}/d" "$bashrc_path"
        fi
        cat >> "$bashrc_path" <<EOF

${MARKER_BEGIN}
source /opt/ros/humble/setup.bash
if [ -f /opt/sanjay_venv/bin/activate ]; then
    source /opt/sanjay_venv/bin/activate
fi

export ROS_DOMAIN_ID=10
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=/opt/config/fastdds_profiles.xml
export ROS_LOCALHOST_ONLY=0

alias rostop='ros2 topic list'
alias rosecho='ros2 topic echo'
alias rosrun='ros2 run'
alias roslaunch='ros2 launch'
alias dc='docker compose'
alias dcup='docker compose --profile autonomy up -d'
alias dclog='docker compose logs -f'
alias dcdown='docker compose down'
${MARKER_END}
EOF
    fi
    run_validation_if_available
    log "User configuration phase complete."
}

case "$MODE" in
    root)
        run_root_phase
        ;;
    user)
        run_user_phase
        ;;
    all)
        if [[ "${EUID}" -eq 0 ]]; then
            run_root_phase
            run_user_phase
        else
            log "Not running as root. Executing user phase only."
            run_user_phase
        fi
        ;;
esac
