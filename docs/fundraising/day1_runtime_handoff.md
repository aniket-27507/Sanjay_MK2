# Day 1 RTX Runtime Handoff

Use this on the Windows machine with the RTX 4060. This is the recommended Day 1 execution path.

## Recommended remote access path

The simplest route is:

1. SSH from the Mac into the Windows host.
2. From the Windows shell, enter WSL2 Ubuntu.
3. Run the Day 1 launcher inside WSL2.

This is simpler and more reliable than trying to SSH directly into the WSL2 guest.

## One-time Windows SSH setup

Run these in an elevated PowerShell window on the Windows machine:

```powershell
Get-WindowsCapability -Online | Where-Object Name -like 'OpenSSH.Server*'
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Start-Service sshd
Set-Service -Name sshd -StartupType Automatic
New-NetFirewallRule -Name sshd -DisplayName "OpenSSH Server" -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22
```

Confirm the Windows username and host IP:

```powershell
whoami
ipconfig
```

From the Mac:

```bash
ssh <windows_user>@<windows_ip>
```

## Enter WSL2 after SSH

Once you are on the Windows host:

```powershell
wsl -l -v
wsl -d Ubuntu -u <linux_user>
```

If you do not know the Linux username yet, open WSL locally on Windows once and run:

```bash
whoami
```

## Exact Day 1 launch steps in WSL2

1. Go to the repo or clone it if needed.

```bash
cd ~
git clone https://github.com/aniket-27507/Sanjay_MK2.git
cd Sanjay_MK2
```

If the repo already exists:

```bash
cd /path/to/Sanjay_MK2
git pull
```

2. Verify the GPU/runtime baseline.

```bash
python3 --version
nvidia-smi
```

For Day 1 baseline training, `Python 3.10.12` or `Python 3.11.x` is acceptable.
`Python 3.11` is still the preferred path when you later move deeper into Isaac Sim alignment.

3. If the runtime is not provisioned yet, use the existing WSL2 setup path.

Root phase:

```bash
sudo bash scripts/setup_wsl2_env.sh --as-root
```

User phase:

```bash
bash scripts/setup_wsl2_env.sh --as-user
source ~/.bashrc
```

4. Run the Day 1 launcher.

The launcher now auto-detects either a repo-local `.venv` or `/opt/sanjay_venv/bin/python`.

```bash
bash scripts/day1_baseline_pipeline.sh
```

If you want to be explicit about the `/opt` runtime:

```bash
PYTHON_BIN=/opt/sanjay_venv/bin/python bash scripts/day1_baseline_pipeline.sh
```

5. Watch the outputs.

```bash
ls reports/day1
tail -f reports/day1/visdrone_setup.log
tail -f reports/day1/visdrone_baseline_day1.log
```

## What the launcher does

It runs these stages in order:

1. runtime verification via `scripts/verify_training_runtime.py`
2. VisDrone bootstrap via `scripts/train_yolo.py --setup-visdrone`
3. dataset audit via `scripts/audit_dataset.py data/visdrone_police`
4. device selection
5. baseline training via `scripts/train_yolo.py --train --model yolo26n.pt --epochs 30 --name visdrone_baseline_day1`

## Expected outputs

- `reports/day1/runtime_check.json`
- `reports/day1/visdrone_setup.log`
- `reports/day1/visdrone_audit.txt`
- `reports/day1/training_device.txt`
- `reports/day1/visdrone_baseline_day1.log`

## Preflight notes

- The Day 1 baseline uses `yolo26n.pt`.
- If the RTX box has internet, Ultralytics will fetch it automatically on first use.
- If the RTX box is offline or DNS-restricted, place the file at `weights/yolo26n.pt` before launching training.
- VisDrone-only audit warnings for `weapon_person`, `fire`, `explosive_device`, and `crowd` are expected on Day 1 and must not block training launch.

## Success criteria

- `reports/day1/runtime_check.json` shows `training_ready: true`
- `reports/day1/training_device.txt` is `0`
- `data/visdrone_police` exists
- `reports/day1/visdrone_audit.txt` shows `person` and `vehicle`, with the other four classes still missing as expected
- `reports/day1/visdrone_baseline_day1.log` shows epoch progress rather than an immediate crash

## Direct SSH into WSL2 (optional)

If you want to SSH directly into the Linux guest instead of hopping through Windows:

Inside WSL2:

```bash
sudo apt-get update
sudo apt-get install -y openssh-server
sudo service ssh start
hostname -I
whoami
```

Then on Windows, forward a host port to the current WSL2 IP from an elevated PowerShell window:

```powershell
netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=2222 connectaddress=<wsl_ip> connectport=22
New-NetFirewallRule -DisplayName "WSL2 SSH 2222" -Direction Inbound -Protocol TCP -LocalPort 2222 -Action Allow
```

From the Mac:

```bash
ssh -p 2222 <linux_user>@<windows_ip>
```

Important limitation:

- the WSL2 IP usually changes after restart, so the port proxy may need to be recreated unless you are using mirrored networking on a recent Windows 11 setup
