#!/usr/bin/env python3
"""
Cross-platform runtime verification for Day 1 edge-AI work.

Checks:
- Python version
- Core training imports
- CUDA availability for RTX/WSL2
- MPS availability for Mac fallback
- Optional `nvidia-smi` visibility
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys


REQUIRED_MODULES = [
    "torch",
    "torchvision",
    "ultralytics",
    "onnx",
    "onnxruntime",
]


def probe_module(name: str) -> dict:
    cmd = [
        sys.executable,
        "-c",
        (
            "import importlib, json; "
            f"m = importlib.import_module('{name}'); "
            "print(json.dumps({'version': getattr(m, '__version__', 'unknown')}))"
        ),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        payload = json.loads(result.stdout.strip())
        return {"ok": True, "version": payload.get("version", "unknown")}
    except Exception as exc:
        return {"ok": False, "version": None, "error": str(exc)}


def probe_torch_backends() -> dict:
    cmd = [
        sys.executable,
        "-c",
        (
            "import json, torch; "
            "payload = {"
            "'version': torch.__version__, "
            "'cuda_available': bool(torch.cuda.is_available()), "
            "'device_count': int(torch.cuda.device_count()), "
            "'device_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], "
            "'mps_available': bool(hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())"
            "}; "
            "print(json.dumps(payload))"
        ),
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return json.loads(result.stdout.strip())


def run_nvidia_smi() -> dict:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"available": False, "detail": "nvidia-smi not found in PATH"}

    try:
        result = subprocess.run(
            [exe, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return {"available": True, "detail": lines}
    except Exception as exc:
        return {"available": False, "detail": f"nvidia-smi failed: {exc}"}


def main() -> int:
    python_version = sys.version_info[:2]
    python_supported = python_version in {(3, 10), (3, 11)}

    report: dict = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "python_ok": python_supported,
        "python_recommended": python_version == (3, 11),
        "modules": {},
        "cuda": {"available": False, "device_count": 0, "device_names": []},
        "mps": {"available": False},
        "nvidia_smi": {},
    }

    import_failures = []
    for module_name in REQUIRED_MODULES:
        probe = probe_module(module_name)
        report["modules"][module_name] = probe
        if not probe["ok"]:
            import_failures.append(module_name)

    if report["modules"]["torch"]["ok"]:
        try:
            torch_probe = probe_torch_backends()
            report["cuda"]["available"] = torch_probe["cuda_available"]
            report["cuda"]["device_count"] = torch_probe["device_count"]
            report["cuda"]["device_names"] = torch_probe["device_names"]
            report["mps"]["available"] = torch_probe["mps_available"]
        except Exception as exc:
            report["cuda"]["error"] = str(exc)
            report["mps"]["error"] = str(exc)

    report["nvidia_smi"] = run_nvidia_smi()

    training_ready = (
        report["python_ok"]
        and not import_failures
        and (report["cuda"]["available"] or report["mps"]["available"])
    )
    report["training_ready"] = training_ready
    report["recommended_device"] = (
        "0" if report["cuda"]["available"] else ("mps" if report["mps"]["available"] else "cpu")
    )

    print(json.dumps(report, indent=2))
    return 0 if training_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
