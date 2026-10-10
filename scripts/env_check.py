"""Phase-0 env verification → logs/env_check.txt (PRD 3.1)."""
import platform, subprocess, sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
LOGS = BASE / "logs"
LOGS.mkdir(exist_ok=True)

lines = []
lines.append(f"python={sys.version.split()[0]} platform={platform.platform()}")
try:
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                           "--format=csv"], capture_output=True, text=True, timeout=30)
    lines.append("nvidia-smi:\n" + (smi.stdout.strip() or smi.stderr.strip()))
except Exception as e:
    lines.append(f"nvidia-smi FAILED: {e}")

try:
    import torch
    cuda = torch.cuda.is_available()
    gpu = torch.cuda.get_device_name(0) if cuda else "none"
    lines += [f"torch={torch.__version__}", f"cuda_available={cuda}", f"gpu_name={gpu}"]
    from src.device import log_device
    lines.append(log_device("env_check"))
except Exception as e:
    lines.append(f"torch check FAILED: {e}")

out = "\n".join(lines) + "\n"
(LOGS / "env_check.txt").write_text(out)
print(out)
