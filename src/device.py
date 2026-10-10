"""Device helper — every compute-heavy script imports DEVICE + log_device."""
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def log_device(tag: str = "") -> str:
    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    msg = f"[{tag}] device={DEVICE} gpu={name} torch={torch.__version__} cuda={torch.cuda.is_available()}"
    print(msg, flush=True)
    return msg
