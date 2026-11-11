DEFAULT_INFERENCE_BACKEND = "torch"
DEFAULT_WORKERS = 4
DEFAULT_THREADS = 8
DEFAULT_BATCH_SIZE = 128
DEFAULT_REPLAY_MIB = 256
DEFAULT_DEVICE = "cpu"

DEPTH_LIMITS = {'easy': 4, 'medium': 8, 'hard': 16, 'expert': None}

def cutoff(difficulty: str, completed_turns: int) -> bool:
    """One su = two player turns; increment only on actual turn completion."""
    if completed_turns < 0:
        raise ValueError("Negative completed turns")
    limit = DEPTH_LIMITS[difficulty]
    return limit is not None and completed_turns >= limit

def device_for(request='auto'):
    import torch
    if request not in ('auto', 'cpu', 'mps'):
        raise ValueError(request)
    available = torch.backends.mps.is_available()
    if request == 'mps' and not available:
        raise RuntimeError('MPS unavailable; use CPU or compatible macOS/PyTorch')
    return torch.device('mps' if available and request != 'cpu' else 'cpu')

def configure(threads=DEFAULT_THREADS):
    import torch
    if threads < 1:
        raise ValueError('threads must be positive')
    torch.set_num_threads(threads)


def validate_inference_device(device, workers, backend):
    if str(device) == 'mps' and backend == 'torch' and workers != 1:
        raise ValueError('MPS inference requires --workers 1. For CPU parallel inference use --device cpu --workers 4.')
