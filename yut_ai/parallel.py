"""Spawned CPU rollout workers; frozen model per round, stable result order."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import io
import multiprocessing
import pickle
import signal
import time
import tempfile
from pathlib import Path
import torch
from .model import ModelConfig, PolicyValueNet
from .runtime import DEFAULT_INFERENCE_BACKEND

_worker_model = None

def initialize_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

def frozen_weights(model):
    buffer = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, buffer)
    return buffer.getvalue()

def run_game(job):
    from .selfplay import play
    global _worker_model
    config, weights, arguments = job
    if _worker_model is None or _worker_model.config != ModelConfig(**config):
        _worker_model = PolicyValueNet(ModelConfig(**config))
    _worker_model.load_state_dict(torch.load(io.BytesIO(weights), weights_only=True, map_location='cpu'))
    _worker_model.eval()
    start = time.perf_counter()
    result = play(_worker_model, *arguments)
    # Bytes avoid tensor shared-memory lifetime dependencies across workers.
    return pickle.dumps((result, time.perf_counter()-start))

_coreml_path = None
_coreml_model = None

def run_coreml_game(job):
    from .selfplay import play
    from .coreml_backend import CoreMLPredictor
    global _coreml_path, _coreml_model
    path, arguments = job
    if _coreml_path != path:
        _coreml_model = CoreMLPredictor(path)
        _coreml_path = path
    start = time.perf_counter()
    return pickle.dumps((play(_coreml_model, *arguments), time.perf_counter()-start))


class GameWorkers:
    def __init__(self, count, inference_backend=DEFAULT_INFERENCE_BACKEND):
        self.inference_backend = inference_backend
        self.count = count
        self.pool = None
        if count > 1:
            self.pool = ProcessPoolExecutor(max_workers=count,
                mp_context=multiprocessing.get_context('spawn'), initializer=initialize_worker)

    def generate(self, model, arguments):
        if self.inference_backend == 'coreml-ne':
            from .coreml_backend import export_model, CoreMLPredictor
            # One fresh snapshot per frozen-model round, shared by all workers.
            # Never reuse a conversion after the learner updates its weights.
            with tempfile.TemporaryDirectory(prefix='yut-coreml-round-') as directory:
                path = export_model(model, Path(directory)/'model.mlpackage')
                if self.pool is not None:
                    return [pickle.loads(result) for result in
                            self.pool.map(run_coreml_game, [(path, a) for a in arguments])]
                from .selfplay import play
                predictor = CoreMLPredictor(path)
                results = []
                for args in arguments:
                    start = time.perf_counter()
                    results.append((play(predictor, *args), time.perf_counter()-start))
                return results
        if self.pool is None:
            from .selfplay import play
            results = []
            for args in arguments:
                start = time.perf_counter()
                results.append((play(model, *args), time.perf_counter()-start))
            return results
        weights = frozen_weights(model)
        jobs = [(asdict(model.config), weights, args) for args in arguments]
        return [pickle.loads(result) for result in self.pool.map(run_game, jobs)]

    def close(self, interrupted=False):
        if self.pool:
            if interrupted and hasattr(self.pool, 'terminate_workers'):
                self.pool.terminate_workers()
            else:
                self.pool.shutdown(wait=True, cancel_futures=interrupted)
