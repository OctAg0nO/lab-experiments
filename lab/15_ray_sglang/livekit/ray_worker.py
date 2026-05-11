"""Ray-scalable LiveKit worker. Each Ray actor runs one LiveKit worker.

Ray is lazily imported — importing this module doesn't require Ray to be installed.
Use create_livekit_worker() to get the Ray actor class.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def create_livekit_worker(num_cpus: int = 2, num_gpus: float = 0.25):
    """Create a Ray actor class for LiveKit workers.

    This factory defers Ray import so the module can be imported
    without Ray installed. The actor is only created when called.

    Usage:
        Worker = create_livekit_worker()
        workers = [Worker.remote(meta_agent) for _ in range(4)]
        ray.get([w.run.remote() for w in workers])
    """
    import ray

    @ray.remote(num_cpus=num_cpus, num_gpus=num_gpus)
    class LiveKitWorker:
        def __init__(self, meta_agent):
            from .worker import create_server
            self._server = create_server(meta_agent)

        def run(self):
            from livekit.agents import cli
            cli.run_app(self._server)

    return LiveKitWorker
