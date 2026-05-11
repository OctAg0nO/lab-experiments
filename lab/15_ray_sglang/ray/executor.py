"""ModuleExecutor — Dual-path executor ABC.

Mirrors the dual-path pattern from Lab 10/14 (InMemoryFrontier / DaprFrontier):
- InProcessExecutor: executes DSPy modules in-process (dev, no infra)
- RayModuleExecutor: executes DSPy modules as Ray tasks (production, cluster)

The circuit-breaker pattern ensures Ray failures don't crash the agent.
On timeout or worker crash, execution falls back to InProcessExecutor.
"""

from __future__ import annotations

import logging
import pickle
import weakref
from abc import ABC, abstractmethod
from typing import Any

import dspy

logger = logging.getLogger(__name__)

# Module-level Ray import guard — clean error if ray not installed
_HAS_RAY = False
_ray = None
_RayExceptions = None
try:
    import ray as _ray
    from ray import exceptions as _RayExceptions
    _HAS_RAY = True
except ImportError:
    pass


class RayNotAvailableError(ImportError):
    """Raised when Ray is required but not installed.

     pip install ray[default]
    """


class ModuleExecutor(ABC):
    """ABC for executing DSPy modules. Mirrors Frontier/Stack dual-path pattern."""

    @abstractmethod
    def execute(self, module: dspy.Module, **kwargs) -> dspy.Prediction:
        ...

    @abstractmethod
    def execute_batch(
        self, modules: list[dspy.Module], batch_kwargs: list[dict]
    ) -> list[dspy.Prediction]:
        ...


class InProcessExecutor(ModuleExecutor):
    """Default: execute modules in the current process. Zero infrastructure."""

    def execute(self, module: dspy.Module, **kwargs) -> dspy.Prediction:
        return module(**kwargs)

    def execute_batch(
        self, modules: list[dspy.Module], batch_kwargs: list[dict]
    ) -> list[dspy.Prediction]:
        return [m(**kw) for m, kw in zip(modules, batch_kwargs)]


def _pickle_module(module: dspy.Module) -> bytes | None:
    """Serialize a DSPy module for Ray transport. Returns None if not picklable."""
    try:
        return pickle.dumps(module)
    except (pickle.PicklingError, AttributeError, TypeError) as e:
        module_type = type(module).__name__
        if module_type not in _pickle_warnings_logged:
            _pickle_warnings_logged.add(module_type)
            logger.warning(
                "Module %s cannot be pickled: %s. "
                "Falling back to in-process execution.",
                module_type, e,
            )
        return None

_pickle_warnings_logged: set[str] = set()


class RayModuleExecutor(ModuleExecutor):
    """Execute modules as Ray tasks across a cluster.

    Modules are serialized via pickle.dumps(); bytes are sent to workers.
    Serialized bytes are cached per module via WeakKeyDictionary.

    The Ray remote function is created once in __init__ (not per-call).

    Circuit-breaker: on timeout or worker crash, falls back to InProcessExecutor.
    Call reset_circuit_breaker() at the start of each workflow iteration.

    Raises RayNotAvailableError if Ray is not installed.
    """

    def __init__(
        self,
        num_gpus: float = 0,
        num_cpus: float = 1,
        timeout: float = 300,
    ):
        if not _HAS_RAY:
            raise RayNotAvailableError(
                "Ray is required for RayModuleExecutor. "
                "Install it with: pip install ray[default]"
            )

        if not _ray.is_initialized():
            _ray.init(ignore_reinit_error=True)

        self.num_gpus = num_gpus
        self.num_cpus = num_cpus
        self.timeout = timeout
        self._circuit_open = False
        self._fallback_count = 0
        self._fallback = InProcessExecutor()
        self._module_bytes_cache: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

        @_ray.remote(num_gpus=self.num_gpus, num_cpus=self.num_cpus)
        def _execute_remote(module_bytes: bytes, kwargs: dict) -> dict:
            module = pickle.loads(module_bytes)
            prediction = module(**kwargs)
            return {"prediction": prediction}

        self._remote_fn = _execute_remote

    def _get_module_bytes(self, module: dspy.Module) -> bytes | None:
        """Get pickled bytes for a module, cached per module via weak ref."""
        cached = self._module_bytes_cache.get(module)
        if cached is None:
            cached = _pickle_module(module)
            if cached is not None:
                self._module_bytes_cache[module] = cached
        return cached

    def execute(self, module: dspy.Module, **kwargs) -> dspy.Prediction:
        if self._circuit_open:
            self._fallback_count += 1
            return self._fallback.execute(module, **kwargs)

        module_bytes = self._get_module_bytes(module)
        if module_bytes is None:
            return self._fallback.execute(module, **kwargs)

        try:
            ref = self._remote_fn.remote(module_bytes, kwargs)
            result = _ray.get(ref, timeout=self.timeout)
            return result["prediction"]
        except _RayExceptions.GetTimeoutError:
            logger.error("Ray task timed out after %ds. Opening circuit.", self.timeout)
            self._circuit_open = True
            self._fallback_count += 1
            return self._fallback.execute(module, **kwargs)
        except _RayExceptions.WorkerCrashedError as e:
            logger.error("Ray worker crashed: %s. Opening circuit.", e)
            self._circuit_open = True
            self._fallback_count += 1
            return self._fallback.execute(module, **kwargs)
        except Exception as e:
            logger.error("Unexpected Ray error: %s. Opening circuit.", e)
            self._circuit_open = True
            self._fallback_count += 1
            return self._fallback.execute(module, **kwargs)

    def execute_batch(
        self, modules: list[dspy.Module], batch_kwargs: list[dict]
    ) -> list[dspy.Prediction]:
        """Execute modules in parallel, preserving input order.

        Each task is handled independently: if one Ray task fails, only
        that task falls back to InProcessExecutor. Successful results
        from other tasks are preserved.
        """
        if self._circuit_open:
            self._fallback_count += 1
            return self._fallback.execute_batch(modules, batch_kwargs)

        results_by_idx: dict[int, dspy.Prediction] = {}
        ray_refs_by_idx: dict[int, Any] = {}

        for idx, (m, kw) in enumerate(zip(modules, batch_kwargs)):
            module_bytes = self._get_module_bytes(m)
            if module_bytes is not None:
                ray_refs_by_idx[idx] = self._remote_fn.remote(module_bytes, kw)
            else:
                results_by_idx[idx] = self._fallback.execute(m, **kw)

        if ray_refs_by_idx:
            any_failed = False
            for idx, ref in ray_refs_by_idx.items():
                try:
                    result = _ray.get(ref, timeout=self.timeout)
                    results_by_idx[idx] = result["prediction"]
                except (_RayExceptions.GetTimeoutError, _RayExceptions.WorkerCrashedError) as e:
                    logger.warning(
                        "Ray task %d failed: %s. Falling back to in-process.", idx, e
                    )
                    results_by_idx[idx] = self._fallback.execute(
                        modules[idx], **batch_kwargs[idx]
                    )
                    any_failed = True
                except Exception as e:
                    logger.error("Unexpected Ray error on task %d: %s.", idx, e)
                    results_by_idx[idx] = self._fallback.execute(
                        modules[idx], **batch_kwargs[idx]
                    )
                    any_failed = True

            if any_failed:
                self._circuit_open = True
                self._fallback_count += 1

        return [results_by_idx[i] for i in range(len(modules))]

    def reset_circuit_breaker(self):
        """Reset the circuit breaker. Call at start of each workflow iteration."""
        if self._circuit_open:
            logger.info("Resetting Ray circuit breaker.")
            self._circuit_open = False

    @property
    def fallback_count(self) -> int:
        return self._fallback_count
