"""In-memory StateStoreService. No Dapr connection needed."""

from dapr_agents.storage.daprstores.stateservice import StateStoreService


class NoopStore(StateStoreService):
    def __init__(self):
        self._data = {}

    def load(self, *, key, default=None, state_metadata=None, return_model=False):
        return self._data.get(key, default)

    def save(self, *, key, value, etag=None, state_metadata=None, state_options=None, ttl_in_seconds=None):
        self._data[key] = value
