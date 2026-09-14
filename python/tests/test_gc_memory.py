"""Native cache operations must permit collection from Python callbacks."""

import json
import subprocess
import sys

import pytest

PROGRAM = r"""
import gc
import json
import sys
import weakref

import cachebox

kind, callback = sys.argv[1:]
cls = getattr(cachebox, kind)
options = {"ttl": 600} if kind in {"TTLCache", "VTTLCache"} else {}
cache = cls(4, **options)
collections = 0

class Key:
    def __hash__(self):
        return 42

    def __eq__(self, other):
        global collections
        if callback == "equality":
            gc.collect()
            collections += 1
        return isinstance(other, Key)

    def __sizeof__(self):
        global collections
        if callback == "size":
            gc.collect()
            collections += 1
        return object.__sizeof__(self)

if callback == "cycle":
    class Value:
        pass
    value = Value()
    value.cache = cache
    reference = weakref.ref(value)
    cache["value"] = value
    assert cache["value"] is value
    del value, cache
    gc.collect()
    assert reference() is None
    print(json.dumps({"collected_cycle": True}))
else:
    key = Key()
    cache[key] = {"value": 42}
    assert cache[Key()] == {"value": 42}
    assert cache[key] == {"value": 42}
    assert collections > 0
    print(json.dumps({"value": 42, "collections": collections}))
"""


@pytest.mark.parametrize(
    "kind", ["Cache", "FIFOCache", "LFUCache", "LRUCache", "TTLCache", "RRCache", "VTTLCache"]
)
@pytest.mark.parametrize("callback", ["size", "equality", "cycle"])
def test_collection_during_native_operations_and_after_cycle_release(kind, callback):
    # A native mutex deadlock cannot be cancelled by a Python test timeout.
    # The child shares the container's limits and imports the installed core.
    result = subprocess.run(
        [sys.executable, "-c", PROGRAM, kind, callback],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    evidence = json.loads(result.stdout)
    if callback == "cycle":
        assert evidence == {"collected_cycle": True}
    else:
        assert evidence["value"] == 42
        assert evidence["collections"] > 0
