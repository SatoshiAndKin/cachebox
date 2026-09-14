import asyncio
import gc
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor

import pytest
from cachebox import LRUCache, cached


class Key:
    def __init__(self, number):
        self.number = number


def test_sync_eviction_releases_keys_and_keeps_cached_values():
    calls = 0
    cache = LRUCache(4)

    @cached(cache)
    def value(key):
        nonlocal calls
        calls += 1
        return {"number": key.number}

    refs = []
    for number in range(128):
        key = Key(number)
        refs.append(weakref.ref(key))
        result = value(key)
        assert result == {"number": number}
        result["number"] = -1
        assert value(key) == {"number": number}
        del key
    assert calls == 128
    assert len(cache) == 4
    assert [ref().number for ref in refs if ref() is not None] == [124, 125, 126, 127]


@pytest.mark.asyncio
async def test_async_eviction_releases_keys_and_keeps_cached_values():
    calls = 0
    cache = LRUCache(4)

    @cached(cache)
    async def value(key):
        nonlocal calls
        calls += 1
        return {"number": key.number}

    refs = []
    for number in range(128):
        key = Key(number)
        refs.append(weakref.ref(key))
        result = await value(key)
        assert result == {"number": number}
        result["number"] = -1
        assert await value(key) == {"number": number}
        del key
    assert calls == 128
    assert len(cache) == 4
    assert [ref().number for ref in refs if ref() is not None] == [124, 125, 126, 127]


def test_sync_failed_call_releases_key_and_can_retry():
    calls = 0

    @cached(LRUCache(4))
    def value(key):
        nonlocal calls
        calls += 1
        raise ValueError(str(key.number))

    def failed():
        key = Key(42)
        ref = weakref.ref(key)
        for _ in range(2):
            with pytest.raises(ValueError, match="^42$"):
                value(key)
        return ref

    ref = failed()
    gc.collect()
    assert calls == 2
    assert ref() is None


@pytest.mark.asyncio
async def test_async_failed_call_releases_key_and_can_retry():
    calls = 0

    @cached(LRUCache(4))
    async def value(key):
        nonlocal calls
        calls += 1
        raise ValueError(str(key.number))

    async def failed():
        key = Key(42)
        ref = weakref.ref(key)
        for _ in range(2):
            with pytest.raises(ValueError, match="^42$"):
                await value(key)
        return ref

    ref = await failed()
    gc.collect()
    assert calls == 2
    assert ref() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_cancelled_waiter_releases_state_after_owner_finishes(fail):
    entered = asyncio.Event()
    release = asyncio.Event()
    cache = LRUCache(1)
    calls = 0

    @cached(cache)
    async def value(key):
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        if fail:
            raise ValueError(str(key.number))
        return key.number

    async def requests():
        key = Key(42)
        ref = weakref.ref(key)
        owner = asyncio.create_task(value(key))
        await entered.wait()
        waiter = asyncio.create_task(value(key))
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        release.set()
        if fail:
            with pytest.raises(ValueError, match="^42$"):
                await owner
        else:
            assert await owner == 42
        return ref

    ref = await requests()
    assert calls == 1
    if not fail:
        assert await value(Key(43)) == 43
    await asyncio.sleep(0)
    gc.collect()
    assert ref() is None


@pytest.mark.asyncio
async def test_shared_failure_releases_key_after_all_waiters_report():
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    @cached(LRUCache(4))
    async def value(key):
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        raise ValueError(str(key.number))

    async def requests():
        key = Key(42)
        ref = weakref.ref(key)
        owner = asyncio.create_task(value(key))
        await entered.wait()
        waiter = asyncio.create_task(value(key))
        await asyncio.sleep(0)
        release.set()
        outcomes = await asyncio.gather(owner, waiter, return_exceptions=True)
        assert [type(error) for error in outcomes] == [ValueError, ValueError]
        assert [str(error) for error in outcomes] == ["42", "42"]
        assert outcomes[0] is outcomes[1]
        return ref

    ref = await requests()
    await asyncio.sleep(0)
    gc.collect()
    assert calls == 1
    assert ref() is None


@pytest.mark.asyncio
async def test_cancelled_owner_does_not_cancel_surviving_waiter():
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    @cached(LRUCache(4))
    async def value(key):
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return key.number

    key = Key(42)
    owner = asyncio.create_task(value(key))
    await entered.wait()
    waiter = asyncio.create_task(value(key))
    await asyncio.sleep(0)
    owner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await owner
    release.set()
    assert await waiter == 42
    assert calls == 2
    assert await value(key) == 42
    assert calls == 2


def test_threaded_callers_share_results_and_release_evicted_keys():
    cache = LRUCache(4)
    calls = 0

    @cached(cache)
    def value(key):
        nonlocal calls
        calls += 1
        return {"number": key.number}

    def request(key, barrier):
        barrier.wait(timeout=10)
        return value(key)

    refs = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for number in range(32):
            key = Key(number)
            refs.append(weakref.ref(key))
            barrier = threading.Barrier(9)
            futures = [executor.submit(request, key, barrier) for _ in range(8)]
            barrier.wait(timeout=10)
            results = [future.result(timeout=10) for future in futures]
            assert results == [{"number": number}] * 8
            assert len({id(result) for result in results}) == 8
            del key
    assert calls == 32
    assert [ref().number for ref in refs if ref() is not None] == [28, 29, 30, 31]


@pytest.mark.asyncio
async def test_cleared_old_call_cannot_remove_the_new_pending_owner():
    cache = LRUCache(4)
    entered = [asyncio.Event(), asyncio.Event()]
    release = [asyncio.Event(), asyncio.Event()]
    calls = 0

    @cached(cache)
    async def value(key):
        nonlocal calls
        index = calls
        calls += 1
        assert index < 2, "a second owner started for the new pending call"
        entered[index].set()
        await release[index].wait()
        return key.number + index

    key = Key(42)
    old = asyncio.create_task(value(key))
    await entered[0].wait()
    value.cache_clear()
    new = asyncio.create_task(value(key))
    await entered[1].wait()
    release[0].set()
    assert await old == 42
    cache.clear()
    waiter = asyncio.create_task(value(key))
    await asyncio.sleep(0)
    release[1].set()
    assert await asyncio.gather(new, waiter) == [43, 43]
    assert calls == 2
    assert await value(key) == 43
