import _thread
import asyncio
import functools
import inspect
import typing
from collections import namedtuple

from ._cachebox import BaseCacheImpl, FIFOCache

KT = typing.TypeVar("KT")
VT = typing.TypeVar("VT")
DT = typing.TypeVar("DT")
FT = typing.TypeVar("FT", bound=typing.Callable[..., typing.Any])


class Frozen(BaseCacheImpl[KT, VT]):  # pragma: no cover
    """
    A wrapper class that prevents modifications to an underlying cache implementation.

    This class provides a read-only view of a cache, optionally allowing silent
    suppression of modification attempts instead of raising exceptions.
    """

    __slots__ = ("__cache", "ignore")

    def __init__(self, cls: BaseCacheImpl[KT, VT], ignore: bool = False) -> None:
        """
        Initialize a frozen cache wrapper.

        :param cls: The underlying cache implementation to be frozen
        :type cls: BaseCacheImpl[KT, VT]
        :param ignore: If True, silently ignores modification attempts; if False, raises TypeError when modification is attempted
        :type ignore: bool, optional
        """
        assert isinstance(cls, BaseCacheImpl)
        assert type(cls) is not Frozen

        self.__cache = cls
        self.ignore = ignore

    @property
    def cache(self) -> BaseCacheImpl[KT, VT]:
        return self.__cache

    @property
    def maxsize(self) -> int:
        return self.__cache.maxsize

    @property
    def maxmemory(self) -> int:
        return self.__cache.maxmemory

    def __len__(self) -> int:
        return len(self.__cache)

    def __sizeof__(self) -> int:
        return self.__cache.__sizeof__()

    def __bool__(self) -> bool:
        return bool(self.__cache)

    def __contains__(self, key: KT) -> bool:
        return key in self.__cache

    def __setitem__(self, key: KT, value: VT) -> None:
        if self.ignore:
            return

        raise TypeError("This cache is frozen.")

    def __getitem__(self, key: KT) -> VT:
        return self.__cache[key]

    def __delitem__(self, key: KT) -> None:
        if self.ignore:
            return None

        raise TypeError("This cache is frozen.")

    def __repr__(self) -> str:
        return f"<Frozen: {self.__cache}>"

    def __iter__(self) -> typing.Iterator[KT]:
        return iter(self.__cache)

    def __richcmp__(self, other: typing.Any, op: int) -> bool:
        return self.__cache.__richcmp__(other, op)

    def capacity(self) -> int:
        return self.__cache.capacity()

    def memory(self) -> int:
        return self.__cache.memory()

    def is_full(self) -> bool:
        return self.__cache.is_full()

    def is_empty(self) -> bool:
        return self.__cache.is_empty()

    def insert(self, key: KT, value: VT, *args, **kwargs) -> typing.Optional[VT]:
        if self.ignore:
            return None

        raise TypeError("This cache is frozen.")

    def get(self, key: KT, default: typing.Optional[DT] = None) -> typing.Union[VT, DT]:
        return self.__cache.get(key, default)

    def pop(self, key: KT, default: typing.Optional[DT] = None) -> typing.Union[VT, DT]:
        if self.ignore:
            return None  # type: ignore[return-value]

        raise TypeError("This cache is frozen.")

    def setdefault(
        self, key: KT, default: typing.Optional[DT] = None, *args, **kwargs
    ) -> typing.Optional[typing.Union[VT, DT]]:
        if self.ignore:
            return None

        raise TypeError("This cache is frozen.")

    def popitem(self) -> typing.Tuple[KT, VT]:
        if self.ignore:
            return  # type: ignore

        raise TypeError("This cache is frozen.")

    def drain(self, n: int) -> int:
        if self.ignore:
            return  # type: ignore

        raise TypeError("This cache is frozen.")

    def clear(self, *, reuse: bool = False) -> None:
        if self.ignore:
            return

        raise TypeError("This cache is frozen.")

    def shrink_to_fit(self) -> None:
        if self.ignore:
            return

        raise TypeError("This cache is frozen.")

    def update(
        self,
        iterable: typing.Union[typing.Iterable[typing.Tuple[KT, VT]], typing.Dict[KT, VT]],
        *args,
        **kwargs,
    ) -> None:
        if self.ignore:
            return

        raise TypeError("This cache is frozen.")

    def keys(self) -> typing.Iterable[KT]:
        return self.__cache.keys()

    def values(self) -> typing.Iterable[VT]:
        return self.__cache.values()

    def items(self) -> typing.Iterable[typing.Tuple[KT, VT]]:
        return self.__cache.items()


class _LockWithCounter:
    """One computation and its registered callers for a cache key."""

    __slots__ = ("lock", "waiters", "error")

    def __init__(self, is_async: bool = False):
        self.lock = _thread.allocate_lock() if not is_async else asyncio.Lock()
        self.waiters = 0
        self.error: typing.Optional[Exception] = None

    async def __aenter__(self) -> None:
        await self.lock.acquire()  # type: ignore[misc]

    async def __aexit__(self, *args, **kwds) -> None:
        self.lock.release()

    def __enter__(self) -> None:
        self.lock.acquire()

    def __exit__(self, *args, **kwds) -> None:
        self.lock.release()


class _LockRegistry:
    """Own per-key state only while a caller computes or waits.

    Reserve a caller under the registry guard before it acquires the per-key
    lock. This prevents removal while another thread is about to wait.
    """

    def __init__(self, is_async: bool = False):
        self._entries: typing.Dict[typing.Hashable, _LockWithCounter] = {}
        self._guard = _thread.allocate_lock()
        self._is_async = is_async

    def reserve(self, key: typing.Hashable) -> _LockWithCounter:
        with self._guard:
            state = self._entries.get(key)
            if state is None:
                state = self._entries[key] = _LockWithCounter(self._is_async)
            state.waiters += 1
            return state

    def release(self, key: typing.Hashable, state: _LockWithCounter) -> None:
        with self._guard:
            state.waiters -= 1
            if state.waiters == 0:
                if self._entries.get(key) is state:
                    del self._entries[key]
                # Tracebacks can refer back to this state through wrapper frames.
                state.error = None

    def clear(self) -> None:
        with self._guard:
            for state in self._entries.values():
                state.error = None
            self._entries.clear()


def _copy_if_need(obj: VT, tocopy=(dict, list, set), level: int = 1) -> VT:
    from copy import copy

    if level == 0:
        return obj

    if level == 2:
        return copy(obj)

    return copy(obj) if (type(obj) in tocopy) else obj


def make_key(args: tuple, kwds: dict, fasttype=(int, str)):
    """
    Create a hashable key from function arguments for caching purposes.

    Args:
        args (tuple): Positional arguments to be used in key generation.
        kwds (dict): Keyword arguments to be used in key generation.
        fasttype (tuple, optional): Types that can be directly used as keys. Defaults to (int, str).

    Returns:
        A hashable key representing the function arguments, optimized for simple single-argument cases.
    """
    key = args
    if kwds:
        key += (object,)
        for item in kwds.items():
            key += item

    if fasttype and len(key) == 1 and type(key[0]) in fasttype:
        return key[0]

    return key


def make_hash_key(args: tuple, kwds: dict):
    """
    Create a hashable hash key from function arguments for caching purposes.

    Args:
        args (tuple): Positional arguments to be used in key generation.
        kwds (dict): Keyword arguments to be used in key generation.

    Returns:
        int: A hash value representing the function arguments.
    """
    return hash(make_key(args, kwds))


def make_typed_key(args: tuple, kwds: dict):
    """
    Create a hashable key from function arguments that includes type information.

    Args:
        args (tuple): Positional arguments to be used in key generation.
        kwds (dict): Keyword arguments to be used in key generation.

    Returns:
        A hashable key representing the function arguments, including the types of the arguments.
    """
    key = make_key(args, kwds, fasttype=())

    key += tuple(type(v) for v in args)
    if kwds:
        key += tuple(type(v) for v in kwds.values())

    return key


CacheInfo = namedtuple("CacheInfo", ["hits", "misses", "maxsize", "length", "memory"])
EVENT_MISS = 1
EVENT_HIT = 2


def _cached_wrapper(
    func,
    cache: typing.Union[BaseCacheImpl, typing.Callable],
    key_maker: typing.Callable[[tuple, dict], typing.Hashable],
    clear_reuse: bool,
    callback: typing.Optional[typing.Callable[[int, typing.Any, typing.Any], typing.Any]],
    copy_level: int,
    is_method: bool,
):
    is_method = cache_is_function = inspect.isfunction(cache)
    _key_maker = (lambda args, kwds: key_maker(args[1:], kwds)) if is_method else key_maker

    hits = 0
    misses = 0
    locks = _LockRegistry()

    def _wrapped(*args, **kwds):
        nonlocal hits, misses

        if kwds.pop("cachebox__ignore", False):
            return func(*args, **kwds)

        _cache = cache(args[0]) if cache_is_function else cache
        key = _key_maker(args, kwds)

        # try to get result from cache
        try:
            result = _cache[key]
        except KeyError:
            pass
        else:
            # A NOTE FOR ME: we don't want to catch KeyError exceptions from `callback`
            # so don't wrap it with try except
            hits += 1

            if callback is not None:
                callback(EVENT_HIT, key, result)

            return _copy_if_need(result, level=copy_level)

        state = locks.reserve(key)
        try:
            with state:
                if state.error is not None:
                    raise state.error

                try:
                    result = _cache[key]
                    hits += 1
                    event = EVENT_HIT
                except KeyError:
                    try:
                        result = func(*args, **kwds)
                    except Exception as error:
                        state.error = error
                        raise
                    else:
                        _cache[key] = result
                        misses += 1
                        event = EVENT_MISS
        finally:
            # Release the registered caller even if lock acquisition fails.
            locks.release(key, state)

        if callback is not None:
            callback(event, key, result)

        return _copy_if_need(result, level=copy_level)

    if not cache_is_function:
        _wrapped.cache = cache
        _wrapped.cache_info = lambda: CacheInfo(
            hits, misses, cache.maxsize, len(cache), cache.memory()
        )

    _wrapped.callback = callback

    if not cache_is_function:

        def cache_clear() -> None:
            nonlocal misses, hits
            cache.clear(reuse=clear_reuse)
            misses = 0
            hits = 0
            locks.clear()

        _wrapped.cache_clear = cache_clear

    return _wrapped


def _async_cached_wrapper(
    func,
    cache: typing.Union[BaseCacheImpl, typing.Callable],
    key_maker: typing.Callable[[tuple, dict], typing.Hashable],
    clear_reuse: bool,
    callback: typing.Optional[typing.Callable[[int, typing.Any, typing.Any], typing.Any]],
    copy_level: int,
    is_method: bool,
):
    is_method = cache_is_function = inspect.isfunction(cache)
    _key_maker = (lambda args, kwds: key_maker(args[1:], kwds)) if is_method else key_maker

    hits = 0
    misses = 0
    locks = _LockRegistry(is_async=True)

    async def _wrapped(*args, **kwds):
        nonlocal hits, misses

        if kwds.pop("cachebox__ignore", False):
            return await func(*args, **kwds)

        _cache = cache(args[0]) if cache_is_function else cache
        key = _key_maker(args, kwds)

        # try to get result from cache
        try:
            result = _cache[key]
        except KeyError:
            pass
        else:
            # A NOTE FOR ME: we don't want to catch KeyError exceptions from `callback`
            # so don't wrap it with try except
            hits += 1

            if callback is not None:
                awaitable = callback(EVENT_HIT, key, result)
                if inspect.isawaitable(awaitable):
                    await awaitable

            return _copy_if_need(result, level=copy_level)

        state = locks.reserve(key)
        try:
            async with state:
                if state.error is not None:
                    raise state.error

                try:
                    result = _cache[key]
                    hits += 1
                    event = EVENT_HIT
                except KeyError:
                    try:
                        result = await func(*args, **kwds)
                    except Exception as error:
                        state.error = error
                        raise
                    else:
                        _cache[key] = result
                        misses += 1
                        event = EVENT_MISS
        finally:
            # Also release callers cancelled while acquiring the async lock.
            locks.release(key, state)

        if callback is not None:
            awaitable = callback(event, key, result)
            if inspect.isawaitable(awaitable):
                await awaitable

        return _copy_if_need(result, level=copy_level)

    if not cache_is_function:
        _wrapped.cache = cache
        _wrapped.cache_info = lambda: CacheInfo(
            hits, misses, cache.maxsize, len(cache), cache.memory()
        )

    _wrapped.callback = callback

    if not cache_is_function:

        def cache_clear() -> None:
            nonlocal misses, hits
            cache.clear(reuse=clear_reuse)
            misses = 0
            hits = 0
            locks.clear()

        _wrapped.cache_clear = cache_clear

    return _wrapped


def cached(
    cache: typing.Union[BaseCacheImpl, dict, typing.Callable[..., BaseCacheImpl], None],
    key_maker: typing.Callable[[tuple, dict], typing.Hashable] = make_key,
    clear_reuse: bool = False,
    callback: typing.Optional[typing.Callable[[int, typing.Any, typing.Any], typing.Any]] = None,
    copy_level: int = 1,
) -> typing.Callable[[FT], FT]:
    """
    Decorator to create a memoized cache for function results.

    Wraps a function to automatically cache and retrieve its results based on input parameters.

    Args:
        cache (BaseCacheImpl, dict, callable): Cache implementation to store results. Defaults to FIFOCache.
                                               Can be a function that got `self` and should return cache.
        key_maker (Callable, optional): Function to generate cache keys from function arguments. Defaults to make_key.
        clear_reuse (bool, optional): Whether to reuse cache during clearing. Defaults to False.
        callback (Callable, optional): Function called on cache hit/miss events. Defaults to None.
        copy_level (int, optional): Level of result copying. Defaults to 1.

    Returns:
        Callable: Decorated function with caching capabilities.

    Example for functions::

        @cachebox.cached(cachebox.LRUCache(128))
        def sum_as_string(a, b):
            return str(a+b)

        assert sum_as_string(1, 2) == "3"

        assert len(sum_as_string.cache) == 1
        sum_as_string.cache_clear()
        assert len(sum_as_string.cache) == 0

    Example for methods::

        class A:
            def __init__(self, num):
                self.num = num
                self._cache = cachebox.FIFOCache(0)

            @cachebox.cached(lambda self: self._cache)
            def method(self, n):
                return self.num * n

        instance = A(10)
        assert A.method(2) == 20
    """
    if cache is None:
        cache = FIFOCache(0)

    if type(cache) is dict:
        cache = FIFOCache(0, cache)

    if not isinstance(cache, BaseCacheImpl) and not inspect.isfunction(cache):
        raise TypeError("we expected cachebox caches or function, got %r" % (cache,))

    def decorator(func: FT) -> FT:
        if inspect.iscoroutinefunction(func):
            wrapper = _async_cached_wrapper(
                func, cache, key_maker, clear_reuse, callback, copy_level, False
            )
        else:
            wrapper = _cached_wrapper(
                func, cache, key_maker, clear_reuse, callback, copy_level, False
            )

        return functools.update_wrapper(wrapper, func)  # type: ignore[return-value]

    return decorator


def cachedmethod(
    cache: typing.Union[BaseCacheImpl, dict, None],
    key_maker: typing.Callable[[tuple, dict], typing.Hashable] = make_key,
    clear_reuse: bool = False,
    callback: typing.Optional[typing.Callable[[int, typing.Any, typing.Any], typing.Any]] = None,
    copy_level: int = 1,
) -> typing.Callable[[FT], FT]:
    """
    **This function is deperecated due to issue [#35](https://github.com/awolverp/cachebox/issues/35)**.
    Use `cached` method instead.

    Decorator to create a method-specific memoized cache for function results.

    Similar to `cached()`, but ignores `self` parameter when generating cache keys.

    Args:
        cache (BaseCacheImpl, dict, optional): Cache implementation to store results. Defaults to FIFOCache.
        key_maker (Callable, optional): Function to generate cache keys from function arguments. Defaults to make_key.
        clear_reuse (bool, optional): Whether to reuse cache during clearing. Defaults to False.
        callback (Callable, optional): Function called on cache hit/miss events. Defaults to None.
        copy_level (int, optional): Level of result copying. Defaults to 1.

    Returns:
        Callable: Decorated method with method-specific caching capabilities.
    """
    import warnings

    warnings.warn(
        "cachedmethod is deprecated, use cached instead. see issue https://github.com/awolverp/cachebox/issues/35",
        DeprecationWarning,
        stacklevel=2,
    )

    if cache is None:
        cache = FIFOCache(0)

    if type(cache) is dict:
        cache = FIFOCache(0, cache)

    if not isinstance(cache, BaseCacheImpl):
        raise TypeError("we expected cachebox caches, got %r" % (cache,))

    def decorator(func: FT) -> FT:
        if inspect.iscoroutinefunction(func):
            wrapper = _async_cached_wrapper(
                func, cache, key_maker, clear_reuse, callback, copy_level, True
            )
        else:
            wrapper = _cached_wrapper(
                func, cache, key_maker, clear_reuse, callback, copy_level, True
            )

        return functools.update_wrapper(wrapper, func)  # type: ignore[return-value]

    return decorator


def is_cached(func: object) -> bool:
    """
    Check if a function/method cached by cachebox or not
    """
    return hasattr(func, "cache") and isinstance(func.cache, BaseCacheImpl)
