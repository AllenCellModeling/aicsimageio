"""
back-ported cached_property from the standard library (functools) in python 3.8.

TODO: remove when we drop support for python 3.7

It transforms a method of a class into a property whose value is computed once and then
cached as a normal attribute for the life of the instance. Similar to property(), with
the addition of caching. Useful for expensive computed properties of instances that are
otherwise effectively immutable.

"""
from typing import Any, Callable, Optional

try:
    from functools import cached_property
except ImportError:
    from threading import RLock

    _NOT_FOUND = object()

    class cached_property:  # type: ignore # noqa
        def __init__(self, func: Callable):
            self.func = func
            self.attrname: Optional[str] = None
            self.__doc__ = func.__doc__
            self.lock = RLock()

        def __set_name__(self, owner: object, name: str) -> None:
            if self.attrname is None:
                self.attrname = name
            elif name != self.attrname:
                raise TypeError(
                    "Cannot assign the same cached_property to two different names "
                    f"({self.attrname!r} and {name!r})."
                )

        def __get__(self, instance: object, owner: type = None) -> Any:
            if instance is None:
                return self
            if self.attrname is None:
                raise TypeError(
                    "Cannot use cached_property instance without calling "
                    "__set_name__ on it."
                )
            try:
                # not all objects have __dict__ (e.g. class defines slots)
                cache = instance.__dict__
            except AttributeError:
                msg = (
                    f"No '__dict__' attribute on {type(instance).__name__!r} "
                    f"instance to cache {self.attrname!r} property."
                )
                raise TypeError(msg) from None
            val = cache.get(self.attrname, _NOT_FOUND)
            if val is _NOT_FOUND:
                with self.lock:
                    # check if another thread filled cache while we awaited lock
                    val = cache.get(self.attrname, _NOT_FOUND)
                    if val is _NOT_FOUND:
                        val = self.func(instance)
                        try:
                            cache[self.attrname] = val
                        except TypeError:
                            name = type(instance).__name__
                            msg = (
                                f"The '__dict__' attribute on {name!r} "
                                "instance does not support item assignment for caching "
                                f"{self.attrname!r} property."
                            )
                            raise TypeError(msg) from None
            return val
