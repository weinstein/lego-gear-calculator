from itertools import tee

def memoize(f):
    memo = {}
    def inner(*args, **kwargs):
        k = args, frozenset(kwargs.items())
        if k not in memo:
            memo[k] = f(*args, **kwargs)
        return memo[k]
    return inner

def memoize_generator(f):
    memo = {}
    def inner(*args, **kwargs):
        k = args, frozenset(kwargs.items())
        it = memo[k] if k in memo else f(*args, **kwargs)
        memo[k], r = tee(it)
        return r
    return inner
