# Standard Library
from collections import ChainMap
from concurrent.futures import ThreadPoolExecutor as ThreadPool
from time import time
from typing import Dict, List, Optional, Generator

# 3rd Party
import numpy as np
from numpy import ndarray
from numpy.random import choice

# My Code
from utils import NO_CPUS, get_char_ps, get_nchar_ps, log


def generate(seed=b'That day', n=6, max_len=(1000 * 5), show_metrics=True) -> Generator[int, None, None]:
    start = time()
    txt: bytearray = bytearray(seed[-n:])
    succ: ndarray = np.array([0 for _ in range(n + 1)], dtype='uint32')
    ps: ndarray = get_char_ps()
    char_idx: ndarray = np.arange(128, dtype='ubyte')

    with ThreadPool(max_workers=NO_CPUS, thread_name_prefix='markov/c') as pool:
        lookup = ChainMap(*[
            task.result() for task in [
                pool.submit(fn=get_nchar_ps, n=(i + 1))
                for i in range(n, 0, -1)]])

    for byte in seed:
        yield byte

    while max_len > 0:
        max_len -= 1
        found = False
        for m in range(n, 0, -1):
            maybe_ps: Optional[Dict[bytes, float]] = \
                    lookup.get(bytes(txt[-m:]), None)
            if maybe_ps is not None and len(maybe_ps) > 1:
                succ[m] += 1
                next_char = choice(
                    a=tuple(maybe_ps.keys()),
                    p=tuple(maybe_ps.values()),
                )
                txt.append(next_char)
                txt = txt[-n:]
                found = True
                yield next_char
                break
        if not found:
            succ[0] += 1
            next_char = choice(a=char_idx, p=ps)
            txt.append(next_char)
            txt = txt[-n:]
            yield next_char

    if show_metrics:
        # metrics
        log.info('-' * (2 + 6 + 15 + 2))
        log.info('%9s%s' % (' ', 'METRICS'))
        log.info('-' * (2 + 6 + 15 + 2))
        log.info('%-2s %-6s %-15s' % ('##', 'PROB', 'NO EXAMPLES'))
        log.info('%-2s %-6s %-15s' % ('-' * 2, '-' * 6, '-' * 15))
        no_gen_char: int = sum(succ)
        for i in range(n, -1, -1):
            log.info('%-2d %-6.4f %-15d' % (i, succ[i] / no_gen_char, succ[i]))

    log.debug(f'[finished] generating text (took {time() - start:4.2f}s)')

    return txt.decode('ascii', 'ignore')
