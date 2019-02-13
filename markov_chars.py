# Standard Library
from concurrent.futures import ThreadPoolExecutor as ThreadPool
from time import time
from typing import Dict, List, Optional

# 3rd Party
import numpy as np
from numpy import ndarray
from numpy.random import choice

# My Code
from utils import NO_CPUS, get_nchar_ps, get_char_ps, log


def generate(seed=b'That day', n=6, max_len=(1000 * 5), show_metrics=True, save=True, force=False) -> str:
    start = time()
    txt: bytearray = bytearray(seed)
    succ: ndarray = np.array([0 for _ in range(n + 1)], dtype='int32')
    ps: ndarray = get_char_ps(force=force, save=save)
    char_idx: ndarray = np.arange(128, dtype='int8')

    with ThreadPool(max_workers=NO_CPUS, thread_name_prefix='markov/c') as pool:
        ps_nchars: List[Dict[bytes, Dict[int, float]]] = [
            task.result() for task in [
                pool.submit(fn=get_nchar_ps, n=(i + 1), force=force, save=save)
                for i in range(n, 0, -1)]]

    while len(txt) < max_len:
        found = False
        for m in range(n, 0, -1):
            chars: bytes = bytes(txt[-m:])
            maybe_ps: Optional[Dict[bytes, float]] = None
            for d in ps_nchars:
                maybe_ps = d.get(chars, None)
                if maybe_ps:
                    break
            if maybe_ps and len(maybe_ps) > 1:
                succ[m] += 1
                txt.append(choice(
                    a=list(maybe_ps.keys()),
                    p=list(maybe_ps.values()),
                ))
                found = True
                break
        if not found:
            succ[0] += 1
            txt.append(choice(a=char_idx, p=ps))

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
