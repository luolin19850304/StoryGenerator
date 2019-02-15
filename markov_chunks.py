# Standard Library
from collections import ChainMap
from concurrent.futures import ThreadPoolExecutor as ThreadPool
from time import time
from typing import Dict, List, Optional, Tuple, Generator

# 3rd Party
import numpy as np
from numpy import ndarray
from numpy.random import choice

# My Code
from utils import AVG_CHUNK_LEN, NO_CPUS, chunk, get_chunk_ps, get_nchunks_ps, log


def generate(seed=b'That day', n=6, max_len=(1000 * 5), show_metrics=True) -> Generator[bytes, None, None]:
    start = time()
    tokens: List[bytes] = [m.group(0) for m in list(chunk(seed))[-n:]]
    no_tokens = len(tokens)
    succ = np.array([0 for _ in range(n + 1)], dtype='uint32')
    ps: Dict[bytes, float] = get_chunk_ps()
    chunks: ndarray = np.array(list(ps.keys()))
    chunk_ps: ndarray = np.array(list(ps.values()))

    with ThreadPool(max_workers=NO_CPUS, thread_name_prefix='markov/w') as pool:
        lookup = ChainMap(*[
            task.result() for task in [
                pool.submit(fn=get_nchunks_ps, n=i)
                for i in range(n, 0, -1)]])
    yield seed

    # token generation
    while no_tokens * AVG_CHUNK_LEN < max_len:
        found = False
        for m in range(n, 0, -1):
            ngram = tuple(tokens[-m:])
            maybe_ps: Optional[Dict[bytes, float]] = lookup.get(ngram, None)
            if maybe_ps is not None and len(maybe_ps) > 1:
                found = True
                succ[m] += 1
                next_chunk: bytes = choice(
                    a=list(maybe_ps.keys()),
                    p=list(maybe_ps.values()),
                )
                yield next_chunk
                tokens.append(next_chunk)
                tokens = tokens[-n:]
                break
        if not found:
            succ[0] += 1
            next_chunk = choice(a=chunks, p=chunk_ps)
            yield next_chunk
            tokens.append(next_chunk)
            tokens = tokens[-n:]
        no_tokens += 1

    if show_metrics:
        # metrics
        log.info('-' * (1 + 6 + 15 + 2))
        log.info('%9s%s' % (' ', 'METRICS'))
        log.info('-' * (1 + 6 + 15 + 2))
        log.info('%-1s %-6s %-15s' % ('#', 'PROB', 'NO EXAMPLES'))
        log.info('%-1s %-6s %-15s' % ('-' * 1, '-' * 6, '-' * 15))
        no_gen_tokens: int = sum(succ)
        for i in range(n, -1, -1):
            log.info('%-1d %-6.4f %-15d' % (i, succ[i] / no_gen_tokens, succ[i]))

    log.debug(f'[finished] generating text (took {time() - start:4.2f}s)')
