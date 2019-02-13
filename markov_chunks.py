# Standard Library
from concurrent.futures import ThreadPoolExecutor as ThreadPool
from time import time
from typing import Dict, List, Optional, Tuple

# 3rd Party
import numpy as np
from numpy import ndarray
from numpy.random import choice

# My Code
from utils import log, NO_CPUS, AVG_CHUNK_LEN, tokenize, get_chunks_ps


def generate(txt=b'That day', n=6, max_avg_txt_len=(1000 * 5), show_metrics=True, save=True, force=False) -> str:
    start = time()
    tokens: List[bytes] = [m.group(0) for m in tokenize(txt)]
    no_tokens = len(tokens)
    succ = [0 for _ in range(n + 1)]
    ps: Dict[bytes, float] = get_chunks_ps(force=force, save=save)
    unique_tokens: ndarray = np.array(list(ps.keys()))
    unique_tokens_ps: ndarray = np.array(list(ps.values()))

    with ThreadPool(max_workers=NO_CPUS, thread_name_prefix='markov/w') as pool:
        ps_ngrams: List[Dict[Tuple, Dict[bytes, float]]] = [
            task.result() for task in [
                pool.submit(fn=get_chunks_ps, n=i, force=force, save=save)
                for i in range(n, 0, -1)]]

    # token generation
    while no_tokens * AVG_CHUNK_LEN < max_avg_txt_len:
        found = False
        for m in range(n, 0, -1):
            ngram = tuple(tokens[-m:])
            maybe_ps: Optional[Dict[bytes, float]] = None
            for d in ps_ngrams:
                maybe_ps = d.get(ngram, None)
                if maybe_ps:
                    break
            if maybe_ps and len(maybe_ps) > 1:
                succ[m] += 1
                tokens.append(choice(
                    a=list(maybe_ps.keys()),
                    p=list(maybe_ps.values()),
                ))
                found = True
                break
        if not found:
            succ[0] += 1
            tokens.append(choice(a=unique_tokens, p=unique_tokens_ps))
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

    # text (outcome)
    return (b''.join(tokens)).decode('ascii', 'ignore')
