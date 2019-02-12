# Standard Library
import logging
import lzma
import pickle
from collections import Counter
from concurrent.futures import ThreadPoolExecutor as ThreadPool
from multiprocessing import cpu_count
from os.path import isfile
from sys import getsizeof
from threading import Lock, Semaphore
from time import time
from typing import Dict, List, Optional

# 3rd Party
import numpy as np
from numpy import ndarray
from numpy.random import choice

# My Code
from utils import root_path, get_text

log = logging.getLogger()

NO_CPUS = cpu_count()
MAX_LOOKBEHIND = 100

NCHAR_PS: List[Optional[Dict[bytes, Dict[int, float]]]] = [None for _ in range(MAX_LOOKBEHIND)]
NCHAR_PS_LKS_R = [Semaphore(NO_CPUS) for _ in range(MAX_LOOKBEHIND)]
NCHAR_PS_LKS_W = [Lock() for _ in range(MAX_LOOKBEHIND)]

COUNTS: Optional[List[int]] = None
COUNTS_LK_R = Semaphore(NO_CPUS)
COUNTS_LK_W = Lock()

PS: Optional[List[float]] = None
PS_LK_R = Semaphore(NO_CPUS)
PS_LK_W = Lock()


def get_counts(save=True, force=False) -> List[int]:
    global COUNTS, COUNTS_LK_R, COUNTS_LK_W
    with COUNTS_LK_R:
        cache_file = root_path('cache', 'word_counts')
        if not force:
            if COUNTS is not None:
                log.debug('[cache hit] got char counts from cache')
                return COUNTS
            elif isfile(cache_file):
                log.debug(f'[cache hit] found char counts in file, loading ...')
                start = time()
                with open(cache_file, mode='rb') as f:
                    COUNTS = pickle.load(f)
                log.info(f'[cache hit] loaded char counts from file (took {time() - start:4.2f}s, size: {getsizeof(COUNTS) / 1e6:4.2f})')
                return COUNTS
        with COUNTS_LK_W:
            log.debug('[generating] char counts')
            start = time()
            bag = Counter(get_text())
            COUNTS = [0 for _ in range(128)]
            for k, v in bag.items():
                if k < 128:
                    COUNTS[k] = v
            if save:
                with open(cache_file, mode='wb') as f:
                    start = time()
                    log.debug(f'[caching] generated char counts')
                    pickle.dump(COUNTS, f)
                    log.debug(f'[finished] caching generated char counts (took {time() - start:4.2f}s)')
            log.info(f'[finished] generating char counts (took {time() - start:4.2f}s, size: {getsizeof(COUNTS) / 1e6:4.2f}MB)')
            return COUNTS


def get_ps(save=True, force=False) -> List[float]:
    global PS, PS_LK_R, PS_LK_W
    cache_file = root_path('cache', 'char_ps')
    with PS_LK_R:
        if not force:
            if PS is not None:
                log.info('[cache hit] got char probabilities from cache')
                return PS
            elif isfile(cache_file):
                log.debug(f'[cache hit] found char probabilities in file, loading ...')
                start = time()
                with open(cache_file, mode='rb') as f:
                    PS = pickle.load(f)
                log.info(f'[cache hit] loaded char probabilities from file (took {time() - start:4.2f}s, size: {getsizeof(PS) / 1e6:4.2f}MB)')
                return PS
        with PS_LK_W:
            start = time()
            log.debug('[generating] char probabilities')
            PS = get_counts(force=force, save=save)
            no_chars: int = sum(PS)
            for c in range(128):
                PS[c] /= no_chars
            if save:
                with open(cache_file, mode='wb') as f:
                    start = time()
                    log.debug(f'[caching] generated char probabilities')
                    pickle.dump(PS, f)
                    log.debug(f'[finished] caching generated char probabilities (took {time() - start:4.2f}s)')
            log.info(f'[finished] generating char probabilities (took {time() - start:4.2f}s, size: {getsizeof(PS) / 1e6:4.2f}MB)')
            return PS


def get_nchar_ps(n=2, save=True, force=False) -> Dict[bytes, Dict[int, float]]:
    assert n >= 1, f'nchar len must be >= 1 but got n = {n}'
    assert n <= 20, f'nchar len must be <= 20 but got n = {n}'
    global NCHAR_PS, NCHAR_PS_LKS_R, NCHAR_PS_LKS_W
    with NCHAR_PS_LKS_R[n]:
        cache_file = root_path('cache', f'{n}char-ps')
        if not force:
            if NCHAR_PS[n] is not None:
                log.info(f'[cache hit] got {n}char probabilities from cache')
                return NCHAR_PS[n]
            elif isfile(cache_file):
                log.debug(f'[cache hit] found char probabilities in file, loading ...')
                start = time()
                with lzma.open(cache_file) as f:
                    NCHAR_PS[n] = pickle.loads(f.read())
                log.info(f'[cache hit] loaded {n}char probabilities from file (took {time() - start:4.2f}s, size: {getsizeof(NCHAR_PS[n]) / 1e6:4.2f}MB)')
                return NCHAR_PS[n]
        with NCHAR_PS_LKS_W[n]:
            start = time()
            log.debug(f'[generating] {n}char probabilities (this might take a couple of minutes)')

            txt: bytes = get_text()

            NCHAR_PS[n] = dict()

            for i in range(len(txt) - n - 1):
                chars_before = txt[i:i + n]
                char_after: int = txt[i + n]
                if chars_before not in NCHAR_PS[n]:
                    NCHAR_PS[n][chars_before] = {char_after: 1}
                elif char_after in NCHAR_PS[n][chars_before]:
                    NCHAR_PS[n][chars_before][char_after] += 1
                else:
                    NCHAR_PS[n][chars_before][char_after] = 1

            for nchar in NCHAR_PS[n]:
                total = 0
                for count in NCHAR_PS[n][nchar].values():
                    total += count
                if total > 0:
                    for char_after in NCHAR_PS[n][nchar]:
                        NCHAR_PS[n][nchar][char_after] /= total

            if save:
                with lzma.open(cache_file, mode='wb') as f:
                    start = time()
                    log.debug(f'[caching] generated {n}char probabilities')
                    f.write(pickle.dumps(NCHAR_PS[n], protocol=-1))
                    log.debug(f'[finished] caching generated {n}char (took {time() - start:4.2f}s)')

            log.info(f'[finished] generating {n}char probabilities (took {time() - start:4.2f}s, size is {getsizeof(NCHAR_PS[n]) / 1e6:4.2f}MB)')
            return NCHAR_PS[n]


def generate(seed=b'That day', n=6, max_len=(1000 * 5), show_metrics=True, save=True, force=False) -> str:
    start = time()
    txt: bytearray = bytearray(seed)
    succ: ndarray = np.array([0 for _ in range(n + 1)], dtype='int32')
    ps: ndarray = np.array(get_ps(force=force, save=save), dtype='float64')
    char_idx: ndarray = np.array(list(range(128)), dtype='int8')

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
