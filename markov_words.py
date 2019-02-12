# Standard Library
import lzma
import pickle
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor as ThreadPool
from os.path import isfile
from re import IGNORECASE
from sys import getsizeof
from threading import Semaphore, Lock
from time import time
from typing import Dict, Iterable, List, Match, Optional, Tuple

# 3rd Party
import numpy as np
from numpy.random import choice
from numpy import ndarray

# My Code
from utils import log, root_path, get_text

AVG_TOKEN_LEN = 5

SPACE: int = ord(b' ')
DQUOTE: int = ord(b'"')
MAX_LOOKBEHIND = 20
NO_CPUS = 4

PUNCT_REGEX = rb'(?P<punct>-{1,2}|[:;,"])'
NL_REGEX = rb'(?P<nl>(\n\r?|\r\n?)+)'
SENT_END_REGEX = rb'(?P<sent_end>(!(!!)?|\?(\?\?)?|\.(\.\.)?))'
WS_REGEX = rb'(?P<ws>\s)'
WORD_REGEX = rb"(?P<word>[A-Za-z]+(-[A-Za-z]+)*?('[a-z]{0,7})?)"
DATE_REGEX = rb"(?P<date>([1-9]\d*)(th|st|[nr]d)|(19|20)\d{2})"
TIME_REGEX = rb"(?P<time>\d+((:\d{2}){1,2}|(\.\d{2}){1,2}))"
CHUNK_REGEX = re.compile(rb'(?P<token>' + rb'|'.join([
    WS_REGEX,
    SENT_END_REGEX,
    PUNCT_REGEX,
    NL_REGEX,
    WORD_REGEX,
    DATE_REGEX,
    TIME_REGEX,
]) + rb')', IGNORECASE)


NGRAM_PS: List[Optional[Dict[Tuple, Dict[bytes, float]]]] = [None for _ in range(MAX_LOOKBEHIND)]
NGRAM_PS_LKS_R = [Semaphore(NO_CPUS) for _ in range(MAX_LOOKBEHIND)]
NGRAM_PS_LKS_W = [Lock() for _ in range(MAX_LOOKBEHIND)]

TOKENS: Optional[List[bytes]] = None
TOKENS_LK_R = Lock()
TOKENS_LK_W = Lock()

COUNTS: Optional[Counter] = None
COUNTS_LK_R = Semaphore(NO_CPUS)
COUNTS_LK_W = Lock()

PS: Optional[Dict[bytes, float]] = None
PS_LK_R = Semaphore(NO_CPUS)
PS_LK_W = Lock()


def capitalize(txt: bytes) -> bytes:
    if len(txt) <= 1:
        return txt
    pos = 0
    while txt[pos] == SPACE:
        pos += 1
    # is lowercase
    if 97 <= txt[pos] <= 122:
        return txt[:pos] + chr(txt[pos] - 32).encode('ascii', 'ignore') + txt[pos + 1:]
    else:
        return txt


def tokenize(txt: bytes) -> Iterable[Match[bytes]]:
    return CHUNK_REGEX.finditer(txt)


def get_tokens(save=True, force=False) -> List[bytes]:
    global TOKENS
    with TOKENS_LK_R:
        cache_file = root_path('cache', 'tokens')
        if not force:
            if TOKENS is not None:
                log.debug('[cache hit] got tokens from cache')
                return TOKENS
            elif isfile(cache_file):
                log.debug('[cache hit] found tokens in file, loading ...')
                start = time()
                with lzma.open(cache_file) as f:
                    TOKENS = pickle.loads(f.read())
                log.info(f'[cache hit] loaded tokens from file (took {time() - start:4.2f}s, size: {getsizeof(TOKENS) / 1e6:4.2f})')
                return TOKENS
        with TOKENS_LK_W:
            log.debug('[generating] tokens')
            start = time()
            ms: List[Match[bytes]] = list(tokenize(get_text()))
            TOKENS = [ms[0].group(0), ms[0].group(0)]
            # not checking for len of tokens because every token has len >= 1
            for i in range(2, len(ms) - 1):
                s: bytes = ms[i].group(0)
                is_q = s[0] == DQUOTE
                is_w = bool(ms[i].group('word'))
                is_ws = bool(ms[i].group('ws'))
                is_p = bool(ms[i].group('punct'))
                is_nl = bool(ms[i].group('nl'))
                is_end = bool(ms[i].group('sent_end'))
                is_cap = 97 <= s[0] <= 122
                if is_w and ms[i - 2].group('sent_end') and (97 <= TOKENS[-2][0] <= 122):
                    TOKENS[-2] = capitalize(TOKENS[-2])
                if is_w and ms[i - 1].group('word'):
                    TOKENS.append(b'. ' if is_cap else b' ')
                if is_w and is_cap and (ms[i - 2].group('sent_end') or ms[i - 1].group('nl')):
                    TOKENS.append(capitalize(s))
                    continue
                elif (is_w and ms[i + 1].group('word')) \
                        or (is_end and not (ms[i + 1].group('ws') or ms[i + 1].group('nl'))) \
                        or (is_p and not (is_q or ms[i + 1].group('ws') or ms[i - 1].group('nl'))):
                    TOKENS.append(s)
                    TOKENS.append(b' ')
                    continue
                elif (is_nl and not ms[i - 1].group('sent_end') and not ms[i + 1].group(0)[0] == DQUOTE and ms[i + 1].group('punct')) \
                        or ((is_end or is_p or is_ws or is_p) and s == ms[i + 1].group(0)) \
                        or (is_ws and (ms[i + 1].group('sent_end') or ms[i + 1].group('nl'))):
                    continue
                else:
                    TOKENS.append(s)
            TOKENS.append(ms[-1].group(0))
            if save:
                with lzma.open(cache_file, mode='wb') as f:
                    start = time()
                    log.debug('[caching] generated tokens')
                    f.write(pickle.dumps(TOKENS, protocol=-1))
                    log.debug(f'[finished] caching generated tokens (took {time() - start:4.2f}s)')
            log.info(f'[finished] generating tokens (took {time() - start:4.2f}s, {len(TOKENS)} tokens, size: {getsizeof(TOKENS) / 1e6:4.2f}MB)')
            return TOKENS


def get_counts(save=True, force=False) -> Counter:
    global COUNTS
    cache_file = root_path('cache', 'word_counts')
    with COUNTS_LK_R:
        if not force:
            if COUNTS is not None:
                log.debug('[cache hit] got word counts from cache')
                return COUNTS
            elif isfile(cache_file):
                log.debug(f'[cache hit] found word counts in file, loading ...')
                start = time()
                with open(cache_file, mode='rb') as f:
                    COUNTS = pickle.load(f)
                log.info(f'[cache hit] loaded word counts from file (took {time() - start:4.2f}s, size: {getsizeof(COUNTS) / 1e6:4.2f})')
                return COUNTS
        with COUNTS_LK_W:
            log.debug('[generating] word counts')
            start = time()
            COUNTS = Counter(get_tokens(save=save, force=force))
            if save:
                with open(cache_file, mode='wb') as f:
                    start = time()
                    log.debug(f'[caching] generated word counts')
                    pickle.dump(COUNTS, f)
                    log.debug(f'[finished] caching generated word counts (took {time() - start:4.2f}s)')
            log.info(f'[finished] generating word counts (took {time() - start:4.2f}s, size: {getsizeof(COUNTS) / 1e6:4.2f}MB)')
            return COUNTS


def get_ps(save=True, force=False) -> Dict[bytes, float]:
    global PS
    cache_file = root_path('cache', 'word_ps')
    with PS_LK_R:
        if not force:
            if PS is not None:
                log.info('[cache hit] got word probabilities from cache')
                return PS
            elif isfile(cache_file):
                log.debug(f'[cache hit] found word probabilities in file, loading ...')
                start = time()
                with open(cache_file, mode='rb') as f:
                    PS = pickle.load(f)
                log.info(f'[cache hit] loaded word probabilities from file (took {time() - start:4.2f}s, size: {getsizeof(PS) / 1e6:4.2f}MB)')
                return PS
        with PS_LK_W:
            start = time()
            log.debug('[generating] word probabilities')
            PS = dict(get_counts(force=force, save=save))
            no_tokens: int = sum(PS.values())
            for token in PS:
                PS[token] /= no_tokens
            if save:
                with open(cache_file, mode='wb') as f:
                    start = time()
                    log.debug(f'[caching] generated word probabilities')
                    pickle.dump(PS, f)
                    log.debug(f'[finished] caching generated word probabilities (took {time() - start:4.2f}s)')
            log.info(f'[finished] generating word probabilities (took {time() - start:4.2f}s, size: {getsizeof(PS) / 1e6:4.2f}MB)')
            return PS


def get_ngram_ps(n=2, save=True, force=False) -> Dict[Tuple, Dict[bytes, float]]:
    assert n >= 1, f'ngram len must be >= 1 but got n = {n}'
    assert n <= 20, f'ngram len must be <= 20 but got n = {n}'
    global NGRAM_PS
    cache_file = root_path('cache', f'{n}gram-ps')
    with NGRAM_PS_LKS_R[n]:
        if not force:
            if NGRAM_PS[n] is not None:
                log.info(f'[cache hit] got {n}gram probabilities from cache')
                return NGRAM_PS[n]
            elif isfile(cache_file):
                log.debug(f'[cache hit] found word probabilities in file, loading ...')
                start = time()
                with lzma.open(cache_file) as f:
                    NGRAM_PS[n] = pickle.loads(f.read())
                log.info(f'[cache hit] loaded {n}gram probabilities from file (took {time() - start:4.2f}s, size: {getsizeof(NGRAM_PS[n]) / 1e6:4.2f}MB)')
                return NGRAM_PS[n]
        with NGRAM_PS_LKS_W[n]:
            start = time()
            log.debug(f'[generating] {n}gram probabilities (this might take a couple of minutes)')

            tokens: List[bytes] = get_tokens(save=save, force=force)

            NGRAM_PS[n] = dict()

            for i in range(len(tokens) - n - 1):
                words_before: Tuple = tuple(tokens[i:i + n])
                next_word: bytes = tokens[i + n]
                if words_before not in NGRAM_PS[n]:
                    NGRAM_PS[n][words_before] = {next_word: 1}
                elif next_word in NGRAM_PS[n][words_before]:
                    NGRAM_PS[n][words_before][next_word] += 1
                else:
                    NGRAM_PS[n][words_before][next_word] = 1

            for ngram in NGRAM_PS[n]:
                total = 0
                for count in NGRAM_PS[n][ngram].values():
                    total += count
                if total > 0:
                    for next_word in NGRAM_PS[n][ngram]:
                        NGRAM_PS[n][ngram][next_word] /= total

            if save:
                with lzma.open(cache_file, mode='wb') as f:
                    start = time()
                    log.debug(f'[caching] generated {n}gram probabilities')
                    f.write(pickle.dumps(NGRAM_PS[n], protocol=-1))
                    log.debug(f'[finished] caching generated {n}grams (took {time() - start:4.2f}s)')

            log.info(f'[finished] generating {n}gram probabilities (took {time() - start:4.2f}s, size is {getsizeof(NGRAM_PS[n]) / 1e6:4.2f}MB)')
            return NGRAM_PS[n]


def generate(txt=b'That day', n=6, max_avg_txt_len=(1000 * 5), show_metrics=True, save=True, force=False) -> str:
    start = time()
    tokens: List[bytes] = [m.group(0) for m in tokenize(txt)]
    no_tokens = len(tokens)
    succ = [0 for _ in range(n + 1)]
    ps: Dict[bytes, float] = get_ps(force=force, save=save)
    unique_tokens: ndarray = np.array(list(ps.keys()))
    unique_tokens_ps: ndarray = np.array(list(ps.values()))

    with ThreadPool(max_workers=NO_CPUS, thread_name_prefix='markov/w') as pool:
        ps_ngrams: List[Dict[Tuple, Dict[bytes, float]]] = [
            task.result() for task in [
                pool.submit(fn=get_ngram_ps, n=i, force=force, save=save)
                for i in range(n, 0, -1)]]

    # token generation
    while no_tokens * AVG_TOKEN_LEN < max_avg_txt_len:
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
