# Standard Library
import logging
import pickle
import re
from collections import Counter
from os import listdir, makedirs
from os.path import abspath, dirname, isdir, isfile, join, relpath
from pathlib import Path
from re import IGNORECASE, MULTILINE
from sys import getsizeof
from time import time
from typing import Dict, Iterable, List, Match, Optional, Tuple

# 3rd Party
from numpy.random import choice

log = logging.getLogger()

ROOT: str = dirname(abspath(__file__))
AVG_TOKEN_LEN = 5

SPACE: int = ord(b' ')
COMMA: int = ord(b',')
COLON: int = ord(b':')
DOT: int = ord(b'.')
E_MARK: int = ord(b'!')
Q_MARK: int = ord(b'?')
SEMICOLON: int = ord(b';')
DQUOTE: int = ord(b'"')

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

CLEAN_REGEX = re.compile(
    rb'^\s*([-*]+\s*)?(chapter|\*|note|volume|section|part|[IVX]+|harry\s+potter|by\s+|(the\s+)?end)[^\n\r]*$|\r+',
    MULTILINE | IGNORECASE)

# processing
NEEDLESS_WRAP = re.compile(rb'([^\n])\n([^\n])')
TOO_MANY_NL = re.compile(rb'\n{3,}')
TOO_MANY_DASHES = re.compile(rb'(-\s*){3,}')
TOO_MANY_DOTS = re.compile(rb'(\.\s*){3,}')

IS_SENT_END = re.compile(rb'(!(!!)?|\?(\?\?)?|\.(\.\.)?\s*|\n+)$')

TEXT: Optional[bytes] = None
NGRAM_PS: List[Optional[Dict[Tuple, Dict[bytes, float]]]] = [None for _ in range(20)]
TOKENS: Optional[List[bytes]] = None
COUNTS: Optional[Counter] = None
PS: Optional[Dict[bytes, float]] = None


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


def decapitalize(txt: bytes) -> bytes:
    if len(txt) <= 1:
        return txt
    pos = 0
    while txt[pos] == SPACE:
        pos += 1
    # is uppercase
    if 65 <= txt[pos] <= 90:
        return txt[:pos] + chr(txt[pos] + 32).encode('ascii', 'ignore') + txt[pos + 1:]
    else:
        return txt


def tokenize(txt: bytes) -> Iterable[Match[bytes]]:
    return CHUNK_REGEX.finditer(txt)


def get_tokens(save=True, force=False) -> List[bytes]:
    global TOKENS
    cache_file = root_path('cache', 'tokens')
    if not force:
        if TOKENS is not None:
            log.debug('[cache hit] got tokens from cache')
            return TOKENS
        elif isfile(cache_file):
            with open(cache_file, mode='rb') as f:
                TOKENS = pickle.load(f)
            log.debug('[cache hit] got tokens from file')
            return TOKENS
    log.info('[generating] tokens')
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
        with open(cache_file, mode='wb') as f:
            log.info('[caching] generated tokens')
            pickle.dump(TOKENS, f)
    log.info(f'[finished] generating tokens (took {time() - start:4.2f} sec, {len(TOKENS)} tokens, size: {getsizeof(TOKENS) / 1e6:4.2f}MB)')
    return TOKENS


def get_counts(save=True, force=False) -> Counter:
    global COUNTS
    cache_file = root_path('cache', 'word_counts')
    if not force:
        if COUNTS is not None:
            log.info('[cache hit] got word counts from cache')
            return COUNTS
        elif isfile(cache_file):
            with open(cache_file, mode='rb') as f:
                COUNTS = pickle.load(f)
            log.info('[cache hit] got word counts from file')
            return COUNTS
    log.info('[generating] word counts')
    start = time()
    COUNTS = Counter(get_tokens(save=save, force=force))
    if save:
        with open(cache_file, mode='wb') as f:
            log.info(f'[caching] generated word counts')
            pickle.dump(COUNTS, f)
    log.info(f'[finished] generating word counts (took {time() - start:4.2f} sec, size: {getsizeof(COUNTS) / 1e6:4.2f}MB)')
    return COUNTS


def get_ps(save=True, force=False) -> Dict[bytes, float]:
    global PS
    cache_file = root_path('cache', 'word_ps')
    if not force:
        if PS is not None:
            log.info('[cache hit] got word probabilities from cache')
            return PS
        elif isfile(cache_file):
            with open(cache_file, mode='rb') as f:
                PS = pickle.load(f)
            log.info('[cache hit] got word probabilities from file')
            return PS
    start = time()
    log.info('[generating] word probabilities')
    PS = dict(get_counts(force=force, save=save))
    no_tokens: int = sum(PS.values())
    for token in PS:
        PS[token] /= no_tokens
    if save:
        with open(cache_file, mode='wb') as f:
            log.info(f'[caching] generated word probabilities')
            pickle.dump(PS, f)
    log.info(f'[finished] generating word probabilities (took {time() - start:4.2f} sec, size: {getsizeof(PS) / 1e6:4.2f}MB)')
    return PS


def get_ngram_ps(n=2, save=True, force=False) -> Dict[Tuple, Dict[bytes, float]]:
    global NGRAM_PS
    cache_file = root_path('cache', f'{n}gram-ps')
    assert n >= 1, f'ngram len must be >= 1 but got n = {n}'
    if not force:
        if NGRAM_PS[n] is not None:
            log.info(f'[cache hit] got {n}gram probabilities from cache')
            return NGRAM_PS[n]
        elif isfile(cache_file):
            with open(cache_file, mode='rb') as f:
                NGRAM_PS[n] = pickle.load(f)
            log.info(f'[cache hit] got {n}gram probabilities from file')
            return NGRAM_PS[n]
    start = time()
    log.info(f'[generating] {n}gram probabilities (this might take a couple of minutes)')

    tokens: List[bytes] = get_tokens(save=save, force=force)

    NGRAM_PS[n] = dict()

    for i in range(len(tokens) - n - 1):
        for m in range(1, n + 1):
            words_before: Tuple = tuple(tokens[i:i + m])
            next_word: bytes = tokens[i + m]
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
        with open(cache_file, mode='wb') as f:
            log.info(f'[caching] generated {n}gram probabilities')
            pickle.dump(NGRAM_PS[n], f)
    log.info(f'[finished] generating {n}gram probabilities (took {time() - start:4.2f} sec, size is {getsizeof(NGRAM_PS[n]) / 1e6:4.2f}MB)')
    return NGRAM_PS[n]


def root_path(*parts, mkparent=True, mkdir=False, mkfile=False) -> str:
    p: str = join(ROOT, *parts)
    if mkparent and not isdir(dirname(p)):
        makedirs(dirname(p))
    if mkdir and not isdir(p):
        makedirs(p)
    elif mkfile and not isfile(p):
        Path(p).touch()
    return p


def get_text(files=None) -> bytes:
    global TEXT
    if TEXT is not None:
        log.debug('[cache hit] got text from cache')
        return TEXT
    start = time()
    if files is None:
        files = [join(ROOT, 'data', fname) for fname in listdir(join(ROOT, 'data')) if fname.endswith('.txt')]
    log.info(f'[loading] text from {len(files)} files')
    texts: List[bytes] = []
    for path in files:
        start_file = time()
        with open(path, mode='rb') as f:
            log.debug(f'[loading] text from file {relpath(path)}')
            try:
                texts.append(f.read())
            except Exception as e:
                log.warning(str(e))
        log.debug(f'[finished] reading from {relpath(path)} (read {getsizeof(texts[-1]) / 1e6:4.2f}MB in {time() - start_file:4.2f}s)')
    TEXT = b'\n\n'.join(texts)
    TEXT = CLEAN_REGEX.sub(b'', TEXT)
    TEXT = NEEDLESS_WRAP.sub(rb'\1 \2', TEXT)
    TEXT = TOO_MANY_NL.sub(b'\n\n', TEXT)
    TEXT = TOO_MANY_DOTS.sub(rb'...', TEXT)
    TEXT = TOO_MANY_DASHES.sub(rb'--', TEXT)
    log.info(f'[finished] reading (read {getsizeof(TEXT) / 1e6:4.2f}MB in {time() - start:4.2f}s)')
    return TEXT


def generate(txt=b'That day', n=6, max_avg_txt_len=(1000 * 5), show_metrics=True, save=True, force=False) -> str:
    start = time()
    tokens: List[bytes] = [m.group(0) for m in tokenize(txt)]
    succ = [0 for _ in range(n + 1)]
    ps: Dict[bytes, float] = get_ps(force=force, save=save)
    unique_tokens: List[bytes] = list(ps.keys())
    unique_tokens_ps: List[float] = list(ps.values())
    ps_ngrams = get_ngram_ps(n=n, force=force, save=save)

    # token generation
    while len(tokens) * AVG_TOKEN_LEN < max_avg_txt_len:
        found = False
        for m in range(n, 0, -1):
            ngram = tuple(tokens[-m:])
            maybe_ps: Optional[Dict[bytes, float]] = ps_ngrams.get(ngram, None)
            if maybe_ps and len(maybe_ps) > 1:
                succ[m] += 1
                tokens.append(choice(a=list(maybe_ps.keys()), p=list(maybe_ps.values())))
                found = True
                break
        if not found:
            succ[0] += 1
            tokens.append(choice(a=unique_tokens, p=unique_tokens_ps))

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

    log.info(f'[finished] generating text (took {time() - start:4.2f} sec)')

    # text (outcome)
    return (b''.join(tokens) + b'.\n\nTHE END.').decode('ascii', 'ignore')
