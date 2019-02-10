import logging
import re
from sys import getsizeof
from collections import Counter
from os import listdir, makedirs
from os.path import dirname, abspath, join, isdir, isfile, basename
from pathlib import Path
from re import MULTILINE, IGNORECASE
from time import time
from typing import List, Optional, Tuple, Dict, Match, Iterable

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
# DQUOTE: int = ord(b'"')

PUNCT_REGEX = rb'(?P<punct>-{1,2}|[:;,"])'
NL_REGEX = rb'(?P<nl>(\n\r?|\r\n?)+)'
SENT_END_REGEX = rb'(?P<sent_end>(!(!!)?|\?(\?\?)?|\.(\.\.)?))'
WS_REGEX = rb'(?P<ws> )'
WORD_REGEX = rb"(?P<word>([A-Z]?[a-z]+|[A-Z][a-z]*)(-[A-Za-z]+)*('[a-z]{0,7})?)"
DATE_REGEX = rb"(?P<date>(([1-9]\d*)(th|[nr]d)|19\d{2}|20\d{2}))"
TIME_REGEX = rb"(?P<time>\d+((:\d{2}){1,2}|(\.\d{2}){1,2}))"
CHUNK_REGEX = re.compile(rb'(?P<token>' + rb'|'.join([
    WS_REGEX,
    SENT_END_REGEX,
    PUNCT_REGEX,
    NL_REGEX,
    WORD_REGEX,
    DATE_REGEX,
    TIME_REGEX,
]) + rb')')

CLEAN_REGEX = re.compile(
    rb'^\s*(-+\s*)?(chapter|\*|note|volume|section|part|[IVX]+|harry\s+potter|by\s+|(the\s+)?end)[^\n\r]*$|\r+',
    MULTILINE | IGNORECASE)

# processing
NEEDLESS_WRAP = re.compile(rb'([^\n])\n([^\n])')
TOO_MANY_NL = re.compile(rb'\n{3,}')
TOO_MANY_DASHES = re.compile(rb'(- *){3,}')
TOO_MANY_DOTS = re.compile(rb'(\. *){3,}')

IS_SENT_END = re.compile(rb'(!(!!)?|\?(\?\?)?|\.(\.\.)?\s*|\n+)$')

TEXT: Optional[bytes] = None
NGRAM_PS: Optional[Dict[Tuple, Dict[bytes, float]]] = None
TOKENS: Optional[List[bytes]] = None
COUNTS: Optional[Counter] = None
PS: Optional[Dict[bytes, float]] = None


def capitalize(txt: bytes) -> bytes:
    if len(txt) <= 1: return txt
    pos = 0
    while txt[pos] == SPACE:
        pos += 1
    # is lowercase
    if 97 <= txt[pos] <= 122:
        return txt[:pos] + chr(txt[pos] - 32).encode('ascii', 'ignore') + txt[pos + 1:]
    else:
        return txt


def decapitalize(txt: bytes) -> bytes:
    if len(txt) <= 1: return txt
    pos = 0
    while txt[pos] == SPACE:
        pos += 1
    # is uppercase
    if 65 <= txt[pos] <= 90:
        return txt[:pos] + chr(txt[pos] + 32).encode('ascii', 'ignore') + txt[pos + 1:]
    else:
        return txt


def postprocess(tokens: List[bytes]) -> None:
    start = time()
    log.info('starting postprocessing')
    for i in range(1, len(tokens) - 1):
        if len(tokens[i]) > 0 and len(tokens[i + 1]) > 0:
            # word. word2 => word. Word2
            if len(tokens[i + 1].lstrip()) > 1 and IS_SENT_END.search(tokens[i]):
                tokens[i + 1] = capitalize(tokens[i + 1])
            # lack of space after punct
            current_last = tokens[i][-1]
            next_first = tokens[i + 1][0]
            if (current_last == DOT or current_last == COMMA or current_last == SEMICOLON or current_last == COLON or current_last == E_MARK or current_last == Q_MARK) and next_first != SPACE:
                tokens[i] += b' '
            # more than 1 consecutive spaces
            elif current_last == SPACE and next_first == SPACE:
                tokens[i + 1] = tokens[i + 1].lstrip()
            # two consecutive words (lack of space to separate them)
            elif 60 <= current_last <= 90 or 97 <= current_last <= 122 and \
                    (60 <= next_first <= 90 or 97 <= next_first <= 122):
                tokens[i] += b' '
            elif current_last == b'"' and next_first == b'"':
                tokens[i] = tokens[i][:-1]
            # elif current_last == DQUOTE and next_first == b'"':
            #     tokens[i] = tokens[i][:-1]
    log.info(f'finished postprocessing (took {time() - start:4.2f} sec)')


def tokenize(txt: bytes) -> Iterable[Match[bytes]]:
    return CHUNK_REGEX.finditer(txt)


def get_tokens() -> List[bytes]:
    global TOKENS
    if TOKENS is not None:
        log.info('got tokens from cache')
        return TOKENS
    log.info('generating tokens')
    start = time()
    ms: List[Match[bytes]] = list(tokenize(get_text()))
    TOKENS = [ms[0].group(0)]
    for i in range(1, len(ms) - 1):
        # if ms[i].group('word') and ms[i -1 ].group('') and (97 <= TOKENS[-1][0] <= 122):
        #     TOKENS[-1] = capitalize(TOKENS[-1])
        if (ms[i].group('word') and ms[i + 1].group('word')) or (
                ms[i].group('sent_end') and not (ms[i + 1].group('ws') or ms[i + 1].group('nl'))
        ) or (ms[i].group('punct') and not (ms[i + 1].group('ws')) or ms[i + 1].group('nl')):
            TOKENS.append(b' ')
        elif (ms[i].group('ws') and (ms[i + 1].group('ws') or ms[i + 1].group('sent_end'))) or (
                ms[i].group('punct') and ms[i + 1].group('punct') and ms[i].group('punct') == ms[i + 1].group('punct')):
            continue
        TOKENS.append(ms[i].group(0))
    TOKENS.append(ms[-1].group(0))
    log.info(f'finished generating tokens (took {time() - start:4.2f} sec, {len(TOKENS)} tokens, object size: {getsizeof(TOKENS) / 1e6:4.2f}MB)')
    return TOKENS


def get_counts() -> Counter:
    global COUNTS
    if COUNTS is not None:
        log.info('got word counts from cache')
        return COUNTS
    log.info('generating word counts')
    start = time()
    COUNTS = Counter(get_tokens())
    log.info(f'finished generating word counts (took {time() - start:4.2f} sec, object size: {getsizeof(COUNTS) / 1e6:4.2f}MB)')
    return COUNTS


def get_ps() -> Dict[bytes, float]:
    global PS
    if PS is not None:
        log.info('got word probabilities from cache')
        return PS
    start = time()
    log.info('generating word probabilities')
    PS = dict(get_counts())
    no_tokens: int = sum(PS.values())
    for token in PS:
        PS[token] /= no_tokens
    log.info(f'finished generating word probabilities (took {time() - start:4.2f} sec, object size: {getsizeof(PS) / 1e6:4.2f}MB)')
    return PS


def get_ngram_ps(n=2) -> Dict[Tuple, Dict[bytes, float]]:
    global NGRAM_PS

    assert n >= 1, f'ngram len must be >= 1 but got n = {n}'
    if NGRAM_PS is not None:
        log.info('got ngram probabilities from cache')
        return NGRAM_PS
    start = time()
    log.info('generating ngram probabilities (this might take a couple of minutes)')

    tokens: List[bytes] = get_tokens()

    NGRAM_PS = dict()

    for i in range(len(tokens) - n - 1):
        for m in range(1, n + 1):
            words_before: Tuple = tuple(tokens[i:i + m])
            next_word: bytes = tokens[i + m]
            if words_before not in NGRAM_PS:
                NGRAM_PS[words_before] = {next_word: 1}
            elif next_word in NGRAM_PS[words_before]:
                NGRAM_PS[words_before][next_word] += 1
            else:
                NGRAM_PS[words_before][next_word] = 1

    for ngram in NGRAM_PS:
        total = 0
        for count in NGRAM_PS[ngram].values():
            total += count
        if total > 0:
            for next_word in NGRAM_PS[ngram]:
                NGRAM_PS[ngram][next_word] /= total

    log.info(f'finished generating ngram probabilities (took {time() - start:4.2f} sec, size is {getsizeof(NGRAM_PS) / 1e6:4.2f}MB)')
    return NGRAM_PS


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
        log.info('got text from cache')
        return TEXT
    start = time()
    if files is None:
        files = [join(ROOT, 'data', fname) for fname in listdir(join(ROOT, 'data')) if fname.endswith('.txt')]
    log.info(f'loading text from {len(files)} files {", ".join((basename(p) for p in files))}')
    texts: List[bytes] = []
    for path in files:
        start_file = time()
        with open(path, mode='rb') as f:
            log.info(f'loading text from file "{path}"')
            try:
                texts.append(f.read())
            except Exception as e:
                log.warning(str(e))
        log.info(f'finished loading text from "{path}" (took {time() - start_file:4.2f} sec, read {len(texts[-1])} bytes, object size: {getsizeof(texts[-1]) / 1e6:4.2f}MB)')
    TEXT = b'\n\n'.join(texts)
    TEXT = CLEAN_REGEX.sub(b'', TEXT)
    TEXT = NEEDLESS_WRAP.sub(rb'\1 \2', TEXT)
    TEXT = TOO_MANY_NL.sub(b'\n\n', TEXT)
    TEXT = TOO_MANY_DOTS.sub(rb'...', TEXT)
    TEXT = TOO_MANY_DASHES.sub(rb'--', TEXT)
    log.info(f'finished loading text (took {time() - start:4.2f} sec, read {len(TEXT)} bytes, object size: {getsizeof(TEXT) / 1e6:4.2f}MB)')
    return TEXT


def generate(txt=b'That day', n=6, max_avg_txt_len=(10000 * 8)) -> str:
    start = time()
    tokens: List[bytes] = [m.group(0) for m in tokenize(txt)]
    succ = [0 for _ in range(n + 1)]
    ps: Dict[bytes, float] = get_ps()
    unique_tokens: List[bytes] = list(ps.keys())
    unique_tokens_ps: List[float] = list(ps.values())
    ps_ngrams = get_ngram_ps(n=n)

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

    # post-processing
    # postprocess(tokens=tokens)

    # metrics
    log.info('-' * (40 + 12 + 2 + 2))
    log.info('%2s %12s %s' % ('NO', 'PROBABILITY', 'NO EXAMPLES'))
    log.info('%2s %12s %s' % ('-' * 2, '-' * 12, '-' * 40))
    no_gen_tokens: int = sum(succ)
    for i in range(n, -1, -1):
        log.info('%2d %12.10f (%d tokens)' % (i, succ[i] / no_gen_tokens, succ[i]))

    log.info(f'finished generating text (took {time() - start:4.2f} sec)')

    # text (outcome)
    return (b''.join(tokens) + b'.\n\nTHE END.').decode('ascii', 'ignore')
