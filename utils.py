import logging
import lzma
import pickle
import re
from collections import Counter
from os import makedirs, listdir
from os.path import dirname, abspath, isfile, isdir, join, relpath
from pathlib import Path
from re import MULTILINE, IGNORECASE
from sys import getsizeof
from threading import Lock, Semaphore
from time import time
from typing import List, Match, Iterable, Dict, Tuple, Any

import nltk
import numpy as np
from numpy import ndarray
from numpy.random import choice

AVG_CHUNK_LEN = 5
NO_CPUS = 4

DQUOTE: int = ord(b'"')
SPACE: int = ord(b' ')

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

log = logging.getLogger()

ROOT: str = dirname(abspath(__file__))

CLEAN_REGEX = re.compile(
    rb'^\s*([-*]+\s*)?(chapter|\*|note|volume|section|part|[IVX]+|harry\s+potter|by\s+|(the\s+)?end)[^\n\r]*$|\r+',
    MULTILINE | IGNORECASE)

# processing
NEEDLESS_WRAP = re.compile(rb'([^\n])\n([^\n])')
TOO_MANY_NL = re.compile(rb'\n{3,}')
TOO_MANY_DASHES = re.compile(rb'(-\s*){3,}')
TOO_MANY_DOTS = re.compile(rb'(\.\s*){3,}')


def cached(what: str, load=True, save=True):
    def outer(fn):
        def inner():
            if globals().get(what + '_LK_R', None) is None:
                globals()[what + '_LK_R'] = Semaphore(NO_CPUS)
            with globals()[what + '_LK_R']:
                if globals().get(what, None) is not None:
                    log.debug(f'[cache hit] found {what}')
                    return globals()[what]
                path = root_path('cache', what)
                if load and isfile(path):
                    if globals().get(what + '_LK_W', None) is None:
                        globals()[what + '_LK_W'] = Lock()
                    with globals()[what + '_LK_W']:
                        log.debug(f'[cache hit] found {what} in file, loading ...')
                        start = time()
                        with lzma.open(path) as f:
                            globals()[what] = pickle.loads(f.read())
                        log.info(f'[cache hit] loaded {what} from file (took {time() - start:4.2f}s, size: {getsizeof(globals()[what]) / 1e6:4.2f}MB)')
                        return globals()[what]
                else:
                    log.info(f'[cache miss, generating] {what}')
                    start = time()
                    globals()[what] = fn()
                    log.debug(f'[finished] generating {what} (took {time() - start:4.2f}s, size: {getsizeof(globals()[what]) / 1e6:4.2f}MB)')
                    if save:
                        log.debug(f'[caching] {what}')
                        start = time()
                        with lzma.open(path, mode='wb') as f:
                            f.write(pickle.dumps(globals()[what]))
                        log.debug(f'[finished] caching {what} (took {time() - start:4.2f}s)')
                    return globals()[what]
        return inner
    return outer


def tokenize(txt: bytes) -> Iterable[Match[bytes]]:
    return CHUNK_REGEX.finditer(txt)


def root_path(*parts, mkparent=True, mkdir=False, mkfile=False) -> str:
    p: str = join(ROOT, *parts)
    if mkparent and not isdir(dirname(p)):
        makedirs(dirname(p))
    if mkdir and not isdir(p):
        makedirs(p)
    elif mkfile and not isfile(p):
        Path(p).touch()
    return p


# noinspection PyDefaultArgument
@cached('TEXT')
def get_text(files=[root_path('data', fname) for fname in listdir(root_path('data')) if fname.endswith('.txt')]) -> bytes:
    log.debug(f'[loading] text from {len(files)} files')
    texts: bytearray = bytearray()
    for path in files:
        start_file = time()
        with open(path, mode='rb') as f:
            log.debug(f'[loading] text from file {relpath(path)}')
            try:
                texts.extend(f.read())
            except Exception as e:
                log.warning(str(e))
        log.debug(f'[finished] reading from {relpath(path)} (read {getsizeof(texts[-1]) / 1e6:4.2f}MB in {time() - start_file:4.2f}s)')
    texts.extend(b'\n\n')
    return b'\n'.join(texts)


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


@cached('CHUNKS')
def get_chunks(save=True, force=False) -> List[bytes]:
    ms: List[Match[bytes]] = list(tokenize(get_text()))
    chunks = [ms[0].group(0), ms[0].group(0)]
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
        if is_w and ms[i - 2].group('sent_end') and (97 <= chunks[-2][0] <= 122):
            chunks[-2] = capitalize(chunks[-2])
        if is_w and ms[i - 1].group('word'):
            chunks.append(b'. ' if is_cap else b' ')
        if is_w and is_cap and (ms[i - 2].group('sent_end') or ms[i - 1].group('nl')):
            chunks.append(capitalize(s))
            continue
        elif (is_w and ms[i + 1].group('word')) \
                or (is_end and not (ms[i + 1].group('ws') or ms[i + 1].group('nl'))) \
                or (is_p and not (is_q or ms[i + 1].group('ws') or ms[i - 1].group('nl'))):
            chunks.append(s)
            chunks.append(b' ')
            continue
        elif (is_nl and not ms[i - 1].group('sent_end') and not ms[i + 1].group(0)[0] == DQUOTE and ms[i + 1].group('punct')) \
                or ((is_end or is_p or is_ws or is_p) and s == ms[i + 1].group(0)) \
                or (is_ws and (ms[i + 1].group('sent_end') or ms[i + 1].group('nl'))):
            continue
        else:
            chunks.append(s)
    chunks.append(ms[-1].group(0))
    return chunks


def get_chunks_ps(n=2, save=True, force=False) -> Dict[Tuple, Dict[bytes, float]]:
    assert n >= 1, f'ngram len must be >= 1 but got n = {n}'
    assert n <= 20, f'ngram len must be <= 20 but got n = {n}'

    @cached(f'{n}CHUNKS_PS')
    def ps():
        tokens: List[bytes] = get_chunks(save=save, force=force)
        ps = dict()
        for i in range(len(tokens) - n - 1):
            words_before: Tuple = tuple(tokens[i:i + n])
            next_word: bytes = tokens[i + n]
            if words_before not in ps:
                ps[words_before] = {next_word: 1}
            elif next_word in ps[n][words_before]:
                ps[words_before][next_word] += 1
            else:
                ps[words_before][next_word] = 1

        for ngram in ps:
            total = 0
            for count in ps[ngram].values():
                total += count
            if total > 0:
                for next_word in ps[ngram]:
                    ps[ngram][next_word] /= total

    return ps()


@cached('CHAR_COUNTS')
def get_counts(save=True, force=False) -> ndarray:
    bag = Counter(get_text())
    counts = np.array([0 for _ in range(128)], dtype='int8')
    for k, v in bag.items():
        if k < 128:
            counts[k] = v
    return counts


@cached('CHAR_PS')
def get_char_ps(save=True, force=False) -> ndarray:
    counts: ndarray = get_counts(force=force, save=save)
    ps: ndarray = np.zeros(shape=(counts.size,), dtype='float64')
    no_chars: int = sum(ps)
    for c in range(128):
        ps[c] = counts[c] / no_chars
    return ps


def get_nchar_ps(n=2, save=True, force=False) -> Dict[bytes, Dict[int, float]]:
    assert n >= 1, f'nchar len must be >= 1 but got n = {n}'
    assert n <= 20, f'nchar len must be <= 20 but got n = {n}'

    @cached(f'{n}CHAR_PS')
    def inner():
        txt: bytes = get_text()
        ps = dict()

        for i in range(len(txt) - n - 1):
            chars_before = txt[i:i + n]
            char_after: int = txt[i + n]
            if chars_before not in ps:
                ps[chars_before] = {char_after: 1}
            elif char_after in ps[chars_before]:
                ps[chars_before][char_after] += 1
            else:
                ps[chars_before][char_after] = 1

        for nchar in ps:
            total = 0
            for count in ps[nchar].values():
                total += count
            if total > 0:
                for char_after in ps[nchar]:
                    ps[nchar][char_after] /= total

        return ps[n]

    return inner()


@cached('WORDS')
def get_words() -> List[str]:
    return nltk.word_tokenize(get_text().decode('ascii', 'ignore'))


@cached('SENTS')
def get_sents() -> List[str]:
    return nltk.sent_tokenize(get_text().decode('ascii', 'ignore'))


@cached('TAGGED_WORDS')
def get_tagged_words() -> List[Tuple[str, str]]:
    return nltk.pos_tag(get_words())


@cached('SENT_STRUCT_PS')
def get_sents_structs_ps() -> Tuple[List[Tuple], ndarray]:
    global SENT_STRUCTS, SENT_STRUCTS_PS
    if SENT_STRUCTS_PS is not None:
        return SENT_STRUCTS, SENT_STRUCTS_PS
    bag: Dict[Tuple, Any] = dict()
    buf: List[str] = []
    for word, tag in get_tagged_words():
        buf.append(tag)
        if tag == '.':
            s: Tuple = tuple(buf)
            bag[s] = bag.get(s, 0) + 1
            buf = []
    total: int = sum(bag.values())
    for s in bag:
        bag[s] /= total
    SENT_STRUCTS_PS = np.array(list(bag.values()))
    SENT_STRUCTS = list(bag.keys())
    return SENT_STRUCTS, SENT_STRUCTS_PS


def rand_sent_struct() -> List[str]:
    structs, ps = get_sents_structs_ps()
    return choice(a=structs, p=ps)


@cached('TAG_PS')
def get_tags_ps() -> Dict[str, Dict[str, float]]:
    global TAG_PS
    if TAG_PS is not None:
        return TAG_PS
    TAG_PS = dict()
    for word, tag in get_tagged_words():
        if TAG_PS.get(tag, None) is None:
            TAG_PS[tag] = dict()
        TAG_PS[tag][word] = TAG_PS[tag].get(word, 0) + 1

    for tag in TAG_PS:
        total = sum(TAG_PS[tag].values())
        for word in TAG_PS[tag]:
            TAG_PS[tag][word] /= total
    return TAG_PS


def rand_word(tag: str) -> str:
    return choice(
        a=tuple(get_tags_ps()[tag].keys()),
        p=tuple(get_tags_ps()[tag].values()))


@cached('TOKEN_COUNTS')
def get_chunks_counts(save=True, force=False) -> Counter:
    return Counter(get_chunks(save=save, force=force))


@cached('CHUNKS_PS')
def get_chunks_ps(save=True, force=False) -> Dict[bytes, float]:
    ps = dict(get_chunks_counts(force=force, save=save))
    no_tokens: int = sum(ps.values())
    for token in ps:
        ps[token] /= no_tokens
    return ps
