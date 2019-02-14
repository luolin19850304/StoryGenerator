import logging
import lzma
import pickle
import re
from concurrent.futures import ThreadPoolExecutor as ThreadPool
from collections import Counter
from os import makedirs, listdir
from os.path import dirname, abspath, isfile, isdir, join, relpath
from pathlib import Path
from re import MULTILINE, IGNORECASE
from sys import getsizeof
from threading import Lock, Semaphore
from time import time
from typing import List, Match, Iterable, Dict, Tuple, Any, Optional, Generator, Union

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

# logging.basicConfig(format='%(levelname)s %(funcName)-13s %(lineno)3d %(message)s')

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


def cached(what: str, keep=True, load=True, save=True, archive=True):
    def outer(fn):
        def inner():
            lock_r = '{0}_LK_R'.format(what)
            if globals().get(lock_r, None) is None:
                globals()[lock_r] = Semaphore(NO_CPUS)
            with globals()[lock_r]:
                if globals().get(what, None) is not None:
                    log.debug(f'[cache hit] found {what}')
                    return globals()[what]
                path = root_path('cache', what)
                if load and isfile(path):
                    lock_w = '{0}_LK_W'.format(what)
                    if globals().get(lock_w, None) is None:
                        globals()[lock_w] = Lock()
                    with globals()[lock_w]:
                        log.debug(f'[cache hit] found {what} in file, loading ...')
                        start = time()
                        result = None
                        with (lzma.open if archive else open)(path, mode='rb') as f:
                            result = pickle.load(f)
                        if keep:
                            globals()[what] = result
                        log.info(f'[cache hit] loaded {what} from file (took {time() - start:4.2f}s, size: {getsizeof(result) / 1e6:4.2f}MB)')
                        return result
                else:
                    log.info(f'[cache miss, generating] {what}')
                    start = time()
                    result = fn()
                    if keep:
                        globals()[what] = result
                    log.debug(f'[finished] generating {what} (took {time() - start:4.2f}s, size: {getsizeof(result) / 1e6:4.2f}MB)')
                    if save:
                        log.debug(f'[caching] {what}')
                        start = time()
                        with (lzma.open if archive else open)(path, mode='wb') as f:
                            pickle.dump(result, f)
                        log.debug(f'[finished] caching {what} (took {time() - start:4.2f}s)')
                    return result
        return inner
    return outer


def chunk(txt: bytes) -> Iterable[Match[bytes]]:
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
@cached('TEXT', keep=True, load=False, save=False)
def get_text(files=[root_path('data', fname) for fname in listdir(root_path('data')) if fname.endswith('.txt')]) -> bytearray:
    log.debug(f'[loading] text from {len(files)} files')
    texts: bytearray = bytearray()
    lock = Lock()
    def read_file(path: str):
        start_file = time()
        fname = relpath(path)
        with open(path, mode='rb') as f:
            log.debug(f'[loading] text from file {fname}')
            try:
                txt = f.read()
                lock.acquire()
                texts.extend(txt)
                texts.extend(b'\n\n')
                lock.release()
            except Exception as e:
                log.warning(str(e))
        log.debug(f'[finished] reading from {fname} (read {getsizeof(texts[-1]) / 1e6:4.2f}MB in {time() - start_file:4.2f}s)')
    with ThreadPool(max_workers=NO_CPUS, thread_name_prefix='get_text') as pool:
        for task in [pool.submit(fn=read_file, path=p) for p in files]:
            task.result()
    return texts


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
def get_chunks() -> List[bytes]:
    ms: List[Match[bytes]] = list(chunk(get_text()))
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


def get_nchunks_ps(n=2) -> Dict[Tuple, Dict[bytes, float]]:
    assert 20 >= n >= 1, f'ngram len must be in [1, 20] but got n = {n}'

    @cached(f'{n}CHUNKS_PS')
    def inner():
        tokens: List[bytes] = get_chunks()
        ps: Dict[Tuple, Dict[bytes, float]] = dict()
        for i in range(len(tokens) - n - 1):
            words_before: Tuple = tuple(tokens[i:i + n])
            next_word: bytes = tokens[i + n]
            if words_before not in ps:
                ps[words_before] = {next_word: 1}
            else:
                ps[words_before][next_word] = \
                        ps[words_before].get(next_word, 0) + 1
        for ngram in ps:
            total = 0
            for count in ps[ngram].values():
                total += count
            if total > 0:
                for next_word in ps[ngram]:
                    ps[ngram][next_word] /= total
        return ps
    return inner()


@cached('CHAR_COUNTS')
def get_char_counts() -> ndarray:
    bag = Counter(get_text())
    counts: ndarray = np.arange(128, dtype='uint32')
    for k, v in bag.items():
        if k < 128:
            counts[k] = v
    return counts


@cached('CHAR_PS')
def get_char_ps() -> ndarray:
    counts: ndarray = get_char_counts()
    ps: ndarray = np.zeros(shape=(counts.size,), dtype='float64')
    no_chars: int = counts.sum()
    for c in range(128):
        ps[c] = counts[c] / no_chars
    return ps


def get_nchar_ps(n=2) -> Dict[bytes, Dict[int, float]]:
    assert 20 >= n >= 1, f'nchar len must be in [1, 20] but got n = {n}'

    @cached(f'{n}CHAR_PS')
    def inner():
        txt: bytes = bytes(get_text())
        ps: Dict[bytes, Dict[int, float]] = dict()

        for i in range(len(txt) - n - 1):
            chars_before: bytes = txt[i:i + n]
            char_after: int = txt[i + n]
            if chars_before not in ps:
                ps[chars_before] = {char_after: 1}
            else:
                ps[chars_before][char_after] = ps[chars_before].get(char_after, 0) + 1

        for nchar in ps:
            total = sum(ps[nchar].values())
            for char_after in ps[nchar]:
                ps[nchar][char_after] /= total

        return ps

    return inner()


@cached('WORDS')
def get_words() -> List[str]:
    return nltk.word_tokenize(get_text().decode('ascii', 'ignore'))

@cached('WORDSPUNCTS')
def get_wordpuncts() -> List[str]:
    return nltk.wordpunct_tokenize(get_text().decode('ascii', 'ignore'))


@cached('SENTS')
def get_sents() -> List[str]:
    return nltk.sent_tokenize(get_text().decode('ascii', 'ignore'))


@cached('TAGGED_WORDS')
def get_tagged_words() -> List[Tuple[str, str]]:
    return nltk.pos_tag(get_words())

# BUGGY NLTK
##  @cached('TAGGED_SENTS')
##  def get_tagged_sents() -> List[List[Tuple[str, str]]]:
    ##  return nltk.pos_tag_sents(
            ##  [nltk.wordpunct_tokenize(sent) for sent in get_sents()])


@cached('SENT_STRUCT_PS')
def get_sent_structs_ps() -> Tuple[List[Tuple], ndarray]:

    @cached('SENT_STRUCTS')
    def get_sent_structs() -> Generator[Tuple, None, None]:
        buf: List[str] = []
        for word, tag in get_tagged_words():
            buf.append(tag)
            if tag == '.':
                yield tuple(buf)
                buf = []

    bag = Counter(get_sent_structs())

    total: int = sum(bag.values())

    for s in bag:
        bag[s] /= total

    return list(bag.keys()), np.array(list(bag.values()), dtype='float64')


def rand_sent_struct() -> List[str]:
    structs, ps = get_sent_structs_ps()
    return choice(a=structs, p=ps)


@cached('TAG_PS')
def get_tag_ps() -> Dict[str, Dict[str, float]]:
    ps: Dict[str, Dict[str, Union[int, float]]] = dict()
    for word, tag in get_tagged_words():
        if ps.get(tag, None) is None:
            ps[tag] = dict()
        ps[tag][word] = ps[tag].get(word, 0) + 1

    for tag in ps:
        total = sum(ps[tag].values())
        for word in ps[tag]:
            ps[tag][word] /= total
    return ps


def rand_word(tag: Optional[str] = None) -> str:
    tags_ps = get_tag_ps()
    if tag is None:
        tag = choice(a=tuple(tags_ps.keys()), p=tuple(tags_ps.values()))
    return choice(
        a=tuple(tags_ps[tag].keys()),
        p=tuple(tags_ps[tag].values()))


@cached('CHUNK_COUNTS')
def get_chunk_counts() -> Counter:
    return Counter(get_chunks())


@cached('CHUNK_PS')
def get_chunk_ps() -> Dict[bytes, float]:
    ps = dict(get_chunk_counts())
    no_tokens: int = sum(ps.values())
    for token in ps:
        ps[token] /= no_tokens
    return ps
