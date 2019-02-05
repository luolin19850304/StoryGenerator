import re
from collections import Counter
from os import listdir, makedirs
from os.path import dirname, abspath, join, isdir, isfile
from pathlib import Path
from sys import stderr
from typing import List, Optional, Tuple, Dict

import numpy as np

ROOT: str = dirname(abspath(__file__))

CHUNK_REGEX = re.compile(
    rb"(([.!?]([.!?]{2})?|-{1,2}|[:;, \n\r])|(([A-Z]?[a-z]+|[A-Z][a-z]*)(-[a-z]+)*('[a-z]{,7})?))")

TEXT: Optional[bytes] = None
NGRAM_PS: Optional[List[Dict[int, float]]] = None
NGRAM_INDEX: Optional[np.ndarray] = None
NGRAM_INDEX_REV: Optional[Dict[Tuple, int]] = None
TOKENS: Optional[List[bytes]] = None
COUNTS: Optional[Counter] = None
PS: Optional[Dict[bytes, float]] = None
NGRAMS: List[Optional[List[Tuple]]] = [None for _ in range(15)]


def tokenize(txt: bytes) -> List[bytes]:
    return list((match[0] for match in CHUNK_REGEX.findall(txt)))


def get_tokens() -> List[bytes]:
    global TOKENS
    if TOKENS is not None:
        return TOKENS
    TOKENS = tokenize(get_text())
    return TOKENS


def get_counts() -> Counter:
    global COUNTS
    if COUNTS is not None:
        return COUNTS
    COUNTS = Counter(get_tokens())
    return COUNTS


def get_ngrams(n=2, max_ngrams=10) -> List[Tuple[bytes]]:
    assert n > 1, f'ngram len must be > 2 but got n = {n}'
    global NGRAMS
    if NGRAMS[n - 1] is not None:
        return NGRAMS[n - 1]
    NGRAMS = [[] for _ in range(max_ngrams)]
    tokens = get_tokens()
    for i in range(len(tokens) - n):
        NGRAMS[n - 1].append(tuple(tokens[i:i + n]))
    return NGRAMS[n - 1]


def get_indexes() -> Dict[bytes, float]:
    global PS
    if PS is not None:
        return PS
    PS = dict(get_counts())
    no_tokens: int = sum(PS.values())
    i = 0
    for token in PS:
        PS[token] /= no_tokens
    return PS


def get_indexes_ngrams(n=2) -> Tuple[np.ndarray, Dict[Tuple, int], List[Dict[bytes, float]]]:
    global NGRAM_INDEX, NGRAM_INDEX_REV, NGRAM_PS

    assert n >= 1, f'ngram len must be >= 1 but got n = {n}'
    if NGRAM_INDEX is not None and NGRAM_INDEX_REV is not None and NGRAM_PS is not None:
        return NGRAM_INDEX, NGRAM_INDEX_REV, NGRAM_PS

    tokens: List[bytes] = get_tokens()

    index: List[Tuple] = []
    index_rev: Dict[Tuple, int] = dict()
    ps: List[Dict[bytes, float]] = []
    idx = 0

    for i in range(len(tokens) - n):
        for m in range(1, n + 1):
            token_group: Tuple = tuple(tokens[i:i + m])
            if token_group not in index_rev:
                index.append(token_group)
                index_rev[token_group] = idx
                idx += 1
                ps.append(dict())

    for i in range(len(tokens) - n - 1):
        for m in range(1, n + 1):
            words_before: Tuple = tuple(tokens[i:i + m])
            words_before_idx: int = index_rev[words_before]
            next_word: bytes = tokens[i + m]
            if next_word in ps[words_before_idx]:
                ps[words_before_idx][next_word] += 1
            else:
                ps[words_before_idx][next_word] = 1

    for words_before_idx in range(len(ps)):
        total = 0
        for count in ps[words_before_idx].values():
            total += count
        for next_word in ps[words_before_idx]:
            ps[words_before_idx][next_word] /= total

    NGRAM_INDEX = np.array(index)
    NGRAM_INDEX_REV = index_rev
    NGRAM_PS = ps
    return NGRAM_INDEX, NGRAM_INDEX_REV, NGRAM_PS


def root_path(*parts, mkparent=True, mkdir=False, mkfile=False) -> str:
    p: str = join(ROOT, *parts)
    if mkparent and not isdir(dirname(p)):
        makedirs(dirname(p))
    if mkdir and not isdir(p):
        makedirs(p)
    elif mkfile and not isfile(p):
        Path(p).touch()
    return p


def logerr(msg: str, label=None) -> None:
    print(msg.upper() if not label else f'{label} {msg.upper()}', file=stderr)


def get_text(files=None, chunk_size_=10, quick=False) -> bytes:
    global TEXT
    if TEXT is not None:
        return TEXT
    if files is None:
        files = [join(ROOT, 'data', fname) for fname in listdir(join(ROOT, 'data'))]
    chunk_size = chunk_size_
    chunks: List[bytes] = []
    for path in files:
        with open(path, mode='rb') as f:
            while True:
                try:
                    c = f.read(chunk_size)
                    if not c:
                        break
                    else:
                        chunks.append(c)
                    chunk_size = min(chunk_size * 2, 100) if not quick else chunk_size * 2
                except Exception as e:
                    logerr(str(e))
                    chunk_size = 10
        logerr(f'FINISHED READING "{path}"')
        chunks.append(b'\n\n')
    TEXT = b''.join(chunks)
    return TEXT
