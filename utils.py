# Standard Library
import logging
import lzma
import pickle
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor as ThreadPool
from os import listdir, makedirs
from os.path import abspath, dirname, isdir, isfile, join, relpath
from pathlib import Path
from random import choice, choices
from re import IGNORECASE, MULTILINE
from sys import getsizeof
from threading import Lock, Semaphore
from time import time
from typing import Any, Dict, Generator, Iterable, Iterator, List, Match, Optional, Pattern, Tuple, Union

# 3rd Party
import nltk
from nltk import pos_tag, sent_tokenize, word_tokenize, wordpunct_tokenize

AVG_CHUNK_LEN = 5
NO_CPUS = 4

DQUOTE: int = ord(b'"')
SPACE: int = ord(b' ')

log = logging.getLogger()

ROOT: str = dirname(abspath(__file__))


def cached(what: str, keep=True, load=True, save=True, archive=True):
    """Decorator function that checks if file WHAT exists,
    if it does it unpickles it and returns it's value,
    otherwise, data is generated from calling fn.
    The data is then pickled (cached) to <PROJECT_ROOT>/cache/<WHAT>.
    """
    def outer(fn):
        def inner():
            lock_r = '_{0}_LK_R'.format(what)
            if globals().get(lock_r, None) is None:
                globals()[lock_r] = Semaphore(NO_CPUS)
            with globals()[lock_r]:
                if globals().get(what, None) is not None:
                    log.debug(f'[cache hit] found {what}')
                    return globals()[what]
                path = root_path('cache', what)
                if load and isfile(path):
                    lock_w = '_{0}_LK_W'.format(what)
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


@cached('CLEAN_REGEX')
def get_clean_regex() -> Pattern[str]:
    return re.compile(
        r'^\s*([-*]+\s*)?(chapter|\*|note|volume|section|part|[IVX]+|harry\s+potter|by\s+|(the\s+)?end)[^\n\r]*$|\r+',
        MULTILINE | IGNORECASE)


@cached('CHUNK_REGEX')
def get_chunk_regex() -> Pattern[str]:
    PUNCT_REGEX_B = r'(?P<punct>-{1,2}|[:;,"])'
    NL_REGEX_B = r'(?P<nl>(\n\r?|\r\n?)+)'
    SENT_END_REGEX_B = r'(?P<sent_end>(!(!!)?|\?(\?\?)?|\.(\.\.)?))'
    WS_REGEX_B = r'(?P<ws>\s)'
    WORD_REGEX_B = r"(?P<word>[A-Za-z]+(-[A-Za-z]+)*?('[a-z]{0,7})?)"
    DATE_REGEX_B = r"(?P<date>([1-9]\d*)(th|st|[nr]d)|(19|20)\d{2})"
    TIME_REGEX_B = r"(?P<time>\d+((:\d{2}){1,2}|(\.\d{2}){1,2}))"
    return re.compile(r'(?P<token>' + r'|'.join([
        WS_REGEX_B,
        SENT_END_REGEX_B,
        PUNCT_REGEX_B,
        NL_REGEX_B,
        WORD_REGEX_B,
        DATE_REGEX_B,
        TIME_REGEX_B,
    ]) + r')', IGNORECASE)


def chunk(txt: str) -> Iterable[Match[str]]:
    """Chunk text (bytes) into chunks (matches).
    """
    return get_chunk_regex().finditer(txt)


def root_path(*parts, mkparent=True, mkdir=False, mkfile=False) -> str:
    """Produce a path relative to the root of this project.
    """
    p: str = join(ROOT, *parts)
    if mkparent and not isdir(dirname(p)):
        makedirs(dirname(p))
    if mkdir and not isdir(p):
        makedirs(p)
    elif mkfile and not isfile(p):
        Path(p).touch()
    return p


# noinspection PyDefaultArgument
@cached('TEXT', keep=False, load=False, save=False)
def get_text(files=[root_path('data', fname) for fname in listdir(root_path('data')) if fname.endswith('.txt')]) -> str:
    """Efficiently read all files into a single bytearray.
    """
    log.debug(f'[loading] text from {len(files)} files')
    texts: List[str] = []
    lock = Lock()

    def read_file(path: str) -> None:
        start_file = time()
        fname = relpath(path)
        with open(path) as f:
            log.debug(f'[loading] text from file {fname}')
            try:
                txt = f.read()
                lock.acquire()
                texts.append(txt)
                texts.append('\n\n')
                lock.release()
            except Exception as e:
                log.warning(str(e))
        log.debug(f'[finished] reading from {fname} (read {getsizeof(texts[-2]) / 1e6:4.2f}MB in {time() - start_file:4.2f}s)')

    with ThreadPool(max_workers=NO_CPUS, thread_name_prefix='get_text') as pool:
        for task in [pool.submit(fn=read_file, path=p) for p in files]:
            task.result()

    return '\n\n'.join(texts)


def capitalize(txt: str) -> str:
    """Captialize the first letter (only) in txt.
    """
    if len(txt) <= 1:
        return txt
    pos = 0
    while ord(txt[pos]) == SPACE:
        pos += 1
    # is lowercase
    if 97 <= ord(txt[pos]) <= 122:
        return txt[:pos] + chr(ord(txt[pos]) - 32) + txt[pos + 1:]
    else:
        return txt


@cached('CHUNKS')
def get_chunks() -> List[str]:
    """Chunks generated from <PROJECT_ROOT>/data/*.txt.
    """
    ms: List[Match[str]] = list(chunk(get_text()))
    chunks = [ms[0].group(0), ms[0].group(0)]
    # not checking for len of tokens because every token has len >= 1
    for i in range(2, len(ms) - 1):
        s: str = ms[i].group(0)
        is_q = s[0] == DQUOTE
        is_w = bool(ms[i].group('word'))
        is_ws = bool(ms[i].group('ws'))
        is_p = bool(ms[i].group('punct'))
        is_nl = bool(ms[i].group('nl'))
        is_end = bool(ms[i].group('sent_end'))
        is_cap = 97 <= ord(s[0]) <= 122
        if is_w and ms[i - 2].group('sent_end') and (97 <= ord(chunks[-2][0]) <= 122):
            chunks[-2] = capitalize(chunks[-2])
        if is_w and ms[i - 1].group('word'):
            chunks.append('. ' if is_cap else ' ')
        if is_w and is_cap and (ms[i - 2].group('sent_end') or ms[i - 1].group('nl')):
            chunks.append(capitalize(s))
            continue
        elif (is_w and ms[i + 1].group('word')) \
                or (is_end and not (ms[i + 1].group('ws') or ms[i + 1].group('nl'))) \
                or (is_p and not (is_q or ms[i + 1].group('ws') or ms[i - 1].group('nl'))):
            chunks.append(s)
            chunks.append(' ')
            continue
        elif (is_nl and not ms[i - 1].group('sent_end') and not ms[i + 1].group(0)[0] == DQUOTE and ms[i + 1].group('punct')) \
                or ((is_end or is_p or is_ws or is_p) and s == ms[i + 1].group(0)) \
                or (is_ws and (ms[i + 1].group('sent_end') or ms[i + 1].group('nl'))):
            continue
        else:
            chunks.append(s)
    chunks.append(ms[-1].group(0))
    return chunks


def get_nchunk_dict(n=2, split='chunk') -> Dict[Tuple, Dict[str, float]]:
    """Dictionary of probabilites for ngrams of len n.
    """
    assert 20 >= n >= 1, f'n{split} len must be in [1, 20] but got n = {n}'

    @cached(f'_{n}{"CHUNK" if split == "chunk" else "WORD" if split == "word" else "WORDPUNCT"}_DICT')
    def inner() -> Dict[Tuple, Dict[str, float]]:
        tokens: List[str] = get_chunks() if split == 'chunk' \
                else get_words() if split == 'word' \
                else get_wordpuncts()
        ps: Dict[Tuple, Dict[str, float]] = dict()
        for i in range(len(tokens) - n - 1):
            words_before: Tuple = tuple(tokens[i:i + n])
            next_word: str = tokens[i + n]
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


@cached('CHAR_PS')
def get_char_ps() -> List[float]:
    bag = Counter(get_text())
    counts: List[int] = list(range(128))
    for k, v in bag.items():
        if k < 128:
            counts[k] = v
    ps: List[float] = [0.0 for _ in range(128)]
    total: int = sum(counts)
    for c in range(128):
        ps[c] = counts[c] / total
    return ps


def get_nchar_dict(n=2) -> Dict[str, Dict[int, float]]:
    assert 20 >= n >= 1, f'nchar len must be in [1, 20] but got n = {n}'

    @cached(f'_{n}CHAR_DICT')
    def inner() -> Dict[str, Dict[int, float]]:
        txt: str = get_text()
        ps: Dict[str, Dict[int, float]] = dict()

        for i in range(len(txt) - n - 1):
            chars_before: str = txt[i:i + n]
            char_after: int = ord(txt[i + n])
            if chars_before not in ps:
                ps[chars_before] = {char_after: 1}
            else:
                ps[chars_before][char_after] = \
                        ps[chars_before].get(char_after, 0) + 1

        for nchar in ps:
            total = sum(ps[nchar].values())
            for char_after in ps[nchar]:
                ps[nchar][char_after] /= total

        return ps

    return inner()


@cached('WORDS')
def get_words() -> List[str]:
    return word_tokenize(get_text())


@cached('WORDPUNCTS')
def get_wordpuncts() -> List[str]:
    return wordpunct_tokenize(get_text())


@cached('SENTS')
def get_sents() -> List[str]:
    return sent_tokenize(get_text())


@cached('TAGGED_WORDS')
def get_tagged_words(universal=False) -> List[Tuple[str, str]]:
    return pos_tag(get_words(), tagset=('universal' if universal else None))


def get_tagged_sents(universal=False) -> List[List[Tuple[str, str]]]:

    @cached(f'{"UNIVERSAL_" if universal else ""}TAGGED_SENTS')
    def inner():
        return [pos_tag(s, tagset=('universal' if universal else None))
                for s in [wordpunct_tokenize(s) for s in get_sents()]]
    return inner()


def get_part_of_speech(tag: str, universal=False) -> Dict[str, float]:

    @cached(f'WORDS_{"_UNIVERSAL" if universal else ""}{tag}')
    def inner() -> Counter:
        is_word_regex = re.compile(r'^[A-Za-z]{1,}$')
        words: Union[Counter, List[str]] = []
        for sent in get_tagged_sents(universal=universal):
            for pair in sent:
                word, tag2 = pair
                if tag == tag2 and is_word_regex.match(word):
                    words.append(word)
        words = Counter(words)
        total = sum(words.values())
        for w, count in words.items():
            words[w] /= total
        return words

    return inner()


@cached('ENTS_OBJECTS')
def get_ents_objects() -> Dict[str, float]:
    is_word_regex = re.compile(r'^[A-Za-z]{4,}$')
    return {e: p for e, p in
            dict(((k, v)
                  for k, v in get_part_of_speech('NOUN', universal=True).items()
                  if is_word_regex.match(k))).items()
            if e.lower() == e}


@cached('ENTS_PEOPLE')
def get_ents_people() -> Dict[str, float]:
    is_word_regex = re.compile(r'^[A-Za-z]{4,}$')
    ents = dict(((k, v)
                 for k, v in get_part_of_speech('NOUN', universal=True).items()
                 if is_word_regex.match(k)))
    return {e: p
            for e, p in ents.items()
            if e.capitalize() == e and e.lower() not in ents}


@cached('SENT_STRUCTS_DICT')
def get_sent_structs_ps(universal=False) -> Dict[Tuple, float]:

    def get_sent_structs() -> Generator[Tuple, None, None]:
        buf: List[str] = []
        for word, tag in get_tagged_words(universal=universal):
            buf.append(tag)
            if tag == '.':
                yield tuple(buf)
                buf = []

    bag = Counter(get_sent_structs())

    total: int = sum(bag.values())

    for s in bag:
        bag[s] /= total

    return bag


def get_tag_dict(universal=False) -> Dict[str, Dict[str, float]]:

    @cached(f'TAG_{"UNIVERSAL_" if universal else ""}DICT')
    def inner() -> Dict[str, Dict[str, float]]:
        ps: Dict[str, Dict[str, Union[int, float]]] = dict()
        for word, tag in get_tagged_words(universal=universal):
            if ps.get(tag, None) is None:
                ps[tag] = dict()
            ps[tag][word] = ps[tag].get(word, 0) + 1

        for tag in ps:
            total = sum(ps[tag].values())
            for word in ps[tag]:
                ps[tag][word] /= total
        return ps

    return inner()


def get_tags(universal=False) -> List[str]:
    """Tag names (as seen by NLTK).
    Universal tags are simplified.
    """

    @cached(f'TAGS{"_UNIVERSAL" if universal else ""}')
    def inner() -> List[str]:
        return list(get_tag_dict(universal=universal).keys())

    return inner()


def get_tag_ps(universal=False) -> List[float]:
    """Probabilities for each tag.
    """

    @cached(f'TAG{"_UNIVERSAL" if universal else ""}_PS')
    def inner() -> List[float]:
        return list(get_tag_dict(universal=universal).values())

    return inner()


def get_chunk_dict(split='chunk') -> Dict[str, float]:
    """Dictionary of { chunk (str) => probability (float), ... } pairs.
    """

    @cached(f'{"CHUNK" if split == "chunk" else "WORD" if split == "word" else "WORDPUNCT"}_DICT')
    def inner() -> Dict[str, float]:
        ps = Counter(
                get_chunks() if split == 'chunk'
                else get_words() if split == 'word'
                else get_wordpuncts())
        total: int = sum(ps.values())
        for token in ps:
            ps[token] /= total
        return ps

    return inner()


def rand_sent_struct(universal=False) -> List[str]:
    a, p = zip(*get_sent_structs_ps(universal=universal).items())
    return choices(tuple(a), tuple(p))[0]


def rand_sent(tagged=False, universal=False) -> Union[str, List[Tuple[str, str]]]:
    return choice(
        get_tagged_sents(universal=universal) if tagged else get_sents())


def rand_word(tag: Optional[str] = None, universal=False) -> str:
    tag_dict = get_tag_dict(universal=universal)
    if tag is None:
        tag = choices(
                get_tags(universal=universal),
                get_tag_ps(universal=universal))[0]
    return choices(
        tuple(tag_dict[tag].keys()),
        tuple(tag_dict[tag].values()))[0]


def rand_ent_person() -> str:
    return choices(
        tuple(get_ents_people().keys()),
        tuple(get_ents_people().values()))[0]


def rand_ent_object() -> str:
    return choices(
        tuple(get_ents_objects().keys()),
        tuple(get_ents_objects().values()))[0]


def get_size(obj: Any, seen=None):
    """Recursively finds size of objects.
    """
    size = getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') \
            and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size
