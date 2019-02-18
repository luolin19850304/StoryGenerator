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
from re import IGNORECASE, MULTILINE, ASCII
from sys import getsizeof
from threading import Lock, Semaphore
from time import time
from typing import Any, Dict, Generator, Iterable, Iterator, List, Match, Optional, Pattern, Tuple, Union

# 3rd Party
import nltk
from nltk import pos_tag, sent_tokenize, word_tokenize, wordpunct_tokenize

AVG_CHUNK_LEN = 5
MIN_PROB: float = 1e-4
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
        MULTILINE | IGNORECASE | ASCII)


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

    with ThreadPool(max_workers=NO_CPUS,
                    thread_name_prefix='get_text') as pool:
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


def get_char_ps(text: Optional[str] = None) -> List[float]:

    use_decorator = text is None

    @cached(what='CHAR_PS',
            load=use_decorator,
            save=use_decorator,
            keep=use_decorator)
    def inner() -> List[float]:
        bag = Counter(get_text() if text is None else text)
        counts: Union[List[int], List[float]] = [0 for _ in range(128)]
        for k, v in bag.items():
            code = ord(k)
            if code < 128:
                counts[code] = v
        del bag

        total: int = sum(counts)

        for c in range(128):
            counts[c] /= total

        return counts

    return inner()


def get_nchar_dict(n=2, text=None) -> Dict[str, Dict[int, float]]:
    assert 20 >= n >= 1, f'nchar len must be in [1, 20] but got n = {n}'

    @cached(what=f'_{n}CHAR_DICT', keep=(text is None), save=(text is None), load=(text is None))
    def inner() -> Dict[str, Dict[int, float]]:
        txt: str = get_text() if text is None else text
        ps: Dict[str, Dict[int, float]] = dict()

        for i in range(len(txt) - n - 1):
            chars_before: str = txt[i:i + n]
            char_after: int = ord(txt[i + n])
            maybe_d: Optional[Dict[int, float]] = ps.get(chars_before, None)
            if maybe_d is None:
                ps[chars_before] = {char_after: 1}
            else:
                maybe_d[char_after] = maybe_d.get(char_after, 0) + 1

        del txt

        for nchar in list(ps.keys()):

            total = sum(ps[nchar].values())

            # prune
            for char_after in list(ps[nchar].keys()):
                if ps[nchar][char_after] == 1:
                    del ps[nchar][char_after]
                    total -= 1
                elif (ps[nchar][char_after] / total) < MIN_PROB:
                    total -= ps[nchar][char_after]
                    del ps[nchar][char_after]

            # empty dict as a result of prunning
            if len(ps[nchar]) == 0:
                del ps[nchar]

            # counts -> probs 
            else:
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


def get_tagged_words(
        universal=False, text: Optional[str] = None) -> List[Tuple[str, str]]:

    use_decorator = text is None

    @cached(what=f'TAGGED{"_UNIVERSAL" if universal else ""}_WORDS',
            keep=use_decorator,
            load=use_decorator,
            save=use_decorator)
    def inner() -> List[Tuple[str, str]]:
        return pos_tag(
                get_words() if text is None else word_tokenize(text),
                tagset=('universal' if universal else None))

    return inner()


def get_tagged_sents(
        universal=False,
        split='word',
        text: Optional[str] = None) -> List[List[Tuple[str, str]]]:

    tokenize = word_tokenize if split == 'word' else wordpunct_tokenize
    tagset = 'universal' if universal else None

    use_decorator = text is None

    @cached(what=f'{"UNIVERSAL_" if universal else ""}TAGGED_SENTS',
            keep=use_decorator,
            save=use_decorator,
            load=use_decorator)
    def inner():
        tokenize = word_tokenize if split == 'word' else wordpunct_tokenize
        return [pos_tag(s, tagset=tagset) for s in [tokenize(s) for s in (get_sents() if text is None else sent_tokenize(text))]]

    return inner()


def get_part_of_speech(tag: str, universal=False, text: Optional[str] = None) -> Dict[str, float]:

    use_decorator = text is None

    @cached(what=f'WORDS_{"_UNIVERSAL" if universal else ""}{tag}',
            save=use_decorator,
            load=use_decorator,
            keep=use_decorator)
    def inner() -> Counter:
        is_word_regex = re.compile(r'^[A-Za-z]{1,}$')
        words: Union[Counter, List[str]] = []
        for sent in get_tagged_sents(universal=universal, text=text):
            for pair in sent:
                word, tag2 = pair
                if tag == tag2 and is_word_regex.match(word):
                    words.append(word)
        del is_word_regex
        words = Counter(words)
        total = sum(words.values())
        for w, count in words.items():
            words[w] /= total
        return words

    return inner()


def get_tag_dict(universal=False,
                 text: Optional[str] = None) -> Dict[str, Dict[str, float]]:

    use_decorator = text is None

    @cached(what=f'TAG_{"UNIVERSAL_" if universal else ""}DICT',
            load=use_decorator,
            keep=use_decorator,
            save=use_decorator)
    def inner() -> Dict[str, Dict[str, float]]:
        ps: Dict[str, Dict[str, Union[int, float]]] = dict()
        for word, tag in get_tagged_words(universal=universal, text=text):
            if ps.get(tag, None) is None:
                ps[tag] = dict()
            ps[tag][word] = ps[tag].get(word, 0) + 1

        for tag in ps:
            total = sum(ps[tag].values())
            for word in ps[tag]:
                ps[tag][word] /= total
        return ps

    return inner()


def get_tags(universal=False, text: Optional[str] = None) -> List[str]:
    """Tag names (as seen by NLTK).
    Universal tags are simplified.
    """

    use_decorator = text is None

    @cached(what=f'TAGS{"_UNIVERSAL" if universal else ""}',
            load=use_decorator,
            keep=use_decorator,
            save=use_decorator)
    def inner() -> List[str]:
        return list(get_tag_dict(universal=universal, text=text).keys())

    return inner()


def get_word_dict(split='word', text: Optional[str] = None) -> Dict[str, float]:
    """Dictionary of { word (str) => probability (float), ... } pairs.
    """

    use_decorator = text is None

    @cached(what=f'{"WORD" if split == "word" else "WORDPUNCT"}_DICT',
            load=use_decorator,
            keep=use_decorator,
            save=use_decorator)
    def inner() -> Dict[str, float]:
        ps = Counter(
                get_words() if split == 'word' and text is None
                else get_wordpuncts() if split == 'wordpunct' and text is None
                else word_tokenize(text) if split == 'word'
                else wordpunct_tokenize(text))
        total: int = sum(ps.values())
        for token in ps:
            ps[token] /= total
        return ps

    return inner()


def rand_sent(tagged=False, universal=False, text: Optional[str] = None) -> Union[str, List[Tuple[str, str]]]:
    return choice(
        get_tagged_sents(universal=universal, text=text) if tagged
        else get_sents() if text is None
        else sent_tokenize(text))


def rand_word(tag: Optional[str] = None, universal=False, text: Optional[str] = None) -> str:
    tag_dict = get_tag_dict(universal=universal, text=text)
    if tag is None:
        tag = choice(get_tags(universal=universal, text=text))
    return choices(
        tuple(tag_dict[tag].keys()),
        tuple(tag_dict[tag].values()))[0]


def get_ents(type='object',
             plural=None,
             text: Optional[str] = None) -> Dict[str, float]:
    assert type in {'object', 'person'}, \
        f'type of entity must be: "object" OR "person"'

    use_decorator = text is None

    @cached(what=(f'ENTS_{"PEOPLE" if type == "person" else "OBJECTS"}_{"SINGULAR" if plural is False else "PLURAL" if plural is True else "ALL"}'),
            save=use_decorator,
            load=use_decorator,
            keep=use_decorator)
    def inner() -> Dict[str, float]:
        ents: Dict[str, float] = get_part_of_speech(
                'NOUN', universal=True, text=text)

        def remove_all(test=(lambda e: False)):
            for e in filter(test, list(ents.keys())):
                p: float = ents[e]
                del ents[e]
                l: int = len(ents)
                share: float = p / l
                for other in ents:
                    ents[other] += share

        is_word_regex = re.compile(r'[A-Za-z]{4,}')
        remove_all(lambda e: not is_word_regex.fullmatch(e))
        del is_word_regex


        if type == 'person':
            remove_all(lambda e: e.capitalize() != e or (e.lower() in ents))
        else:
            remove_all(lambda e: e.capitalize() == e or (e.lower() not in ents))


        if plural is True:
            remove_all(lambda e: not (e.endswith('s') or e.endswith('i')))

        elif plural is False:
            remove_all(lambda e: e.endswith('s') or e.endswith('i'))

        return ents

    return inner()


def rand_ent(type='object', plural=None, text: Optional[str] = None) -> str:
    assert type in {'object', 'person'}, \
            f'type of entity must be: "object" OR "person"'

    d = get_ents(text=text, plural=plural, type=type)

    xs: Tuple[str] = tuple(d.keys())
    ps: Tuple[float] = tuple(d.values())

    del d

    def pick() -> str:
        return choices(xs, ps)[0]

    candidate: str = pick()

    if plural is False:
        while candidate.endswith('s') or candidate.endswith('i'):
            candidate = pick()
    elif plural is True:
        while not (candidate.endswith('s') or candidate.endswith('i')):
            candidate = pick()

    return candidate


def get_size(obj: Any, seen=None):
    """Recursively find size of an object.
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
