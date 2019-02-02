import re
from os.path import join, dirname, isfile, abspath
from typing import Dict, List, Optional, Set
from pickle import dump, load
from sys import stderr

from numpy.random import choice, random
from string import ascii_letters

ROOT: str = dirname(abspath(__file__))

LETTERS: Set[int] = {ord(c) for c in ascii_letters if str.islower(c)}
LETTERS_U: Set[int] = {ord(c) for c in ascii_letters if str.isupper(c)}
CONSONANTS: Set[int] = {ord(c) for c in
                        {'b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x',
                         'y', 'w', 'z'}}
VOWELS: List[int] = [l for l in LETTERS if not (l in CONSONANTS)]
CONSONANTS_U: Set[int] = {ord(c) for c in
                          {'B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
                           'X', 'Y', 'W', 'Z'}}
VOWELS_U: Set[int] = {l for l in LETTERS_U if not (l in CONSONANTS_U)}
LETTERS: List[str] = list(LETTERS)

ACTORS_AND_LETTERS: List[str] = LETTERS

SPACE: int = ord(' ')

ASCII: List[int] = list(range(128))
SENT_STARTERS = list(VOWELS_U | CONSONANTS_U)


def logerr(msg: str) -> None:
    print(msg.upper(), file=stderr)


def get_text(chunk_size=10) -> str:
    from os import listdir
    txt = ''
    for fname in listdir(join(ROOT, 'data')):
        with open(join(ROOT, 'data', fname)) as f:
            while True:
                try:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    else:
                        txt += chunk
                    chunk_size *= 2
                except Exception as e:
                    logerr(str(e))
                    chunk_size = 10
        logerr(f'FINISHED READING "{fname}"')
        txt += '\n\n'
    return txt


def get_counts(txt: str, n: int = 4) -> Dict[str, List[int]]:
    counts: Dict[str, List[int]] = dict()
    for i in range(len(txt) - n - 20):
        ngram: str = txt[i:i + n]
        next_char: int = ord(txt[i + n])
        if next_char > 127:
            continue
        # print(f'ngram: {ngram}, next char: {next_char}')
        if not (ngram in counts):
            counts[ngram] = [0 for i in range(128)]
            counts[ngram][next_char] = 1
        else:
            counts[ngram][next_char] += 1
    return counts


def get_ps(counts: Dict[str, List[int]]) -> Dict[str, List[float]]:
    for substr in counts:
        total: int = sum(counts[substr])
        for i in range(len(counts[substr])):
            counts[substr][i] /= total
    return counts


def predict(txt: str, counts: Dict[str, List[float]], n=4) -> int:
    slice = txt[-n:]
    maybe_ps: Optional[List[float]] = counts.get(slice, None)
    if maybe_ps:
        return choice(a=ASCII, p=maybe_ps)
    last = slice[-1]
    if last == ',' or last == '.' or last or last == ';' or last == '-':
        return SPACE
    elif slice[-2:] == '. ' or slice[-2:] == '.\n' or slice[-2:] == '.\r':
        return choice(a=SENT_STARTERS)
    elif last in CONSONANTS:
        return choice(a=VOWELS)
    else:
        return choice(a=ACTORS_AND_LETTERS)


def main(text='Harry Potter was in a great mood that day.', n=6, text_len=1000, force=False, actors=['Harry', 'Ron']) -> None:
    SENT_STARTERS.extend(actors)
    ACTORS_AND_LETTERS.extend(actors)
    model_path: str = join(dirname(__file__), 'model')
    counts: Optional[Dict[str, List[float]]] = None
    if force or not isfile(model_path):
        logerr('creating new model')
        with open(model_path, mode='wb') as f:
            counts = get_ps(get_counts(get_text(), n))
            dump(counts, f)
            logerr(f'model saved in {model_path}')
    else:
        logerr(f'loading model from {model_path}')
        with open(model_path, mode='rb') as f:
            counts = load(f)

    while len(text) < text_len:
        text += chr(predict(text, counts, n))
        if (text.endswith(' ') or text.endswith('.')) and random() <= 0.01:
            this_sent = text[-text.rfind('.'):]
            mentioned = False
            for actor in actors:
                if actor in this_sent:
                    mentioned = True
                    break
            if not mentioned:
                text += choice(actors)

    print(re.compile(r'(\n\r?)+|\s{2,}').sub(' ', text) + '. THE END.')


if __name__ == '__main__':
    main(n=6, force=True, text_len=60000)
