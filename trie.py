# Standard Library
from typing import Dict, Generator, Iterable, Iterator, List, Optional, Tuple

# 3rd Party
from nltk import wordpunct_tokenize
from numpy.random import choice

# My Code
from utils import gen_sent, get_trie


def generate(max_sents=20) -> Generator[str, None, None]:
    trie = get_trie()
    for _ in range(max_sents):
        yield from gen_sent(trie)
