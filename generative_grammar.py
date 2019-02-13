# Standard Library
from typing import Dict, List, Tuple

# 3rd Party
from numpy import ndarray
from numpy.random import choice

# My Code
from utils import rand_word, rand_sent_struct

SENTS: List[str] = None
TOKENS: List[str] = None
TAGGED_PAIRS: List[Tuple[str, str]] = None
TAG_PS: Dict[str, Dict[str, float]] = None
SENT_STRUCTS_PS: ndarray = None
SENT_STRUCTS: List[Tuple] = None


# noinspection PyDefaultArgument
def generate(no_sents=10, heroes=['Anne', 'Amy', 'Johanna']) -> List[List[str]]:
    sents: List[List[str]] = []
    for _ in range(no_sents):
        struct = list(rand_sent_struct())
        candidate = rand_word(struct[0])
        struct[0] = candidate[0].upper() + candidate[1:]
        for i in range(1, len(struct)):
            if struct[i] in {'NNV', 'NN', 'NNP'}:
                struct[i] = choice(heroes)
            else:
                candidate = rand_word(struct[i])
                struct[i] = candidate
        sents.append(struct)
    return sents
