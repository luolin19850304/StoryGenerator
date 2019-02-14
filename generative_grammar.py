# Standard Library
from typing import List, Sized

# 3rd Party
import numpy as np
from numpy import ndarray
from numpy.random import choice

# My Code
from utils import rand_word, rand_sent_struct


def get_hero_ps(heroes: Sized) -> ndarray:
    ps: List[float] = [.6]
    while len(ps) < len(heroes) - 1:
        ps.append((1 - sum(ps)) / 2)
    ps.append(1 - sum(ps))
    return np.array(ps, dtype='float64')


# noinspection PyDefaultArgument
def generate(no_sents=10, heroes=['Anne', 'Amy', 'Johanna']) -> List[List[str]]:
    sents: List[List[str]] = []
    heroes = np.array(heroes)
    ps = get_hero_ps(heroes)
    for _ in range(no_sents):
        struct = list(rand_sent_struct())
        candidate = rand_word(struct[0])
        struct[0] = candidate[0].upper() + candidate[1:]
        for i in range(1, len(struct)):
            struct[i] = choice(a=heroes, p=ps) \
                if struct[i] in {'NNV', 'NN', 'NNP'} \
                else rand_word(struct[i])
        sents.append(struct)
    return sents
