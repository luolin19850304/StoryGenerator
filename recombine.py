#!/usr/bin/env python3 

# Standard Library
from random import choices, randrange
from typing import Any, List, Set, Tuple
from pprint import pprint

# 3rd Party
from nltk import pos_tag, wordpunct_tokenize
# My Code
from utils import get_sents, rand_word, cached


def generate(no_sents=10, heroes=['Anne', 'Amy', 'Johanna']) -> Any:
    sents_all: List[str] = get_sents()
    idx_start: int = randrange(0, len(sents_all) - no_sents)
    skip_count = no_sents // 2 
    sents: List[str] = sents_all[idx_start:idx_start + no_sents + skip_count]
    print('\n'.join(sents))
    for _ in range(skip_count):
        sents.pop(randrange(0, len(sents)))
    sents_tokenized: List[List[str]] = [wordpunct_tokenize(s) for s in sents]
    sents_tagged: List[List[Tuple[str, str]]] = [pos_tag(s, tagset='universal') for s in sents_tokenized]
    ents: Set[str] = set()
    people = None
    objects = None
    for i in range(len(sents_tagged)):
        for j in range(len(sents_tagged[i])):
            word, tag = sents_tagged[i][j]
            if tag == 'NOUN' and j > 0:
                ents.add(word)
            # print(tag, word)
    ents = {e for e in ents if len(e) > 2}
    objects = {e for e in ents if e.lower() == e}
    people = {e for e in ents if e.capitalize() == e and e.lower() not in ents}
    pprint(dict(ents=ents, objects=objects, people=people))


if __name__ == '__main__':
    generate()
