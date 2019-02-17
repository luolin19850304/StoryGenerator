# Standard Library
import logging
import os
import sys
from logging import Logger
from pprint import pprint
# import types for static typing (mypy, pycharm etc)
from typing import Iterable, Iterator, List, Tuple, Set
from random import choice, random

# 3rd Party
import nltk
from nltk import pos_tag, wordpunct_tokenize
from nltk.corpus import wordnet as wn
from utils import get_sents


def generate():
    sents: List[str] = get_sents()
    doc: List[List[str]] = [wordpunct_tokenize(s) for s in sents]
    doc_tagged: List[List[Tuple[str, str]]] = [pos_tag(s) for s in sents]
    ents: Set[str] = set()
    objects: Set[str] = set()
    people: Set[str] = set()


class Generator():

    def __init__(self,
                 people=["Anne", "Joseph", "Suzie", "Johanna"],
                 objects=["dog", "cat", "table", "smoothie"]):
        self.sents = [[]]
        self.objects = objects
        self.people = people

    def make_story(self, max_len=10) -> List[List[str]]:
        self.sents = [[]]
        while len(self.sents) < max_len:
            self.make_sent()
        self.sents = self.sents[:-1]
        pprint(self.sents)
        self.postprocess()
        return self.sents

    def postprocess(self) -> None:
        for s in range(len(self.sents)):
            self.sents[s][0] = \
                    self.sents[s][0].capitalize()

    def make_sent(self) -> None:
        self.make_np()
        self.make_vp()
        self.sents.append([])

    def make_np(self) -> None:
        if random() >= 0.3:
            self.make_np_obj()
        else:
            self.make_np_person()

    def make_np_obj(self) -> None:
        self.sents[-1].append(choice(self.objects))

    def make_np_person(self) -> None:
        self.sents[-1].append(choice(self.people))

    def make_vp(self) -> None:
        self.sents[-1].append(
                choice(
                    ['ate', 'barked', 'shined', 'told', 'said', 'thought']))
