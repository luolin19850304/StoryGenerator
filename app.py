#!/usr/bin/env python3
# -*- coding: utf-8 -*-

if __name__ != "__main__":
    raise Exception('Must be run as a script.')

# run only if run as script
from flask import Flask, request, render_template
from typing import Iterable, Iterator, List
from logging import Logger
import sys
import os
import logging
from utils import generate

# Standard Library
# import types for static typing (mypy, pycharm etc)

# 3rd Party

# initalise logging with sane configuration
logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s:%(asctime)s  %(message)s"
)

log: Logger = logging.getLogger()


app = Flask(__name__)


@app.route("/", methods=['POST'])
def serve():
    return generate(txt=request.get_data(), n=8, max_avg_txt_len=5000)

@app.route("/")
def home():
    return render_template('index.html')

generate(n=8)

app.run(port=3000)
