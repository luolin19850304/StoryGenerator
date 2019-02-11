#!/usr/bin/env python3
# -*- coding: utf-8 -*-

if __name__ != "__main__":
    raise Exception('Must be run as a script.')

from utils import generate, log
from flask import Flask, render_template, request
from typing import Iterable, Iterator, List
from logging import Logger
import sys
import os
import logging

app = Flask(__name__)


@app.route("/", methods=['POST'])
def serve():
    return generate(txt=request.get_data(), n=9, max_avg_txt_len=5000)


@app.route("/")
def home():
    return render_template('index.html')


# pre-load
log.warn('pre-loading...')
generate(n=9)

app.run(port=3000)
