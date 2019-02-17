#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os.path import join, abspath, dirname
from typing import Dict, Union, List
from time import time
from flask import Flask, request, Response, send_file
from markov_chunks import generate as genw
from markov_chars import generate as genc
from os import getenv
import logging
from utils import log, root_path

logging.basicConfig(
    level={
        'noset': 0,
        'debug': 10,
        'info': 20,
        'warning': 30,
        'error': 40,
        'critical': 50,
    }[getenv('LOG', 'info')],
    format='%(levelname)s %(funcName)-13s %(lineno)3d %(message)s')

ROOT: str = dirname(abspath(__file__))
app = Flask(__name__)


@app.route('/')
def index():
    return send_file(root_path('static', 'index.html'))


@app.route("/", methods=['POST'])
def story():
    d: Dict[str, Union[str, int, bool, List, Dict, None]] = request.get_json()
    algo: str = d['algo']
    if algo == 'word':
        print('is word')

        return Response(
                map(lambda b: b.decode('ascii', 'ignore'),
                    genw(seed=d['seed'].encode('ascii', 'ignore'),
                        n=d['n'],
                        max_len=d['max_len'])),
                mimetype='text/plain')

    elif algo == 'char':
        return Response(
                map(chr,
                    genc(seed=d['seed'].encode('ascii', 'ignore'),
                        n=d['n'],
                        max_len=d['max_len'])),
                mimetype='text/plain')
    else:
        raise Exception('unknown algorithm')


# pre-load
log.warning('[pre-loading] this should take < 1min ...')
start = time()
list(genw(n=8, max_len=10))
list(genc(n=8, max_len=10))
log.warn(f'[finished] pre-loading (took {time() - start:4.2f}s)')

app.run(port=3000)
