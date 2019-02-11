#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os.path import join, abspath, dirname
from typing import Dict, Union
from time import time
from flask import Flask, request, send_from_directory
from utils import generate, log
from os import getenv
import logging

logging.basicConfig(
    level={'noset': 0, 'debug': 10, 'info': 20, 'warning': 30, 'error': 40, 'critical': 50}[getenv('LOG', 'info')],
    format='%(levelname)s %(funcName)-13s %(lineno)3d %(message)s')

ROOT: str = dirname(abspath(__file__))
app = Flask(__name__)


@app.route('/')
def index():
    return send_from_directory(join(ROOT, 'static'), 'index.html')


@app.route("/", methods=['POST'])
def story():
    d: Dict[str, Union[str, int]] = request.get_json()
    print(d)
    return generate(
        txt=d.get('seed', '').encode('ascii', 'ignore'),
        n=d.get('n', 6),
        max_avg_txt_len=d.get('len', 5000))


# pre-load
log.warning('[pre-loading] this should take < 1min ...')
start = time()
generate(n=6)
log.warn(f'[finished] pre-loading (took {time() - start:4.2f}s)')

app.run(port=3000)
