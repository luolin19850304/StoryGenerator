#!/usr/bin/env python3

if __name__ != '__main__':
    raise Exception('Must be run as a script')

# Standard Library
from sys import stdout
import argparse
import logging
from os.path import abspath, basename, dirname, relpath
from typing import List
from time import strftime

# My Code
from markov_chars import generate as genc
from markov_chunks import generate as genw
from utils import root_path


output_file: str = root_path(
    'generated',
    f"{strftime('%H:%M:%S#%y-%m-%d')}.txt",
)

parser = argparse.ArgumentParser(
    description='Generate prose.',
    prog=basename(dirname(abspath(__file__))))

parser.add_argument(
    '--lookbehind',
    '-n',
    metavar='INT',
    type=int,
    nargs='?',
    default=6,
    choices=tuple(range(1, 20)),
    required=False,
    help='max length of lookbehind ngrams')
parser.add_argument(
    '--length',
    '-l',
    metavar='INT',
    nargs='?',
    type=int,
    required=False,
    default=1000,
    help='length of generated text')
parser.add_argument(
    '--no-output',
    '-O',
    action='store_true',
    dest='no_output',
    required=False,
    help=f"don't save output in file")
parser.add_argument(
    '--output',
    '-o',
    metavar='PATH',
    nargs='?',
    required=False,
    default=output_file,
    help=f'output path (DEFAULT: ./{relpath(output_file)})')
parser.add_argument(
    '--no-print',
    '-P',
    action='store_false',
    dest='print',
    default=True,
    required=False,
    help="don't print the output")
parser.add_argument(
    '--logging',
    '-t',
    metavar='THRESHOLD',
    nargs='?',
    required=False,
    choices={'noset', 'debug', 'info', 'warning', 'error', 'criticial'},
    default='info',
    help='set logging threshold')
parser.add_argument(
    '--algo',
    '-a',
    required=False,
    default='chars',
    choices={'char', 'word', 'grammar'},
    help='which algorithm to use')
parser.add_argument(
    '--seed',
    '-s',
    required=False,
    metavar='TEXT',
    default='',
    help='seed (text to bootstrap the algorithm)')

args = parser.parse_args()

logging.basicConfig(
    level={'noset': 0, 'debug': 10, 'info': 20, 'warning': 30, 'error': 40, 'critical': 50}[args.logging],
    format='%(levelname)-s %(threadName)-s/%(module)-s:%(lineno)-d.%(funcName)-s %(message)s')

ASCII_TABLE_B: List[bytes] = [
        chr(i).encode('ascii', 'ignore')
        for i in range(128)]

ASCII_TABLE_S: List[str] = [
        chr(i)
        for i in range(128)]

if args.algo == 'char':

    def chars():
        return genc(seed=args.seed.encode('ascii', 'ignore'), n=args.lookbehind, max_len=args.length)

    i = 0

    if args.print and not args.no_output:
        with open(args.output, mode='wb') as f:
            for c in chars():
                f.write(ASCII_TABLE_B[c])
                stdout.write(ASCII_TABLE_S[c])
                if i == 20:
                    stdout.flush()
                    i = 0
                i += 1

    elif not args.print:
        with open(args.output, mode='wb') as f:
            for c in chars():
                f.write(ASCII_TABLE_B[c])

    elif not args.no_output:
        for c in chars():
            stdout.write(ASCII_TABLE_S[c])
            if i == 20:
                stdout.flush()
                i = 0
            i += 1


elif args.algo == 'word':

    def chunks():
        return genw(seed=args.seed.encode('ascii', 'ignore'), n=args.lookbehind, max_len=args.length)

    if args.print and not args.no_output:
        with open(args.output, mode='wb') as f:
            for chunk in chunks():
                f.write(chunk)
                stdout.write(chunk.decode('ascii', 'ignore'))
                stdout.flush()

    elif not args.print:
        with open(args.output, mode='wb') as f:
            for chunk in chunks():
                f.write(chunk)

    elif not args.no_output:
        for chunk in chunks():
            stdout.write(chunk.decode('ascii', errors='ignore'))
            stdout.flush()
