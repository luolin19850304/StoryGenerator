#!/usr/bin/env python3

# Standard Library
import argparse
import logging
from os.path import abspath, basename, dirname, join, relpath
from time import strftime

# 3rd Party
from markov_chunks import generate as genw
from markov_chars import generate as genc
from utils import ROOT

if __name__ != '__main__':
    raise Exception('Must be run as a script')

output_file: str = join(
    ROOT,
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
    dest='no_print',
    default=False,
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
    '--force',
    '-f',
    required=False,
    action='store_true',
    default=False,
    help="don't use cache")
parser.add_argument(
    '--chars',
    '-c',
    required=False,
    action='store_true',
    default=False,
    help='use char lookbehind')
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
    format='%(levelname)s %(funcName)-13s %(lineno)3d %(message)s')

txt: str = (genc if args.chars else genw)(args.seed.encode('ascii', 'ignore'), args.lookbehind, args.length, force=args.force)
if not args.no_print:
    print(txt)
if not args.no_output:
    with open(args.output, mode='w') as save_file:
        save_file.write(txt)
