#!/usr/bin/python3

# Standard Library
import argparse
import logging
from os.path import abspath, basename, dirname, join, relpath
from time import strftime

# 3rd Party
from utils import ROOT, generate

if __name__ != '__main__':
    raise Exception('Must be run as a script')

output_file: str = join(
    ROOT,
    'generated',
    f"{strftime('%H:%M:%S#%y-%m-%d')}.txt",
)

parser = argparse.ArgumentParser(
    description='Process some integers.',
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
    '--print',
    '-p',
    action='store_true',
    required=False,
    default=True,
    help='print output')
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

args = parser.parse_args()

logging.basicConfig(
    level={'noset': 0, 'debug': 10, 'info': 20, 'warning': 30, 'error': 40, 'criticial': 50}[args.logging],
    format='%(levelname)s %(funcName)-13s %(lineno)3d %(message)s')

txt: str = generate(
    n=args.lookbehind,
    max_avg_txt_len=args.length)
if not args.no_print:
    print(txt)
if not args.no_output:
    with open(args.output, mode='w') as save_file:
        save_file.write(txt)
