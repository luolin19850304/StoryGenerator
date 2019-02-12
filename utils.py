import logging
import re
from os import makedirs, listdir
from os.path import dirname, abspath, isfile, isdir, join, relpath
from pathlib import Path
from re import MULTILINE, IGNORECASE
from sys import getsizeof
from threading import Lock
from time import time
from typing import Optional

log = logging.getLogger()

ROOT: str = dirname(abspath(__file__))

CLEAN_REGEX = re.compile(
    rb'^\s*([-*]+\s*)?(chapter|\*|note|volume|section|part|[IVX]+|harry\s+potter|by\s+|(the\s+)?end)[^\n\r]*$|\r+',
    MULTILINE | IGNORECASE)

TEXT: Optional[bytes] = None
TEXT_LK: Lock = Lock()

# processing
NEEDLESS_WRAP = re.compile(rb'([^\n])\n([^\n])')
TOO_MANY_NL = re.compile(rb'\n{3,}')
TOO_MANY_DASHES = re.compile(rb'(-\s*){3,}')
TOO_MANY_DOTS = re.compile(rb'(\.\s*){3,}')


def root_path(*parts, mkparent=True, mkdir=False, mkfile=False) -> str:
    p: str = join(ROOT, *parts)
    if mkparent and not isdir(dirname(p)):
        makedirs(dirname(p))
    if mkdir and not isdir(p):
        makedirs(p)
    elif mkfile and not isfile(p):
        Path(p).touch()
    return p


def get_text(files=None) -> bytes:
    global TEXT, TEXT_LK
    TEXT_LK.acquire()
    if TEXT is not None:
        log.debug('[cache hit] got text from cache')
        TEXT_LK.release()
        return TEXT
    start = time()
    if files is None:
        files = [join(ROOT, 'data', fname) for fname in listdir(join(ROOT, 'data')) if fname.endswith('.txt')]
    log.debug(f'[loading] text from {len(files)} files')
    texts: bytearray = bytearray()
    for path in files:
        start_file = time()
        with open(path, mode='rb') as f:
            log.debug(f'[loading] text from file {relpath(path)}')
            try:
                texts.extend(f.read())
            except Exception as e:
                log.warning(str(e))
        log.debug(f'[finished] reading from {relpath(path)} (read {getsizeof(texts[-1]) / 1e6:4.2f}MB in {time() - start_file:4.2f}s)')
    texts.extend(b'\n\n')
    TEXT = bytes(texts)
    TEXT = CLEAN_REGEX.sub(b'', TEXT)
    TEXT = NEEDLESS_WRAP.sub(rb'\1 \2', TEXT)
    TEXT = TOO_MANY_NL.sub(b'\n\n', TEXT)
    TEXT = TOO_MANY_DOTS.sub(rb'...', TEXT)
    TEXT = TOO_MANY_DASHES.sub(rb'--', TEXT)
    log.info(f'[finished] reading (read {getsizeof(TEXT) / 1e6:4.2f}MB in {time() - start:4.2f}s)')
    TEXT_LK.release()
    return TEXT
