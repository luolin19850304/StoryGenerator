from typing import List, Optional, Dict
import numpy as np

from utils import tokenize, get_ngram_ps, get_ps


def main(txt=b'Harry was in a great mood.', n=6, max_no_tokens=1000) -> None:
    tokens: List[bytes] = tokenize(txt)
    ps: Dict[bytes, float] = get_ps()
    unique_tokens: List[bytes] = []
    unique_tokens_ps: List[float] = []
    for t, p in ps.items():
        unique_tokens.append(t)
        unique_tokens_ps.append(p)
    ps_ngrams = get_ngram_ps(n=n)

    while len(tokens) < max_no_tokens:
        found = False
        for n in range(n, 0, -1):
            ngram = tuple(tokens[-n:])
            maybe_ps: Optional[Dict[bytes, float]] = ps_ngrams.get(ngram, None)
            if maybe_ps:
                ts = []
                probs = []
                for t, p in maybe_ps.items():
                    ts.append(t)
                    probs.append(p)
                next_word = np.random.choice(a=ts, p=probs)
                tokens.append(next_word)
                found = True
                break
        if not found:
            tokens.append(np.random.choice(a=unique_tokens, p=unique_tokens_ps))

    print((b'A HARRY POTTER STORY\n\n' + b''.join(tokens) + b'.\n\nTHE END.').decode('utf-8'))


if __name__ == '__main__':
    main()
