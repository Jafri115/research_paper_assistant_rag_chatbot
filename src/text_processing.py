import re
from typing import Iterable

# Basic English stopwords (small set to avoid extra dependency); extend if needed
BASIC_STOPWORDS = {
    'the','and','a','an','of','in','to','is','are','for','on','with','that','this','by','from','at','as','it','be','or','we','can','our','their','these','those','using','used'
}

LATEX_EQ_RE = re.compile(r'\$\$.*?\$\$|\$[^$]*\$', re.DOTALL)
URL_RE = re.compile(r'https?://\S+|www\.\S+')
MULTI_WS_RE = re.compile(r'\s+')
INLINE_LATEX_CMD_RE = re.compile(r'\\(?:cite|ref|label|eqref|begin|end|textbf|emph|mathrm|mathbb)\{[^}]*\}')


def remove_latex(text: str) -> str:
    text = LATEX_EQ_RE.sub(' ', text)
    text = INLINE_LATEX_CMD_RE.sub(' ', text)
    return text


def remove_urls(text: str) -> str:
    return URL_RE.sub(' ', text)


def normalize_whitespace(text: str) -> str:
    return MULTI_WS_RE.sub(' ', text).strip()


def strip_stopwords(tokens: Iterable[str]) -> str:
    return ' '.join(t for t in tokens if t not in BASIC_STOPWORDS)


def clean_text(text: str, lowercase: bool = False, remove_stopwords: bool = False) -> str:
    if not text:
        return ''
    t = remove_urls(text)
    t = remove_latex(t)
    if lowercase:
        t = t.lower()
    # Tokenize very simply on whitespace after basic cleanup
    t = normalize_whitespace(t)
    if remove_stopwords:
        tokens = t.split()
        t = strip_stopwords(tokens)
    return t
