"""Atomic <tp:WORD> tokenization for toki pona on top of a pretrained HF tokenizer.

Every word in categories/typst.toml becomes one added token, registered in the
same order as the CNN classes in ocr-demo/class_mapping.json, so

    token_id = tp_offset + glyph_id

Proper names (capitalized words) are wrapped in <cartouche> ... </cartouche> and
left to the base BPE, e.g.

    "jan Sonja li pona."  ->  <tp:jan><cartouche> Sonja</cartouche><tp:li><tp:pona>.

Non-typst lowercase words (lanpan, oko, ...) also go through the base BPE.
"""

import json
import os
import re
import tomllib

HERE = os.path.dirname(os.path.abspath(__file__))
CLASS_MAPPING = os.path.join(HERE, "..", "ocr-demo", "class_mapping.json")
TYPST = os.path.join(HERE, "categories", "typst.toml")

CART_OPEN, CART_CLOSE = "<cartouche>", "</cartouche>"
WORD_RE = re.compile(r"[A-Za-z]+|[0-9]+|\S")
NO_SPACE_BEFORE = set(".,!?:;)]}…%")
NO_SPACE_AFTER = set("([{")
QUOTES = set("\"“”'‘’«»")
INTERJ_RE = re.compile(r"^(?:a+|n+|m+u+)$")


def tp_word_list():
    """Glyph classes in CNN order (reject class excluded); falls back to typst.toml."""
    if os.path.exists(CLASS_MAPPING):
        with open(CLASS_MAPPING) as f:
            mapping = json.load(f)
        words = [mapping[str(i)]["word"] for i in range(len(mapping))]
        return [w for w in words if w != "not_sitelen_pona"]
    with open(TYPST, "rb") as f:
        return [w.removesuffix(".toml") for w in tomllib.load(f)["wordlist"]["list"]]


def tp_token(word):
    return f"<tp:{word}>"


def cartouche_to_name(glyph_words):
    """Sitelen pona cartouches spell names acrophonically: [soweli olin nasin jan a] -> Sonja."""
    name = "".join(w[0] for w in glyph_words if w)
    return name[:1].upper() + name[1:]


class TokiPonaTokenizer:
    def __init__(self, base_tokenizer, words=None):
        self.base = base_tokenizer
        self.words = words or tp_word_list()
        self.word_set = set(self.words)
        new_tokens = [tp_token(w) for w in self.words] + [CART_OPEN, CART_CLOSE]
        missing = [t for t in new_tokens if t not in self.base.get_vocab()]
        if missing:
            self.base.add_tokens(missing, special_tokens=False)
        self.tp_ids = [self.base.convert_tokens_to_ids(tp_token(w)) for w in self.words]
        self.tp_offset = self.tp_ids[0]
        assert self.tp_ids == list(range(self.tp_offset, self.tp_offset + len(self.words))), \
            "tp tokens must be contiguous to keep token_id = tp_offset + glyph_id"
        self.cart_open_id = self.base.convert_tokens_to_ids(CART_OPEN)
        self.cart_close_id = self.base.convert_tokens_to_ids(CART_CLOSE)
        self.id_to_word = {i: w for i, w in zip(self.tp_ids, self.words)}

    # --------------------------------------------------------- text <-> marked string

    def mark(self, text):
        """toki pona text -> string with <tp:...> and cartouche markers (no inter-word spaces needed)."""
        out = []
        for tok in WORD_RE.findall(text):
            low = tok.lower()
            if tok.isalpha() and low in self.word_set:
                # also catches sentence-initial capitals and ALL-CAPS shouting
                out.append(tp_token(low))
            elif tok.isalpha() and tok[0].isupper() and not INTERJ_RE.match(low):
                out.append(f"{CART_OPEN} {tok}{CART_CLOSE}")
            elif tok.isalnum():
                # community words outside typst.toml, elongated "aaa", numbers
                out.append(" " + tok)
            else:
                out.append(tok)
        return "".join(out)

    def encode(self, text, add_special_tokens=False):
        return self.base(self.mark(text), add_special_tokens=add_special_tokens)["input_ids"]

    def word_spans(self, text):
        """Per input word: (word, [token indices]) so attention can be pooled to word level."""
        ids = self.encode(text)
        spans, cur = [], None
        in_cart = False
        for k, t in enumerate(ids):
            if t in self.id_to_word:
                spans.append((self.id_to_word[t], [k]))
            elif t == self.cart_open_id:
                in_cart, cur = True, ["", []]
                cur[1].append(k)
            elif t == self.cart_close_id and cur is not None:
                cur[1].append(k)
                spans.append((cur[0].strip(), cur[1]))
                in_cart, cur = False, None
            elif in_cart:
                cur[0] += self.base.decode([t])
                cur[1].append(k)
            else:
                piece = self.base.decode([t])
                if piece.startswith(" ") or not spans or not piece.strip().isalnum():
                    spans.append((piece.strip(), [k]))
                else:
                    spans[-1] = (spans[-1][0] + piece, spans[-1][1] + [k])
        return ids, [s for s in spans if s[0]]

    def decode(self, ids):
        """Inverse of encode: rebuild plain toki pona text with normal spacing."""
        pieces = []  # (text, is_word)
        buf = []

        def flush():
            if buf:
                pieces.extend(_split_bpe(self.base.decode(buf)))
                buf.clear()

        in_cart = False
        for t in ids:
            if t in self.id_to_word:
                flush()
                pieces.append((self.id_to_word[t], True))
            elif t == self.cart_open_id:
                flush()
                in_cart = True
            elif t == self.cart_close_id:
                name = self.base.decode(buf).strip()
                buf.clear()
                in_cart = False
                if name:
                    pieces.append((name[:1].upper() + name[1:], True))
            else:
                buf.append(t)
        if not in_cart:
            flush()
        return _join(pieces)


def _split_bpe(text):
    return [(m, m[0].isalnum()) for m in WORD_RE.findall(text)]


def _join(pieces):
    out = ""
    open_quote = False
    prev = None
    for text, is_word in pieces:
        if not out:
            out = text
        elif text in QUOTES:
            if open_quote:
                out += text
            else:
                out += " " + text
            open_quote = not open_quote
            prev = text
            continue
        elif text in NO_SPACE_BEFORE or text[0] in NO_SPACE_BEFORE:
            out += text
        elif prev in NO_SPACE_AFTER or (prev in QUOTES and open_quote):
            out += text
        else:
            out += " " + text
        if text in QUOTES:
            open_quote = not open_quote
        prev = text
    return out
