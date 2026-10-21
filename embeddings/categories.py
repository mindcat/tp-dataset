"""Semantic categories from Huang & Wu (2026), "Examining the Limits of Word2Vec with Toki Pona"
(SCiL 2026). Vendored verbatim from https://github.com/cubedhuang/toki-pona-word2vec
(v1.0, categories.py, CC-BY-4.0; doi:10.5281/zenodo.20297806). Do not edit by hand.
"""

CATEGORIES = {
    # CONCRETE ENTITIES
    "colors": {"laso", "loje", "jelo", "walo", "pimeja", "kule"},
    "body": {
        "lawa",
        "luka",
        "noka",
        "sinpin",
        "monsi",
        "poka",
        "sijelo",
        "selo",
        "uta",
        "nena",
    },
    "animals": {"soweli", "waso", "kala", "akesi", "pipi", "kijetesantakalu"},
    "plants_and_food": {"kasi", "kili", "pan", "soko", "moku", "namako", "suwi"},
    "kinship": {"mije", "meli", "tonsi", "olin", "kulupu", "mama", "unpa", "jan"},
    "person": {"jan", "kulupu"},
    "place": {"ma", "tomo"},
    "substance": {"kiwen", "ko", "telo", "kon"},
    "artifact": {"ilo", "len", "poki"},
    # ABSTRACT CONCEPTS
    "quantity": {"wan", "tu", "luka", "mute", "ale", "nanpa", "kipisi"},
    "size": {"suli", "lili", "mute"},
    "time.relative": {"tenpo", "pini", "open", "kama"},
    "time.period": {"tenpo", "sike", "suno", "mun"},
    "cognition": {"toki", "nimi", "sona", "sitelen", "lipu", "nasin"},
    # SPACE
    "spatial": {"sinpin", "monsi", "anpa", "sewi", "poka", "insa"},
    "movement": {"tawa", "weka", "tan", "kama", "awen"},
    "shapes": {"linja", "leko", "palisa", "sike", "lupa", "nena", "supa"},
    # ACTIONS
    "exchange": {"pali", "pana", "esun", "jo", "mani"},
    "conflict": {"utala", "pakala", "moli", "alasa", "monsuta"},
    "action": {"pilin", "lape", "olin", "musi", "wile"},
    "sense.visual": {"lukin", "sitelen", "lipu"},
    "sense.auditory": {"kalama", "kute", "mu"},
    # QUALITIES
    "evaluative.positive": {"pona", "suli", "wawa", "suwi"},
    "evaluative.negative": {"jaki", "nasa", "monsuta", "ike"},
    "temperature": {"seli", "lete"},
    "relational": {"ante", "sama", "sin"},
    "medicine": {"misikeke"},
}

CATEGORIES_EN = {
    "colors": ["red", "blue", "yellow", "green", "black", "white"],
    "body": ["head", "arm", "leg", "face", "back", "body", "skin", "mouth", "nose"],
    "animals": ["mammal", "bird", "fish", "reptile", "insect", "animal"],
    "plants_and_food": ["plant", "fruit", "bread", "mushroom", "food", "sugar"],
    "kinship": ["man", "woman", "love", "parent", "brother", "sister", "sex"],
    "person": ["person", "human", "community", "group"],
    "place": ["land", "country", "house", "room"],
    "substance": ["stone", "metal", "powder", "water", "air", "gas"],
    "artifact": ["tool", "machine", "cloth", "container", "box"],
    "quantity": ["one", "two", "five", "many", "all", "number", "part"],
    "size": ["big", "small", "long", "short"],
    "time.relative": ["time", "past", "future", "start", "finish"],
    "time.period": ["year", "day", "month", "era"],
    "cognition": ["language", "word", "knowledge", "image", "book", "way", "method"],
    "spatial": ["front", "back", "bottom", "top", "side", "inside", "center"],
    "movement": ["go", "move", "leave", "come", "stay", "wait"],
    "shapes": ["line", "square", "block", "rod", "circle", "hole", "bump", "surface"],
    "exchange": ["work", "make", "give", "buy", "sell", "money"],
    "conflict": ["fight", "war", "break", "kill", "hunt", "monster", "fear"],
    "action": ["feel", "sleep", "play", "want", "desire"],
    "sense.visual": ["see", "look", "image", "picture"],
    "sense.auditory": ["sound", "hear", "listen", "noise"],
    "evaluative.positive": ["good", "important", "awesome", "cute"],
    "evaluative.negative": ["bad", "weak", "gross", "odd", "scary"],
    "temperature": ["hot", "warm", "cold", "cool"],
    "relational": ["different", "other", "same", "new"],
    "medicine": ["medicine", "cure", "doctor"],
}


CATEGORIES_EXCLUDED = {
    # excluded because of unique grammatical function
    "li",
    "e",
    "la",
    "pi",
    "en",
    "o",
    "anu",
    "a",
    "n",
    "taso",
    "kin",
    "ala",
    # pronouns
    "mi",
    "sina",
    "ona",
    "ni",
    "seme",
    # excluded for being too general
    "ijo",
    # modal verbs and prepositions
    "ken",
    "lon",
    "kepeken",
    # excluded for being named entities
    "pu",
    "ku",
}

ALL_CATEGORIZED_WORDS = set().union(*CATEGORIES.values())
