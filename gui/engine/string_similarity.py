
from difflib import SequenceMatcher

def get_similarity(text1, text2) -> float:
    s = SequenceMatcher(None, text1, text2)
    return s.ratio()


import jellyfish

def get_levenshtein_distance(text1, text2)-> int:
    return jellyfish.levenshtein_distance(text1, text2)

def get_jaro_similarity(text1, text2):
    return jellyfish.jaro_similarity(text1, text2)

def get_jaccard_similarity(text1, text2) -> float:
    return jellyfish.jaccard_similarity(text1, text2)