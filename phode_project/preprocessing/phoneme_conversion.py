import re
from nltk.corpus import cmudict

CMU_DICT = cmudict.dict()

def text_to_phonemes(text):
    """
    Convert a text sentence into a flat list of phonemes.
    """

    # 1. Clean text
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)

    # 2. Split into words
    words = text.split()

    phonemes = []

    for word in words:
        if word in CMU_DICT:
            # Use first pronunciation
            word_phonemes = CMU_DICT[word][0]

            # Remove stress numbers (AH0 → AH)
            word_phonemes = [re.sub(r"\d", "", p) for p in word_phonemes]

            phonemes.extend(word_phonemes)

    return phonemes