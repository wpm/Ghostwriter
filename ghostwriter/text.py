from typing import Iterable, Optional, TextIO

from collections.__init__ import defaultdict
from cytoolz import take
from cytoolz.itertoolz import concat


class Codec:
    """
    A codec is a isomorphism from a set of strings (called the vocabulary) to a set of integer indexes. The empty string
    is called a padding token and is always mapping to index 0.
    """
    PADDING_TOKEN = ""
    PADDING_INDEX = 0

    def __init__(self, tokens: Iterable[str], maximum_vocabulary: Optional[int] = None):
        token_count = defaultdict(int)
        for token in tokens:
            token_count[token] += 1
        self.index_to_token = [self.PADDING_TOKEN] + \
                              [token for _, token in
                               sorted(((count, token) for token, count in token_count.items()),
                                      reverse=True)[:maximum_vocabulary]]
        self.token_to_index = dict((token, index) for index, token in enumerate(self.index_to_token))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}: vocabulary size {self.vocabulary_size}"

    def encode(self, tokens: Iterable[str]) -> Iterable[Optional[int]]:
        return (self.token_to_index.get(token, None) for token in tokens)

    def decode(self, indexes: Iterable[int]) -> Iterable[Optional[str]]:
        return (self.decode_token(index) for index in indexes)

    def decode_token(self, index) -> Optional[str]:
        try:
            return self.index_to_token[index]
        except IndexError:
            return None

    @property
    def vocabulary_size(self) -> int:
        return len(self.index_to_token)


def characters_from_text_files(text_files: Iterable[TextIO], n: Optional[int] = None) -> Iterable[str]:
    """
    Iterate over a list of text files, returning all the characters in them. Optionally only return the first n
    characters in the set of files. Once each file is exhausted its pointer is reset to the head of the file, so this
    function can be multiple times in a row and return the same results.

    :param text_files: open text file handles
    :param n: the number of characters to take, or take all if this is None
    :return: the first n characters in the text files
    """
    tokens = concat(characters_from_text_file(text_file) for text_file in text_files)
    if n is not None:
        tokens = take(n, tokens)
    return tokens


def characters_from_text_file(text_file: TextIO) -> Iterable[str]:
    """
    Iterate over all the characters in a text file and then reset the pointer back to the start of the file.

    :param text_file: open text file handle
    :return: the characters in the text file
    """

    def cs() -> Iterable[str]:
        for line in text_file:
            for c in list(line):
                yield c

    tokens = cs()
    text_file.seek(0)
    return tokens
