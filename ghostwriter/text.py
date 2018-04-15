from typing import Iterable, Optional, TextIO, Tuple, List

from collections.__init__ import defaultdict
from cytoolz.itertoolz import concat, sliding_window, take
from numpy import array, zeros


class TokenCodec:
    """
    A token codec is a isomorphism from a set of strings (called the vocabulary) to a set of integer indexes. It always
    has two tokens, a padding token with index 0 and an unknown token with the largest index in the codec. Decoding a
    token not in vocabulary will return the unknown token, Both the padding and unknown tokens count towards the
    vocabulary size.
    """
    PADDING = ""
    PADDING_INDEX = 0
    UNKNOWN = None

    @classmethod
    def create_from_text(cls, tokens: Iterable[str], maximum_vocabulary: Optional[int] = None) -> "TokenCodec":
        """
        Create a codec from a sequence of tokens. The vocabulary will consist of all the tokens. Token indexes are
        ordered by frequency then token order.

        :param tokens: sequence of token from which to build the codec
        :param maximum_vocabulary: optionally clip the vocabulary to this many of the most frequent tokens
        :return: a codec
        """
        token_count = defaultdict(int)
        for token in tokens:
            token_count[token] += 1
        index_to_token = [token for _, token in
                          sorted(((count, token) for token, count in token_count.items()),
                                 key=lambda t: (-t[0], t[1]))[:maximum_vocabulary]]
        return cls(index_to_token)

    def __init__(self, index_to_token: List[str]):
        """
        Create a token codec from a unique list of the strings it can encode.

        :param index_to_token: list of tokens that can be encoded
        """
        if index_to_token[0] != self.PADDING:
            index_to_token.insert(0, self.PADDING)
        assert len(index_to_token) == len(set(index_to_token))
        self.index_to_token = index_to_token
        self.token_to_index = dict((token, index) for index, token in enumerate(self.index_to_token))

    def __repr__(self) -> str:
        return f"Token Codec: vocabulary size {self.vocabulary_size}"

    @property
    def vocabulary_size(self) -> int:
        return len(self.index_to_token) + 1

    def encode(self, tokens: Iterable[str]) -> Iterable[int]:
        return (self.token_to_index.get(token, self.vocabulary_size - 1) for token in tokens)

    def decode(self, indexes: Iterable[int]) -> Iterable[Optional[str]]:
        return (self.decode_token(index) for index in indexes)

    def decode_token(self, index) -> Optional[str]:
        try:
            return self.index_to_token[index]
        except IndexError:
            return self.UNKNOWN


def labeled_language_model_data(codec: TokenCodec, tokens: Iterable[str], context_size: int) -> Tuple[array, array]:
    """
    Transform a sequence of tokens into encoded data that can be used to train a language model.

    :param codec: a codec that encodes strings as integers
    :param tokens: a sequence of tokens
    :param context_size: number of consecutive tokens used to predict the following token
    :return: vectors of context size with labels corresponding to the following tokens
    """

    def vectors_and_labels() -> Iterable[Tuple[array, array]]:
        padding = [codec.PADDING_INDEX] * context_size
        encoded_tokens = codec.encode(tokens)
        for encoding in sliding_window(context_size + 1, concat([padding, encoded_tokens, padding])):
            vector = array(encoding[:-1])
            label = zeros(codec.vocabulary_size)
            label[encoding[-1]] = 1
            yield vector, label

    vectors, labels = zip(*vectors_and_labels())
    vectors = array(vectors)
    samples = array(vectors).shape[0]
    return array(vectors).reshape(samples, context_size, 1), array(labels)


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
