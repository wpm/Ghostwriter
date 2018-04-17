import pickle
from copy import copy
from operator import attrgetter
from typing import Iterable, Optional, TextIO, Tuple, List, Set, Union, Any, Sequence

import spacy
from collections.__init__ import defaultdict
from cytoolz.itertoolz import concat, sliding_window, take
from numpy import array, zeros, concatenate
from spacy.language import Language
from spacy.tokens import Doc
from spacy.vocab import Vocab


class Token:
    @classmethod
    def create(cls, token: Any) -> "Token":
        if not isinstance(token, cls):
            token = cls(str(token))
        return token

    @classmethod
    def meta(cls, token: str) -> "Token":
        return cls(token, True)

    def __init__(self, token: str, is_meta: bool = False):
        self.token = token
        self.is_meta = is_meta

    def __repr__(self) -> str:
        return self.token

    def __hash__(self):
        return hash((self.token, self.is_meta))

    def __eq__(self, other: Union["Token", str]) -> bool:
        # Note that you cannot have Token and strings as keys in the same dictionary.
        if isinstance(other, str):
            return not self.is_meta and self.token == other
        else:
            return (self.token, self.is_meta) == (other.token, other.is_meta)

    def __lt__(self, other: "Token") -> bool:
        return (self.token, self.is_meta) < (other.token, other.is_meta)


class TokenCodec:
    """
    A token codec is a isomorphism from a set of tokens (called the vocabulary) to a set of integer indexes. It always
    has two meta tokens, a padding token and an out-of-vocabulary token. Decoding a index not in vocabulary will return
    the out-of-vocabulary token.
    """
    OOV = Token.meta("-OOV-")
    PAD = Token.meta("-PAD-")

    @classmethod
    def create_from_tokens(cls, tokens: Iterable, maximum_vocabulary: Optional[int] = None) -> "TokenCodec":
        """
        Create a codec from a sequence of tokens. The vocabulary will consist of all the tokens. Token indexes are
        ordered by frequency then token order.

        :param tokens: sequence of token from which to build the codec
        :param maximum_vocabulary: optionally clip the vocabulary to this many of the most frequent tokens
        :return: a codec
        """
        token_count = defaultdict(int)
        for token in tokens:
            token = Token.create(token)
            token_count[token] += 1
        index_to_token = [token for _, token in
                          sorted(((count, token) for token, count in token_count.items()),
                                 key=lambda t: (-t[0], t[1]))[:maximum_vocabulary]]
        return cls(index_to_token)

    def __init__(self, index_to_token: List[Token], meta_tokens: Set[str] = ()):
        """
        Create a codec from a unique list of the tokens it can encode.

        :param index_to_token: list of tokens that can be encoded
        :param meta_tokens: option meta-tokens to add to the vocabulary
        """
        self.meta_tokens = [self.PAD, self.OOV] + [Token.meta(token) for token in sorted(meta_tokens)]
        self.index_to_token = self.meta_tokens + index_to_token
        self.token_to_index = dict((token, index) for index, token in enumerate(self.index_to_token))
        assert len(self.index_to_token) == len(set(self.index_to_token))

    @property
    def oov_index(self) -> int:
        return self.token_to_index[self.OOV]

    @property
    def pad_index(self) -> int:
        return self.token_to_index[self.PAD]

    @property
    def vocabulary_size(self) -> int:
        return len(self.index_to_token)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}: vocabulary size {self.vocabulary_size}"

    def encode(self, tokens: Iterable) -> Iterable[int]:
        for token in tokens:
            if not isinstance(token, Token):
                token = Token.create(token)
            yield self.token_to_index.get(token, self.oov_index)

    def decode(self, indexes: Iterable[int]) -> Iterable[Token]:
        return (self.decode_index(index) for index in indexes)

    def decode_index(self, index: int) -> Token:
        try:
            return self.index_to_token[index]
        except IndexError:
            return self.OOV

    def __contains__(self, token: Token) -> bool:
        return token in self.token_to_index


class GloVeCodec(TokenCodec):

    def __init__(self, vocabulary: Vocab, maximum_vocabulary: Optional[int] = None, meta_tokens: Set[str] = ()):
        lexemes = sorted([lexeme for lexeme in vocabulary if lexeme.has_vector],
                         key=attrgetter("rank"))[:maximum_vocabulary]
        super().__init__([Token(lexeme.orth_) for lexeme in lexemes], meta_tokens)
        n = len(self.meta_tokens)
        self.rank_to_index = dict((lexeme.rank, index) for index, lexeme in enumerate(lexemes, n))
        self.embedding_matrix = zeros((self.vocabulary_size, vocabulary.vectors_length))
        for index, lexeme in enumerate(lexemes, n):
            self.embedding_matrix[index] = lexeme.vector


def labeled_words_in_sentences_language_model_data(codec: GloVeCodec, document: Doc, context_size: int) \
        -> Tuple[array, array]:
    eos = Token.meta("-EOS-")
    assert eos in codec
    vectors = []
    labels = []
    for sentence in document.sents:
        tokens = [Token(token.orth_) for token in sentence] + [eos]
        sentence_vectors, sentence_labels = labeled_language_model_data(codec, tokens, context_size)
        vectors.append(sentence_vectors)
        labels.append(sentence_labels)
    return concatenate(vectors), concatenate(labels)


def labeled_language_model_data(codec: TokenCodec, tokens: Iterable, context_size: int) -> Tuple[array, array]:
    """
    Transform a sequence of tokens into encoded data that can be used to train a language model.

    :param codec: a codec that encodes strings as integers
    :param tokens: a sequence of tokens
    :param context_size: number of consecutive tokens used to predict the following token
    :return: vectors of context size with labels corresponding to the following tokens
    """

    def vectors_and_labels() -> Iterable[Tuple[array, array]]:
        padding = [codec.pad_index] * context_size
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


class Tokenizer:
    def __init__(self, codec: TokenCodec, context_size: int):
        self.codec = codec
        self.context_size = context_size

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}: vocabulary size {self.vocabulary_size}, context size {self.context_size}"

    @property
    def vocabulary_size(self) -> int:
        return self.codec.vocabulary_size

    def unencoded_from_documents(self, documents: Iterable[str]) -> Iterable[Tuple[Sequence[Token], Token]]:
        raise NotImplemented

    def encoded_training_set_from_documents(self, documents: Iterable[str]) -> Tuple[array, array]:

        def vectors_and_labels() -> Iterable[Tuple[array, array]]:
            for context, token in self.unencoded_from_documents(documents):
                vector = array(list(self.codec.encode(context)))
                label = zeros(self.vocabulary_size)
                encoded_token = list(self.codec.encode([token]))[0]
                label[encoded_token] = 1
                yield vector, label

        vectors, labels = zip(*vectors_and_labels())
        vectors = array(vectors)
        samples = array(vectors).shape[0]
        return array(vectors).reshape(samples, self.context_size, 1), array(labels)

    @classmethod
    def deserialize(cls, f: TextIO) -> "Tokenizer":
        return pickle.load(f)

    def serialize(self, f: TextIO):
        pickle.dump(self, f)

    def context_and_token(self, tokens: Iterable[Token]) -> Iterable[Tuple[Sequence[Token], Token]]:
        padding = [self.codec.PAD] * self.context_size
        for window in sliding_window(self.context_size + 1, concat([padding, tokens, padding])):
            yield (window[:-1], window[-1])


class CharacterTokenizer(Tokenizer):
    """
    Iterate over a list of text files, returning all the characters in them. Once each file is exhausted its pointer is
    reset to the head of the file, so this tokenizer can be multiple times in a row and return the same results.
    """

    @classmethod
    def create_from_documents(cls, documents: Iterable[str], context_size: int,
                              maximum_vocabulary: Optional[int] = None) -> "CharacterTokenizer":
        codec = TokenCodec.create_from_tokens(cls.characters_from_documents(documents), maximum_vocabulary)
        return cls(codec, context_size)

    def unencoded_from_documents(self, documents: Iterable[str]) -> Iterable[Tuple[Sequence[Token], Token]]:
        return self.context_and_token(self.characters_from_documents(documents))

    @staticmethod
    def characters_from_documents(documents: Iterable[str]) -> Iterable[Token]:
        for document in documents:
            for character in document:
                yield Token(character)


class SentenceTokenizer(Tokenizer):
    """
    Iterate over a list of text files, using spaCy to divide them into sentences. Append a special -EOS- token to the
    end of each sentence and generate contexts and following tokens from the sentences.
    """

    def __init__(self, nlp: Language, nlp_name: str, context_size: int, maximum_vocabulary: Optional[int] = None):
        self.nlp = nlp
        self.nlp_name = nlp_name
        self.context_size = context_size
        self.maximum_vocabulary = maximum_vocabulary
        codec = GloVeCodec(self.nlp.vocab, self.maximum_vocabulary, {"-EOS-"})
        super().__init__(codec, context_size)

    def unencoded_from_documents(self, documents: Iterable[str]) -> Iterable[Tuple[Sequence[Token], Token]]:
        eos = Token.meta("-EOS-")
        for document in self.nlp.pipe(documents):
            for sentence in document.sents:
                for context, token in self.context_and_token([Token(t.orth_) for t in sentence] + [eos]):
                    yield context, token

    def __getstate__(self):
        d = copy(self.__dict__)
        del d["nlp"]
        return d

    def __setstate__(self, d):
        d["nlp"] = spacy.load(d["nlp_name"])
        self.__dict__ = d


def documents_from_text_files(text_files: Iterable[TextIO]) -> Iterable[str]:
    """
    Iterate over a list of text files, returning their contents. Once each file is exhausted its pointer is reset to the
    head of the file, so this function can be multiple times in a row and return the same results.

    :param text_files: open text file handles
    :return: the context of each text file
    """
    for text_file in text_files:
        yield text_file.read()
        text_file.seek(0)


def characters_from_text_files(text_files: Iterable[TextIO], n: Optional[int] = None) -> Iterable[str]:
    """
    Iterate over a list of text files, returning all the characters in them. Optionally only return the first n
    characters in the set of files. Once each file is exhausted its pointer is reset to the head of the file, so this
    function can be multiple times in a row and return the same results.

    :param text_files: open text file handles
    :param n: the number of characters to take, or take all if this is None
    :return: the first n characters in the text files
    """
    characters = concat(characters_from_text_file(text_file) for text_file in text_files)
    if n is not None:
        characters = take(n, characters)
    return characters


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

    characters = cs()
    text_file.seek(0)
    return characters
