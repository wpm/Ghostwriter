import os
from pickle import load, dump
from random import choices
from typing import Iterable, Tuple, TextIO, Optional, Sequence

from collections.__init__ import defaultdict
from cytoolz import take
from cytoolz.itertoolz import sliding_window, concat
from numpy import ma, roll
from numpy.core.multiarray import array, zeros


class Codec:
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


class LanguageModel:
    @classmethod
    def create(cls, hidden: int, context_size: int, dropout: float, codec: Codec) -> "LanguageModel":
        from keras import Sequential
        from keras.layers import LSTM, Dropout, Dense

        model = Sequential()
        model.add(LSTM(hidden, input_shape=(context_size, 1)))
        model.add(Dropout(dropout))
        model.add(Dense(codec.vocabulary_size, activation="softmax"))
        model.compile(loss="categorical_crossentropy", optimizer="adam")
        return cls(model, codec)

    @classmethod
    def load(cls, directory) -> "LanguageModel":
        from keras.models import load_model
        model = load_model(cls.model_path(directory))
        with open(cls.codec_path(directory), "rb") as f:
            codec = load(f)
        return cls(model, codec)

    def __init__(self, model, codec: Codec):
        self.model = model
        self.codec = codec

    def save(self, directory: str):
        os.makedirs(directory)
        if not os.path.exists(self.model_path(directory)):
            self.model.save(self.model_path(directory))
        with open(self.codec_path(directory), "wb") as f:
            dump(self.codec, f)

    def labeled_data(self, tokens: Iterable[str]) -> Tuple[array, array]:
        def vectors_and_labels() -> Iterable[Tuple[array, array]]:
            padding = [self.codec.PADDING_INDEX] * self.context_size
            encoded_tokens = self.codec.encode(tokens)
            for encoding in sliding_window(self.context_size + 1, concat([padding, encoded_tokens, padding])):
                vector = array(encoding[:-1])
                label = zeros(self.codec.vocabulary_size)
                label[encoding[-1]] = 1
                yield vector, label

        vectors, labels = zip(*vectors_and_labels())
        vectors = array(vectors)
        samples = array(vectors).shape[0]
        return array(vectors).reshape(samples, self.context_size, 1), array(labels)

    def perplexity(self, tokens: Iterable[str]) -> float:
        vectors, labels = self.labeled_data(tokens)
        predictions = self.model.predict(vectors)
        n = predictions.shape[0]
        return 2 ** (-ma.log2(predictions * labels).filled(0).sum() / n)

    def generate(self, seed: Sequence[str]) -> Iterable[str]:
        context = list(self.codec.encode(seed))
        context = array([self.codec.PADDING_INDEX] * (self.context_size - len(context)))
        while True:
            predicted_distribution = self.model.predict(context)
            sample = choices(range(self.codec.vocabulary_size), weights=predicted_distribution)
            yield self.codec.decode_token(sample)
            context = roll(context, -1)
            context[-1] = sample

    @property
    def context_size(self) -> int:
        return self.model.inputs[0].shape[1].value

    @staticmethod
    def model_path(directory: str):
        return os.path.join(directory, "model.h5")

    @staticmethod
    def codec_path(directory: str):
        return os.path.join(directory, "codec.pk")


def characters(data: TextIO, n: Optional[int] = None) -> Iterable[str]:
    def cs() -> Iterable[str]:
        for line in data:
            for c in list(line):
                yield c

    tokens = cs()
    if n is not None:
        tokens = take(n, tokens)
    data.seek(0)
    return tokens