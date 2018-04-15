import logging
import os
import sys
from io import StringIO
from pickle import load, dump
from random import choices
from typing import Optional, Iterable, Sequence

from numpy import array, ma, roll

from ghostwriter import __version__
from ghostwriter.text import TokenCodec, labeled_language_model_data


class LanguageModel:
    """
    An LSTM language model.
    """

    @classmethod
    def create(cls, hidden: int, context_size: int, dropout: float, codec: TokenCodec) -> "LanguageModel":
        from keras import Sequential
        from keras.layers import LSTM, Dropout, Dense

        if codec.vocabulary_size == 0:
            logging.warning("Creating a model with zero-sized codec.")
        model = Sequential()
        model.add(LSTM(hidden, input_shape=(context_size, 1)))
        model.add(Dropout(dropout))
        model.add(Dense(codec.vocabulary_size, activation="softmax"))
        model.compile(loss="categorical_crossentropy", optimizer="adam")
        language_model = cls(model, codec)
        return language_model

    @classmethod
    def load(cls, directory) -> "LanguageModel":
        from keras.models import load_model
        try:
            model = load_model(cls.model_path(directory))
            with open(cls.codec_path(directory), "rb") as f:
                codec = load(f)
        except IOError:
            raise ValueError(f"Cannot read language model from {directory}")
        return cls(model, codec)

    def __init__(self, model, codec: TokenCodec):
        self.model = model
        self.codec = codec

    def __repr__(self):
        return f"Language Model: {self.hidden_nodes} hidden nodes, {self.codec}"

    def __str__(self):
        def model_topology():
            old_stdout = sys.stdout
            sys.stdout = s = StringIO()
            self.model.summary()
            sys.stdout = old_stdout
            return s.getvalue()

        return "%s\n\n%s\nVersion %s" % (repr(self), model_topology(), __version__)

    def save(self, directory: str):
        os.makedirs(directory)
        if not os.path.exists(self.model_path(directory)):
            self.model.save(self.model_path(directory))
        with open(self.codec_path(directory), "wb") as f:
            dump(self.codec, f)

    def train(self, tokens: Iterable[str], epochs: int, model_directory: Optional[str]):
        from keras.callbacks import ProgbarLogger, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

        vectors, labels = labeled_language_model_data(self.codec, tokens, self.context_size)
        callbacks = [ProgbarLogger(), EarlyStopping(monitor="loss"), ReduceLROnPlateau(monitor="loss")]
        if model_directory is not None:
            callbacks.append(ModelCheckpoint(self.model_path(model_directory)))
        history = self.model.fit(vectors, labels, epochs=epochs, callbacks=callbacks)
        if model_directory is not None:
            with open(self.history_path(model_directory), "wb") as f:
                dump({"epoch": history.epoch, "history": history.history, "params": history.params}, f)
        return history

    def perplexity(self, tokens: Iterable[str]) -> float:
        vectors, labels = labeled_language_model_data(self.codec, tokens, self.context_size)
        predictions = self.model.predict(vectors)
        n = predictions.shape[0]
        return 2 ** (-ma.log2(predictions * labels).filled(0).sum() / n)

    def generate(self, seed: Sequence[str]) -> Iterable[str]:
        context = list(self.codec.encode(seed))
        context = ([self.codec.PADDING_INDEX] * (self.context_size - len(context)) + context)[-self.context_size:]
        context = array(context).reshape(1, self.context_size, 1)
        while True:
            predicted_distribution = self.model.predict(context).reshape(self.vocabulary_size)
            sample = choices(range(self.vocabulary_size), weights=predicted_distribution)[0]
            yield self.codec.decode_token(sample)
            context = roll(context, -1, axis=1)
            context[-1][-1][0] = sample

    @property
    def context_size(self) -> int:
        return self.model.inputs[0].shape[1].value

    @property
    def vocabulary_size(self) -> int:
        return self.codec.vocabulary_size

    @property
    def hidden_nodes(self) -> int:
        return self.model.layers[0].output_shape[1]

    @staticmethod
    def model_path(directory: str):
        return os.path.join(directory, "model.h5")

    @staticmethod
    def codec_path(directory: str):
        return os.path.join(directory, "codec.pk")

    @staticmethod
    def history_path(directory: str):
        return os.path.join(directory, "history.pk")
