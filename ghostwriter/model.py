import json
import logging
import os
import sys
from io import StringIO
from json import JSONDecodeError
from pickle import load, dump
from random import choices
from typing import Optional, Iterable, Sequence, Mapping, List

from cytoolz.itertoolz import concat
from numpy import array, ma, roll

from ghostwriter import __version__
from ghostwriter.text import TokenCodec, labeled_language_model_data


class TrainingHistory:
    @classmethod
    def from_keras_history(cls, history) -> "TrainingHistory":
        # Map to float so that these numbers can be serialized in JSON.
        h = dict((k, [float(x) for x in history.history[k]]) for k in history.history)
        return cls([history.epoch], [h], [history.params])

    def __init__(self, epoch: List[List[int]] = (), history: List[Mapping[str, Sequence[float]]] = (),
                 params: List[Mapping] = ()):
        self.epoch = list(epoch)
        self.history = list(history)
        self.params = list(params)

    def __repr__(self) -> str:
        s = f"History: {self.iterations} iterations"
        if self.final_loss is not None:
            s += f", final loss {self.final_loss:0.4f}"
        return s

    def __add__(self, other: "TrainingHistory") -> "TrainingHistory":
        return TrainingHistory(self.epoch + other.epoch, self.history + other.history, self.params + other.params)

    @property
    def iterations(self) -> int:
        return len(list(concat(self.epoch)))

    @property
    def final_loss(self) -> Optional[float]:
        try:
            return self.history[-1]["loss"][-1]
        except IndexError:
            return None

    def to_json(self, filename: str):
        with open(filename, "w") as f:
            json.dump({"epoch": self.epoch, "history": self.history, "params": self.params},
                      f, indent=4, sort_keys=True)

    @classmethod
    def from_json(cls, filename: str) -> "TrainingHistory":
        with open(filename) as f:
            h = json.load(f)
        return cls(h["epoch"], h["history"], h["params"])


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

    def __init__(self, model, codec: TokenCodec, history: TrainingHistory = TrainingHistory()):
        self.model = model
        self.codec = codec
        self.history = history

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

    def train(self, tokens: Iterable[str], epochs: int, directory: Optional[str]):
        from keras.callbacks import ProgbarLogger, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

        self.save(directory)
        vectors, labels = labeled_language_model_data(self.codec, tokens, self.context_size)
        callbacks = [ProgbarLogger(), EarlyStopping(monitor="loss"), ReduceLROnPlateau(monitor="loss")]
        if directory is not None:
            callbacks.append(ModelCheckpoint(self.model_path(directory)))
        if self.history.iterations < epochs:
            history = self.model.fit(vectors, labels,
                                     epochs=epochs, initial_epoch=self.history.iterations,
                                     callbacks=callbacks)
            self.history += TrainingHistory.from_keras_history(history)
            self.save(directory)
        return self.history

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

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        if not os.path.exists(self.codec_path(directory)):
            with open(self.codec_path(directory), "wb") as f:
                dump(self.codec, f)
        if not os.path.exists(self.model_path(directory)):
            self.model.save(self.model_path(directory))
        self.history.to_json(self.history_path(directory))

    @classmethod
    def load(cls, directory) -> "LanguageModel":
        from keras.models import load_model
        try:
            model = load_model(cls.model_path(directory))
            with open(cls.codec_path(directory), "rb") as f:
                codec = load(f)
            history = TrainingHistory.from_json(cls.history_path(directory))
        except (IOError, JSONDecodeError):
            raise ValueError(f"Cannot read language model from {directory}")
        return cls(model, codec, history)

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
        return os.path.join(directory, "history.json")
