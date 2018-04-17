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
from ghostwriter.text import Token, Tokenizer


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
    def create(cls, tokenizer: Tokenizer, hidden: int, dropout: float) -> "LanguageModel":
        from keras import Sequential
        from keras.layers import LSTM, Dropout, Dense

        if tokenizer.vocabulary_size == 0:
            logging.warning("Creating a model using a codec with an empty vocabulary.")
        model = Sequential()
        model.add(LSTM(hidden, input_shape=(tokenizer.context_size, 1)))
        model.add(Dropout(dropout))
        model.add(Dense(tokenizer.vocabulary_size, activation="softmax"))
        model.compile(loss="categorical_crossentropy", optimizer="adam")
        return cls(model, tokenizer)

    def __init__(self, model, tokenizer: Tokenizer, history: TrainingHistory = TrainingHistory()):
        self.model = model
        self.tokenizer = tokenizer
        self.history = history

    def __repr__(self):
        return f"Language Model: {self.hidden_nodes} hidden nodes, {self.tokenizer}"

    def __str__(self):
        def model_topology():
            old_stdout = sys.stdout
            sys.stdout = s = StringIO()
            self.model.summary()
            sys.stdout = old_stdout
            return s.getvalue()

        return "%s\n\n%s\nGhostwriter version %s" % (repr(self), model_topology(), __version__)

    def train(self, documents: Iterable[str], epochs: int, directory: Optional[str], progress_bar: bool = True):
        from keras.callbacks import ProgbarLogger, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

        self.save(directory)
        vectors, labels = self.tokenizer.encoded_training_set_from_documents(documents)
        callbacks = [EarlyStopping(monitor="loss"), ReduceLROnPlateau(monitor="loss")]
        if directory is not None:
            callbacks.append(ModelCheckpoint(self.model_path(directory)))
        if progress_bar:
            callbacks.append(ProgbarLogger())
        if self.history.iterations < epochs:
            history = self.model.fit(vectors, labels,
                                     epochs=epochs, initial_epoch=self.history.iterations,
                                     verbose=int(progress_bar), callbacks=callbacks)
            self.history += TrainingHistory.from_keras_history(history)
            self.save(directory)
        return self.history

    def perplexity(self, documents: Iterable[str]) -> float:
        vectors, labels = self.tokenizer.encoded_training_set_from_documents(documents)
        predictions = self.model.predict(vectors)
        n = predictions.shape[0]
        return 2 ** (-ma.log2(predictions * labels).filled(0).sum() / n)

    def generate(self, seed: Sequence[str]) -> Iterable[Token]:
        context = list(self.tokenizer.codec.encode(seed))
        context = ([self.tokenizer.codec.pad_index] * (self.context_size - len(context)) + context)[-self.context_size:]
        context = array(context).reshape(1, self.context_size, 1)
        while True:
            predicted_distribution = self.model.predict(context).reshape(self.vocabulary_size)
            sample = choices(range(self.vocabulary_size), weights=predicted_distribution)[0]
            yield self.tokenizer.codec.decode_index(sample)
            context = roll(context, -1, axis=1)
            context[-1][-1][0] = sample

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        with open(self.description_path(directory), "w") as f:
            f.write(str(self) + "\n")
        if not os.path.exists(self.tokenizer_path(directory)):
            with open(self.tokenizer_path(directory), "wb") as f:
                dump(self.tokenizer, f)
        if not os.path.exists(self.model_path(directory)):
            self.model.save(self.model_path(directory))
        self.history.to_json(self.history_path(directory))

    @classmethod
    def load(cls, directory) -> "LanguageModel":
        from keras.models import load_model
        try:
            model = load_model(cls.model_path(directory))
        except IOError:
            raise ValueError(f"Cannot read the model from {cls.model_path(directory)}")
        try:
            with open(cls.tokenizer_path(directory), "rb") as f:
                tokenizer = load(f)
        except (IOError, OSError):
            raise ValueError(f"Cannot read the tokenizer from {cls.tokenizer_path(directory)}")
        try:
            history = TrainingHistory.from_json(cls.history_path(directory))
        except (IOError, JSONDecodeError):
            raise ValueError(f"Cannot read history from {cls.history_path(directory)}")
        return cls(model, tokenizer, history)

    @property
    def context_size(self) -> int:
        return self.model.inputs[0].shape[1].value

    @property
    def vocabulary_size(self) -> int:
        return self.tokenizer.vocabulary_size

    @property
    def hidden_nodes(self) -> int:
        return self.model.layers[0].output_shape[1]

    @staticmethod
    def description_path(directory: str):
        return os.path.join(directory, "description.txt")

    @staticmethod
    def model_path(directory: str):
        return os.path.join(directory, "model.h5")

    @staticmethod
    def tokenizer_path(directory: str):
        return os.path.join(directory, "tokenizer.pk")

    @staticmethod
    def history_path(directory: str):
        return os.path.join(directory, "history.json")
