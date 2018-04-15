import os
import unittest
from collections import Iterator

from click.testing import CliRunner

from ghostwriter.command import ghostwriter
from ghostwriter.model import LanguageModel
from ghostwriter.text import characters_from_text_files, TokenCodec


class TestGhostwriter(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self.kakfa = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kafka.txt")

    def test_train_and_perplexity(self):
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(ghostwriter, ["train", self.kakfa, "--model=model", "--epochs=2"])
            self.assertEqual(0, result.exit_code, result.output)
            model = LanguageModel.load("model")
            self.assertIsInstance(model, LanguageModel)
            result = self.runner.invoke(ghostwriter, ["perplexity", "model", self.kakfa])
            self.assertEqual(0, result.exit_code, result.output)
            result = self.runner.invoke(ghostwriter, ["generate", "model", "The quick brown "])
            self.assertEqual(0, result.exit_code, result.output)

    def test_characters_from_text_file(self):
        kafka = open(self.kakfa)
        s1 = list(characters_from_text_files([kafka]))
        s2 = list(characters_from_text_files([kafka]))
        self.assertEqual(s1, s2)
        kafka.close()


class TestTokenCodec(unittest.TestCase):
    def setUp(self):
        self.codec = TokenCodec.create_from_text(list("the red balloon"))

    def test_token_codec_internals(self):
        self.assertEqual(["", " ", "e", "l", "o", "a", "b", "d", "h", "n", "r", "t"], self.codec.index_to_token)
        self.assertEqual(
            {"": 0, " ": 1, "e": 2, "l": 3, "o": 4, "a": 5, "b": 6, "d": 7, "h": 8, "n": 9, "r": 10, "t": 11},
            self.codec.token_to_index)

    def test_vocabulary_size(self):
        self.assertEqual(13, self.codec.vocabulary_size)

    def test_stringification(self):
        self.assertEqual("Token Codec: vocabulary size 13", repr(self.codec))
        self.assertEqual("Token Codec: vocabulary size 13", str(self.codec))

    def test_encode_decode(self):
        encoded = self.codec.encode(list("red"))
        self.assertIsInstance(encoded, Iterator)
        self.assertEqual([10, 2, 7], list(encoded))
        decoded = self.codec.decode([10, 2, 7])
        self.assertIsInstance(encoded, Iterator)
        self.assertEqual(["r", "e", "d"], list(decoded))

    def test_encode_decode_with_unknown(self):
        encoded = self.codec.encode(list("zed"))
        self.assertIsInstance(encoded, Iterator)
        self.assertEqual([12, 2, 7], list(encoded))
        decoded = self.codec.decode([12, 2, 7])
        self.assertIsInstance(encoded, Iterator)
        self.assertEqual([self.codec.UNKNOWN, "e", "d"], list(decoded))


class TestModel(unittest.TestCase):
    def test_stringification(self):
        model = LanguageModel.create(256, 10, 0.2,
                                     TokenCodec.create_from_text(list("The quick brown fox jumps over the lazy dog.")))
        self.assertEqual("Language Model: 256 hidden nodes, Token Codec: vocabulary size 31", repr(model))
