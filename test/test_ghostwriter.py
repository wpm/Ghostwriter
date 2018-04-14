import os
import unittest

from click.testing import CliRunner

from ghostwriter import LanguageModel
from ghostwriter.command import ghostwriter


class TestGhostwriter(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self.kakfa = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kafka.txt")

    def test_train_and_perplexity(self):
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(ghostwriter, ["train", self.kakfa, "--model=model", "--epochs=2"])
            self.assertEqual(0, result.exit_code)
            result = self.runner.invoke(ghostwriter, ["perplexity", "model", self.kakfa])
            self.assertEqual(0, result.exit_code)
            model = LanguageModel.load("model")
            self.assertIsInstance(model, LanguageModel)
