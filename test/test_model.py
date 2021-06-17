import unittest
import torch
import torch.nn as nn
import numpy as np
from numpy.testing import assert_array_equal
from seqmodel import model


class TestModel(unittest.TestCase):

    def assert_shape_equal_to(self, x: np.ndarray, shape: list):
        self.assertListEqual(list(x.shape), shape)

    def assert_tensors_equal(self, x: torch.Tensor, y: torch.Tensor):
        assert_array_equal(x.detach().numpy(), y.detach().numpy())

    def assert_modules_equal_to(self, module: nn.Module, types: list):
        for m, t in zip(module.modules(), types):
            self.assertIsInstance(m, t)

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_LinearDecoder(self):
        batch = 32
        seq_len = 100
        dim = 128
        out_dim = 4
        x = torch.randn(batch, 1, dim)
        x = x.repeat((1, seq_len, 1))

        decoder = model.LinearDecoder(dim, out_dim)
        self.assert_modules_equal_to(decoder.model,
            [nn.Sequential, nn.Conv1d, nn.ReLU, nn.Conv1d])
        y = decoder.forward(x)
        self.assert_shape_equal_to(y, [batch, seq_len, out_dim])
        for i in range(seq_len):
            self.assert_tensors_equal(y[:, 0, :], y[:, i, :])

        hparams = {
            'decode_dims': 64,
            'n_decode_layers': 5,
            'decode_dropout': 0.2,
        }
        decoder = model.LinearDecoder(dim, out_dim,
                nn.SELU, nn.AlphaDropout, **hparams)
        y = decoder.forward(x)
        self.assert_shape_equal_to(y, [batch, seq_len, out_dim])
        self.assert_modules_equal_to(decoder.model,
            [nn.Sequential] + \
            [nn.Conv1d, nn.SELU, nn.AlphaDropout] * 4 + \
            [nn.Conv1d])


if __name__ == '__main__':
    unittest.main()
