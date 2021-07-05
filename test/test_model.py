import unittest
import torch
import torch.nn as nn
import numpy as np
from numpy.testing import assert_array_almost_equal
from seqmodel import model
from seqmodel.model.transformer import TransformerEncoderLayer
from seqmodel.model.transformer import MultiheadAttention


class TestModel(unittest.TestCase):

    def assert_shape_equal_to(self, x: np.ndarray, shape: list):
        self.assertListEqual(list(x.shape), shape)

    def assert_tensors_equal(self, x: torch.Tensor, y: torch.Tensor):
        assert_array_almost_equal(x.detach().numpy(), y.detach().numpy())

    def assert_modules_equal_to(self, module: nn.Module, types: list):
        for m, t in zip(module.modules(), types):
            self.assertIsInstance(m, t)

    def setUp(self) -> None:
        self.b = 32         # batch size
        self.s = 100        # seq len
        self.d = 128        # dims
        self.d_out = 4      # output dims
        self.d_ff = 256     # feedforward dims
        self.h = 4          # number of heads

    def test_LinearDecoder(self):
        x = torch.randn(self.b, 1, self.d)
        x = x.repeat((1, self.s, 1))

        decoder = model.LinearDecoder(self.d, self.d_out)
        self.assert_modules_equal_to(decoder.model,
            [nn.Sequential, nn.Linear, nn.ReLU, nn.Linear])
        y = decoder.forward(x)
        self.assert_shape_equal_to(y, [self.b, self.s, self.d_out])
        for i in range(self.s):
            self.assert_tensors_equal(y[:, 0, :], y[:, i, :])

        hparams = {
            'decode_dims': 64,
            'n_decode_layers': 5,
            'decode_dropout': 0.2,
        }
        decoder = model.LinearDecoder(self.d, self.d_out,
                nn.SELU, nn.AlphaDropout, **hparams)
        y = decoder.forward(x)
        self.assert_shape_equal_to(y, [self.b, self.s, self.d_out])
        self.assert_modules_equal_to(decoder.model,
            [nn.Sequential] + \
            [nn.Linear, nn.SELU, nn.AlphaDropout] * 4 + \
            [nn.Linear])

    def test_MultiheadAttention(self):
        x = torch.randn(self.b, 1, self.d)
        x = x.repeat((1, self.s, 1))
        layer = MultiheadAttention(self.d, self.h, 0.)
        y, _ = layer.forward(x, x, x)
        self.assert_shape_equal_to(y, [self.b, self.s, self.d])
        for i in range(self.s):
            self.assert_tensors_equal(y[:, 0, :], y[:, i, :])
        pytorch_layer = nn.MultiheadAttention(self.d, self.h,
                                        0., batch_first=True)
        pytorch_layer.load_state_dict(layer.state_dict())
        y_ref, _ = pytorch_layer.forward(x, x, x)
        self.assert_tensors_equal(y, y_ref)

    def test_TransformerEncoderLayer(self):
        x = torch.randn(self.b, 1, self.d)
        x = x.repeat((1, self.s, 1))

        with self.assertRaises(ValueError):
            layer = TransformerEncoderLayer(self.d + 1, self.h, self.d_ff, 0.)
        layer = TransformerEncoderLayer(self.d, self.h, self.d_ff, 0.)
        y, w = layer.forward(x, need_weights=True)
        self.assert_shape_equal_to(y, [self.b, self.s, self.d])
        self.assert_shape_equal_to(w, [self.b, self.h, self.s, self.s])
        for i in range(self.s):
            self.assert_tensors_equal(y[:, 0, :], y[:, i, :])
        self.assertTrue(
            torch.allclose(w, torch.full_like(w, w[0, 0, 0, 0].item())))

        # compare with pytorch default implementation
        x_2 = torch.randn(self.b, self.s, self.d)
        src_mask = (torch.randn(self.s, self.s) > 0)
        src_key_padding_mask = (torch.randn(self.b, self.s) > 0)
        pytorch_layer = nn.TransformerEncoderLayer(self.d, self.h, self.d_ff,
                                                    0., batch_first=True)
        # rename keys in state_dict before replacing pytorch_layer's state_dict
        state_dict = self.rename_state_dict_keys(layer.state_dict())
        pytorch_layer.load_state_dict(state_dict)
        y = pytorch_layer.forward(x)

        y_ref = pytorch_layer.forward(x_2, src_mask, src_key_padding_mask)
        y, _ = layer.forward(x_2, src_mask, src_key_padding_mask)
        self.assert_tensors_equal(y, y_ref)
        y, w = layer.forward(x_2, src_mask, src_key_padding_mask, need_weights=True)
        self.assert_tensors_equal(y, y_ref)

    def rename_state_dict_keys(self, original_state_dict):
        state_dict = {}
        for k, v in original_state_dict.items():
            if k == 'linear.model.0.weight':
                k = 'linear1.weight'
            elif k == 'linear.model.0.bias':
                k = 'linear1.bias'
            elif k == 'linear.model.2.weight':
                k = 'linear2.weight'
            elif k == 'linear.model.2.bias':
                k = 'linear2.bias'
            state_dict[k] = v
        return state_dict

    def test_TransformerEncoder(self):
        hparams = {
            'repr_dims': self.d,
            'feedforward_dims': self.d_ff,
            'n_heads': self.h,
            'n_layers': 3,
            'dropout': 0.,
        }
        x = torch.randn(self.b, self.s, self.d)
        encoder = model.TransformerEncoder(
                nn.Identity(), nn.ReLU, nn.Dropout, nn.LayerNorm, **hparams)
        y, outs, weights = encoder.forward(x)
        self.assert_shape_equal_to(y, [self.b, self.s, self.d])
        self.assertEqual(outs, [])
        self.assertEqual(weights, [])
        y, outs, weights = encoder.forward(x,
                            save_intermediate_outputs={0, 1, 2})
        self.assertEqual(len(outs), 3)
        self.assertEqual(len(weights), 3)
        for a, b in zip(outs, weights):
            self.assert_shape_equal_to(a, [self.b, self.s, self.d])
            self.assert_shape_equal_to(b, [self.b, self.h, self.s, self.s])
        y, outs, weights = encoder.forward(x,
                            save_intermediate_outputs={1})
        self.assertEqual(len(outs), 1)
        self.assertEqual(len(weights), 1)


if __name__ == '__main__':
    unittest.main()
