import typing
import warnings
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from seqmodel.hparam import PositionEncoderHparams
from seqmodel.hparam import TransformerEncoderHparams
from seqmodel.hparam import LinearDecoderHparams
from seqmodel.model.decoder import LinearDecoder


class PositionEncoder(nn.Module):

    def __init__(self, hparams: PositionEncoderHparams, 
        in_dims: int, out_dims: int,
    ):
        super().__init__()
        self.hparams = hparams

    def forward(self, x):
        return x #TODO


class TransformerEncoder(nn.Module):
    """Modifies `TransformerEncoder` module in pytorch.
    Provide option to return weights for every head in multihead attention.
    Sets default dim order to (N, S, E).
    Also includes positional encoder module.

    Args:
        hparams (TransformerEncoderHparams): hyperparameters (tracked, see hparam.py).
        pos_encoder (nn.Module): positional encoder, applied first.
        ActivationFn (nn.Module): type of activation function
            to apply between feedforward layers.
        DropoutFn (nn.Module): type of dropout to apply between
            feedforward layers and residual connections.
        LayerNormFn (nn.Module): type of normalization to apply
            on residual connections, use lambda function to specify
            args if needed, since instances are created as `LayerNormFn()`.
    """
    def __init__(self,
        hparams: TransformerEncoderHparams,
        pos_encoder: nn.Module,
        ActivationFn: nn.Module,
        DropoutFn: nn.Module,
        LayerNormFn: nn.Module,
    ):
        super().__init__()
        self.hparams = hparams
        self.pos_encoder = pos_encoder
        self.attn_layers = []
        feedforward_dims = self.hparams.feedforward_dims
        if feedforward_dims is None:
            feedforward_dims = 2 * self.hparams.repr_dims
        for _ in range(self.hparams.n_layers):
            self.attn_layers.append(
                TransformerEncoderLayer(
                    self.hparams.repr_dims,
                    self.hparams.n_heads,
                    feedforward_dims,
                    self.hparams.dropout,
                    ActivationFn,
                    DropoutFn,
                    LayerNormFn,
                )
            )

    def forward(self,
        src: Tensor,
        src_mask: Tensor = None,
        src_key_padding_mask: Tensor = None,
        save_intermediate_outputs: typing.Set[int] = {},
    ) -> typing.Tuple[Tensor, typing.List[Tensor], typing.List[Tensor]]:
        """Identical to `torch.nn.TransformerEncoder().forward()`,
        except applies positional encoder to input first, and
        has argument for selecting intermediate outputs and attention weights
        to return.

        Args:
            src (Tensor): input in (N, S, E) dimensions
            src_mask (Tensor, optional): the additive mask for the
                src sequence (same as pytorch). Defaults to None.
            src_key_padding_mask (Tensor, optional): the ByteTensor
                mask for src keys per batch (same as pytorch). Defaults to None.
            save_intermediate_outputs (typing.Set[int], optional): set of
                layer numbers for which intermediate representations and
                attention weights are returned. Defaults to {}.

        Returns:
            Tensor, list[Tensor], list[Tensor]:
                tuple of output tensor, list of intermediate outputs
                ordered by layer number, and list of weight tensors.
                Lists are empty if no layers in `save_intermediate_outputs`.
        """
        y, w = [], []
        src = self.pos_encoder(src)
        for i, layer in enumerate(self.attn_layers):
            need_weights = (i in save_intermediate_outputs)
            src, weights = layer.forward(src, src_mask, src_key_padding_mask,
                                        need_weights=need_weights)
            if need_weights:
                y.append(src)
                w.append(weights)
        return src, y, w


class TransformerEncoderLayer(nn.Module):
    """Internal object modifying pytorch's TransformerEncoderLayer,
    for use by `seqmodel.model.transformer.TransformerEncoder`.
    See `TransformerEncoder` for documentation.
    """
    def __init__(self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        ActivationFn: nn.Module = nn.ReLU,
        DropoutFn: nn.Module = nn.Dropout,
        LayerNormFn: nn.Module = nn.LayerNorm,
    ):
        """See documentation in `seqmodel.model.transformer.TransformerEncoder`.
        """
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f'd_model={d_model} not divisible by nhead={nhead}')
        # use custom MultiheadAttention object
        self.self_attn = MultiheadAttention(
                d_model, nhead, dropout=dropout)
        # create feedforward layers
        self.linear = LinearDecoder(
            LinearDecoderHparams(
                decode_dims=dim_feedforward,
                n_decode_layers=2,
                decode_dropout=dropout),
            d_model, d_model, ActivationFn, DropoutFn)
        self.dropout1 = DropoutFn(dropout)
        self.dropout2 = DropoutFn(dropout)
        self.norm1 = LayerNormFn(d_model)
        self.norm2 = LayerNormFn(d_model)

    def forward(self,
        src: Tensor,
        src_mask: Tensor = None,
        src_key_padding_mask: Tensor = None,
        need_weights = False,
    ):
        """Identical to pytorch version.
        """
        src2, weights = self.self_attn.forward(src, src, src,
                    attn_mask=src_mask, key_padding_mask=src_key_padding_mask,
                    need_weights=need_weights)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, weights

class MultiheadAttention(nn.MultiheadAttention):

    def __init__(self,
        embed_dim,
        num_heads,
        dropout=0.,
        add_zero_attn=False,
        ):
        """Same as `torch.nn.MultiheadAttention`, but with option
        to return attention weights for each head separately instead of
        as average. Disallow specifying bias, add_bias_kv, kdim, vdim.
        """
        super().__init__(
            embed_dim, num_heads, dropout=dropout,
            bias=True, add_bias_kv=False,
            add_zero_attn=add_zero_attn, kdim=None, vdim=None)
        self.id_out_weight = torch.eye(self.head_dim)
        self.id_out_bias = torch.zeros(self.head_dim)

    def forward(self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=False,
        attn_mask=None,
    ):
        """Same as `torch.nn.MultiheadAttention`, but if `need_weights=True`
        replace with modified code which avoids averaging attention weight
        over heads as in pytorch.

        Returns:
            Tensor, Tensor: if need_weights is False,
                returns output and None, else returns output and
                tensor of attention weights with shape (N, H, S, S)
                where H is number of attention heads.
        """
        # transpose to (S, N, E) for attention
        query = query.transpose(1, 0)
        key = key.transpose(1, 0)
        value = value.transpose(1, 0)
        # use modified function if weights needed, else use pytorch version
        if need_weights:
            y, w = multi_head_attention_forward(
                    query, key, value, self.embed_dim, self.num_heads,
                    self.in_proj_weight,
                    self.in_proj_bias,
                    None, None, self.add_zero_attn, self.dropout,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    training=self.training,
                    key_padding_mask=key_padding_mask, need_weights=True,
                    attn_mask=attn_mask)
        else:
            y, w = super().forward(
                query, key, value,
                key_padding_mask=key_padding_mask,
                need_weights=False, attn_mask=attn_mask)
        # transpose back to (N, S, E) for attention
        return y.transpose(1, 0), w

# the following code is from pytorch 1.9
# https://github.com/pytorch/pytorch/blob/release/1.9/torch/nn/functional.py
# only one line is changed at end to avoid averaging attention weights over heads
def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    bias_k: Tensor,
    bias_v: Tensor,
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
    training: bool = True,
    key_padding_mask: Tensor = None,
    need_weights: bool = True,
    attn_mask: Tensor = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Tensor = None,
    k_proj_weight: Tensor = None,
    v_proj_weight: Tensor = None,
    static_k: Tensor = None,
    static_v: Tensor = None,
) -> typing.Tuple[Tensor, Tensor]:
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """
    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert key.shape[:2] == value.shape[:2], \
            f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    #
    # compute in-projection
    #
    if not use_separate_proj_weight:
        q, k, v = F._in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
        assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
        assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = F._in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

    # prep attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)
        else:
            assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
                f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    # prep key padding mask
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask = key_padding_mask.to(torch.bool)

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_k.size(0) == bsz * num_heads, \
            f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
        assert static_k.size(2) == head_dim, \
            f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        k = static_k
    if static_v is None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_v.size(0) == bsz * num_heads, \
            f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        assert static_v.size(2) == head_dim, \
            f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        v = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), \
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
            expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

    # convert mask to float
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #
    attn_output, attn_output_weights = F._scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        # line is changed from pytorch to remove sum over dim=-1
        return attn_output, attn_output_weights
    else:
        return attn_output, None