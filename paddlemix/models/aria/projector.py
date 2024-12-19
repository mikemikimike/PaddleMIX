import paddle
import paddlenlp
from paddlemix.models.imagebind.transformer import MultiheadAttention
from paddlemix.activations import ACT2FN
class FFN(paddle.nn.Layer):
    """
    Feed-Forward Network module.

    Args:
        embed_dim (int): Input embedding dimension.
        ff_dim (int): Hidden dimension of the feed-forward network.
        output_dim (int): Output dimension.
    """

    def __init__(self, embed_dim, ff_dim, output_dim):
        super().__init__()
        self.linear_in = paddle.nn.Linear(in_features=embed_dim,
            out_features=ff_dim, bias_attr=False)
        self.linear_out = paddle.nn.Linear(in_features=ff_dim, out_features
            =output_dim, bias_attr=False)
        self.act = ACT2FN['gelu_new']

    def forward(self, hidden_states):
        hidden_states = self.act(self.linear_in(hidden_states))
        hidden_states = self.linear_out(hidden_states)
        return hidden_states


class CrossAttention(paddle.nn.Layer):
    """
    Cross-Attention module.

    Args:
        kv_dim (int): Dimension of key and value.
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        drop_out_rate (float): Dropout rate. Default is 0.
    """

    def __init__(self, kv_dim, embed_dim, num_heads, drop_out_rate=0):
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = paddle.nn.Linear(in_features=embed_dim, out_features=
            embed_dim, bias_attr=False)
        self.k_proj = paddle.nn.Linear(in_features=kv_dim, out_features=
            embed_dim, bias_attr=False)
        self.v_proj = paddle.nn.Linear(in_features=kv_dim, out_features=
            embed_dim, bias_attr=False)
        self.multihead_attn = MultiheadAttention(embed_dim, num_heads) # 报错要自己实现MultiheadAttention
        self.linear = paddle.nn.Linear(in_features=embed_dim, out_features=
            embed_dim)
        self.dropout = paddle.nn.Dropout(p=drop_out_rate)
        self.layer_norm = paddle.nn.LayerNorm(normalized_shape=embed_dim)
        self.ln_kv = paddle.nn.LayerNorm(normalized_shape=kv_dim)

    def forward(self, x, hidden_states, attn_mask=None, add_residual=False):
        """
        Forward pass of the CrossAttention module.

        Args:
            x (torch.Tensor): Input tensor for key and value.
            hidden_states (torch.Tensor): Input tensor for query.
            attn_mask (torch.Tensor, optional): Attention mask. Default is None.
            add_residual (bool): Whether to add residual connection. Default is False.

        Returns:
            torch.Tensor: Output tensor after cross-attention.
        """
        normed_hidden_states = self.layer_norm(hidden_states)
        query = self.q_proj(normed_hidden_states).transpose(perm=[1, 0, 2])
        x = self.ln_kv(x)
        key = self.k_proj(x).transpose(perm=[1, 0, 2])
        value = self.v_proj(x).transpose(perm=[1, 0, 2])
        attn_output, _ = self.multihead_attn(query, key, value, attn_mask=
            attn_mask)
        attn_output = attn_output.transpose(perm=[1, 0, 2])
        if add_residual:
            attn_output = hidden_states + self.dropout(self.linear(attn_output)
                )
        else:
            attn_output = self.dropout(self.linear(attn_output))
        return attn_output


class AriaProjector(paddle.nn.Layer):
    """
    A projection module with one cross attention layer and one FFN layer, which projects ViT's outputs into MoE's inputs.

    Args:
        patch_to_query_dict (dict): Maps patch numbers to their corresponding query numbers,
            e.g., {1225: 128, 4900: 256}. This allows for different query sizes based on image resolution.
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        kv_dim (int): Dimension of key and value.
        ff_dim (int): Hidden dimension of the feed-forward network.
        output_dim (int): Output dimension.
        norm_layer (nn.Module): Normalization layer. Default is nn.LayerNorm.

    Outputs:
        A tensor with the shape of (batch_size, query_number, output_dim)
    """

    def __init__(self, patch_to_query_dict, embed_dim, num_heads, kv_dim,
        ff_dim, output_dim, norm_layer=paddle.nn.LayerNorm):
        super().__init__()
        self.patch_to_query_dict = patch_to_query_dict
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query = paddle.base.framework.EagerParamBase.from_tensor(tensor
            =paddle.zeros(shape=[max(patch_to_query_dict.values()), self.
            embed_dim]))
        init_TruncatedNormal = paddle.nn.initializer.TruncatedNormal(std=0.02)
        init_TruncatedNormal(self.query)
        self.cross_attn = CrossAttention(kv_dim, embed_dim, num_heads)
        self.ln_ffn = norm_layer(embed_dim)
        self.ffn = FFN(embed_dim, ff_dim, output_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, paddle.nn.Linear):
            init_TruncatedNormal = paddle.nn.initializer.TruncatedNormal(std
                =0.02)
            init_TruncatedNormal(m.weight)
            if isinstance(m, paddle.nn.Linear) and m.bias is not None:
                init_Constant = paddle.nn.initializer.Constant(value=0)
                init_Constant(m.bias)
        elif isinstance(m, paddle.nn.LayerNorm):
            init_Constant = paddle.nn.initializer.Constant(value=0)
            init_Constant(m.bias)
            init_Constant = paddle.nn.initializer.Constant(value=1.0)
            init_Constant(m.weight)

    def forward(self, x, attn_mask=None):
        """
        Forward pass of the Projector module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_patches, kv_dim).
            attn_mask (torch.Tensor, optional): Attention mask. Default is None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, query_number, output_dim).
        """
        bs = tuple(x.shape)[0]
        queries = self.query.unsqueeze(axis=0).tile(repeat_times=[bs, 1, 1])
        query_num = self.patch_to_query_dict.get(tuple(x.shape)[1], None)
        assert query_num is not None, f'Query number for {tuple(x.shape)[1]} patches is not provided'
        queries = queries[:, :query_num, :]
        if attn_mask is not None:
            attn_mask = attn_mask.repeat_interleave(repeats=self.num_heads,
                axis=0)
            attn_mask = attn_mask.unsqueeze(axis=1).expand(shape=[-1,
                queries.shape[1], -1])
        attention_out = self.cross_attn(x, queries, attn_mask=attn_mask)
        out = self.ffn(self.ln_ffn(attention_out))
        return out
