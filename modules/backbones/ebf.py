import torch
import torch.nn as nn

from modules.backbones.attention import Attention, CrossAttention
from modules.backbones.layers import LayerScale, RMSNorm, GLUFFN, FFN, CgMLP


class PAC(nn.Module):
    """Parallel Attention + Convolution block, merged via depthwise conv + linear."""

    def __init__(
            self, dim, num_heads, head_dim,
            c_kernel_size=31, m_kernel_size=31, use_rope=True, rope_cache=True,
            dropout_attn: float = 0.0, out_drop: float = 0.0, c_out_drop=0.1,
            c_latent_drop=0.0,
    ):
        super().__init__()
        self.attn = Attention(dim, num_heads, head_dim, use_rope, rope_cache, dropout_attn, out_drop)
        self.c = CgMLP(
            dim, kernel_size=c_kernel_size,
            latent_drop=c_latent_drop, out_drop=c_out_drop
        )

        self.a_norm = RMSNorm(dim)
        self.c_norm = RMSNorm(dim)

        self.merge_linear = nn.Linear(dim * 2, dim)
        self.merge_dw_conv = (
            nn.Conv1d(
                dim * 2, dim * 2, kernel_size=m_kernel_size, stride=1,
                padding=m_kernel_size // 2,
                groups=dim * 2
            )
            if m_kernel_size != 0 else
            None
        )

    def forward(self, x):
        a_o = self.attn(self.a_norm(x))
        c_o = self.c(self.c_norm(x))
        m_o = torch.cat([a_o, c_o], dim=-1)

        if self.merge_dw_conv is not None:
            m_o = self.merge_dw_conv(m_o.transpose(1, 2)).transpose(1, 2) + m_o
        m_o = self.merge_linear(m_o)
        return m_o


class EBF(nn.Module):
    """One EBF layer: FFN -> PAC -> FFN with residual connections.

    Padding is handled via masked_fill, no attn_mask.
    """

    def __init__(
            self, dim, num_heads, head_dim,
            c_kernel_size=31, m_kernel_size=31, use_rope=True, rope_cache=True,
            dropout_attn: float = 0.0, out_drop: float = 0.0, c_out_drop=0.1,
            c_latent_drop=0.0, use_ls=True, ffn_type='glu', ffn_latent_drop=0.1, ffn_out_drop=0.1,
            skip_first_ffn=False, skip_out_ffn=False,
    ):
        super().__init__()
        self.skip_first_ffn = skip_first_ffn
        self.skip_out_ffn = skip_out_ffn

        if ffn_type == 'glu':
            if not skip_first_ffn:
                self.ffn1 = GLUFFN(
                    dim, latent_dim=dim * 4, dropout_latent=ffn_latent_drop,
                    dropout_output=ffn_out_drop
                )
            if not skip_out_ffn:
                self.ffn2 = GLUFFN(
                    dim, latent_dim=dim * 4, dropout_latent=ffn_latent_drop,
                    dropout_output=ffn_out_drop
                )
        elif ffn_type == 'ffn':
            if not skip_first_ffn:
                self.ffn1 = FFN(
                    dim, latent_dim=dim * 4,
                    dropout_latent=ffn_latent_drop,
                    dropout_output=ffn_out_drop
                )
            if not skip_out_ffn:
                self.ffn2 = FFN(
                    dim, latent_dim=dim * 4,
                    dropout_latent=ffn_latent_drop,
                    dropout_output=ffn_out_drop
                )
        elif ffn_type == 'cgmlp':
            if not skip_first_ffn:
                self.ffn1 = CgMLP(
                    dim, latent_dim=int(dim * 2.5), latent_drop=ffn_latent_drop,
                    out_drop=ffn_out_drop, kernel_size=21
                )
            if not skip_out_ffn:
                self.ffn2 = CgMLP(
                    dim, latent_dim=int(dim * 2.5), latent_drop=ffn_latent_drop,
                    out_drop=ffn_out_drop, kernel_size=7
                )
        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type}")

        self.attn = PAC(
            dim, num_heads, head_dim, c_kernel_size, m_kernel_size, use_rope, rope_cache, dropout_attn,
            out_drop, c_out_drop, c_latent_drop
        )
        if not skip_first_ffn:
            self.norm1 = RMSNorm(dim)
        if not skip_out_ffn:
            self.norm2 = RMSNorm(dim)

        if use_ls:
            if not skip_first_ffn:
                self.layer_scale1 = LayerScale(dim)
            self.layer_scale2 = LayerScale(dim)
            if not skip_out_ffn:
                self.layer_scale3 = LayerScale(dim)
        else:
            if not skip_first_ffn:
                self.layer_scale1 = nn.Identity()
            self.layer_scale2 = nn.Identity()
            if not skip_out_ffn:
                self.layer_scale3 = nn.Identity()

    def forward(self, x, mask=None):
        if not self.skip_first_ffn:
            if mask is not None:
                x = x.masked_fill(~mask.unsqueeze(-1), 0)
            x = self.layer_scale1(self.ffn1(self.norm1(x))) * 0.5 + x
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0)
        x = self.layer_scale2(self.attn(x)) + x
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0)
        if not self.skip_out_ffn:
            x = self.layer_scale3(self.ffn2(self.norm2(x))) * 0.5 + x
            if mask is not None:
                x = x.masked_fill(~mask.unsqueeze(-1), 0)
        return x


class EBFEncoderBackbone(nn.Module):
    """Two-stream encoder backbone with cross-attention.

    Frame stream (``x``, pre-projected at ``x_in_dim``) and token stream
    (``token``, pre-embedded at ``token_in_dim``) are projected to ``dim`` by
    separate input projections. The token stream is encoded with kernel-7 EBF
    layers; the frame stream is encoded with kernel-31 EBF layers, then fused
    with the token stream via stacked CrossAttention layers (x attends to
    token). Both streams are projected to their respective output dimensions.

    Constructor signature: (x_in_dim, token_in_dim, x_out_dim, token_out_dim, **kwargs)
    Forward signature:     forward(x, token, t_mask, n_mask)
                           -> (x_features, token_features, attn)

    ``attn`` is a list of cross-attention weight tensors [B, H, T, N], one
    per CA layer.
    """

    def __init__(
            self,
            x_in_dim: int,
            token_in_dim: int,
            x_out_dim: int,
            token_out_dim: int,
            # shared params
            dim: int = 256,
            num_heads: int = 8,
            head_dim: int = 64,
            # token encoder
            token_num_layers: int = 4,
            token_c_kernel_size: int = 7,
            token_m_kernel_size: int = 7,
            # frame encoder
            x_enc_num_layers: int = 4,
            x_enc_c_kernel_size: int = 31,
            x_enc_m_kernel_size: int = 31,
            # cross-attention
            num_ca_layers: int = 1,
            ca_dropout_attn: float = 0.0,
            ca_out_drop: float = 0.0,
            qk_head_dim: int | None = None,
            # general EBF params
            use_rope: bool = True,
            rope_cache: bool = True,
            dropout_attn: float = 0.0,
            out_drop: float = 0.0,
            c_out_drop: float = 0.1,
            c_latent_drop: float = 0.0,
            use_ls: bool = True,
            ffn_type: str = 'glu',
            ffn_latent_drop: float = 0.1,
            ffn_out_drop: float = 0.1,
            skip_first_ffn: bool = True,
            skip_out_ffn: bool = False,
            use_out_norm: bool = True,
    ):
        super().__init__()

        self.use_out_norm = use_out_norm
        self.num_ca_layers = num_ca_layers
        self.dim = dim

        # ============ Input Projections (frame & token) ============
        self.x_input_proj = nn.Linear(x_in_dim, dim)
        self.token_input_proj = nn.Linear(token_in_dim, dim)

        # ============ Token Encoder (kernel=7) ============
        self.token_encoder = nn.ModuleList([
            EBF(dim=dim, num_heads=num_heads, head_dim=head_dim,
                c_kernel_size=token_c_kernel_size, m_kernel_size=token_m_kernel_size,
                use_rope=use_rope, rope_cache=rope_cache,
                dropout_attn=dropout_attn, out_drop=out_drop,
                c_out_drop=c_out_drop, c_latent_drop=c_latent_drop,
                use_ls=use_ls, ffn_type=ffn_type,
                ffn_latent_drop=ffn_latent_drop, ffn_out_drop=ffn_out_drop,
                skip_first_ffn=skip_first_ffn, skip_out_ffn=skip_out_ffn)
            for _ in range(token_num_layers)
        ])
        self.token_enc_norm = RMSNorm(dim)

        # ============ Frame Encoder (kernel=31) ============
        self.x_encoder = nn.ModuleList([
            EBF(dim=dim, num_heads=num_heads, head_dim=head_dim,
                c_kernel_size=x_enc_c_kernel_size, m_kernel_size=x_enc_m_kernel_size,
                use_rope=use_rope, rope_cache=rope_cache,
                dropout_attn=dropout_attn, out_drop=out_drop,
                c_out_drop=c_out_drop, c_latent_drop=c_latent_drop,
                use_ls=use_ls, ffn_type=ffn_type,
                ffn_latent_drop=ffn_latent_drop, ffn_out_drop=ffn_out_drop,
                skip_first_ffn=skip_first_ffn, skip_out_ffn=skip_out_ffn)
            for _ in range(x_enc_num_layers)
        ])
        self.x_enc_norm = RMSNorm(dim)

        # ============ Cross-Attention Layers (x <- token) ============
        self.cross_attn_layers = nn.ModuleList()
        self.ca_pre_norms = nn.ModuleList()
        self.ca_layer_scales = nn.ModuleList()

        for _ in range(num_ca_layers):
            self.cross_attn_layers.append(
                CrossAttention(
                    dim=dim, cross_dim=dim, qk_head_dim=qk_head_dim,
                    num_heads=num_heads, head_dim=head_dim,
                    dropout_attn=ca_dropout_attn, out_drop=ca_out_drop
                )
            )
            self.ca_pre_norms.append(RMSNorm(dim))
            self.ca_layer_scales.append(LayerScale(dim, layer_scale_init_value=1) if use_ls else nn.Identity())

        # ============ Output Heads ============
        if self.use_out_norm:
            self.output_norm_x = RMSNorm(dim)
            self.output_norm_token = RMSNorm(dim)
        self.output_proj_x = nn.Linear(dim, x_out_dim)
        self.output_proj_token = nn.Linear(dim, token_out_dim)

    def forward(self, x, token, t_mask, n_mask):
        """
        Args:
            x:      [B, T, x_in_dim]      input features
            token:  [B, N, token_in_dim]  pre-embedded tokens
            t_mask: [B, T] bool, True = valid frame
            n_mask: [B, N] bool, True = valid token
        Returns:
            x_features:     [B, T, x_out_dim]
            token_features: [B, N, token_out_dim]
            attn:           list of [B, H, T, N], length = num_ca_layers
        """
        # ============ Input Projections ============
        x = self.x_input_proj(x)
        token = self.token_input_proj(token)

        # ============ Token Encoder ============
        for layer in self.token_encoder:
            token = layer(token, mask=n_mask)
        token = self.token_enc_norm(token)

        # ============ Frame Encoder ============
        for layer in self.x_encoder:
            x = layer(x, mask=t_mask)
        x = self.x_enc_norm(x)

        # ============ Cross-Attention (x attends to token) ============
        # CrossAttention mask semantic: True = padding, so invert n_mask
        token_padding_mask = ~n_mask if n_mask is not None else None

        attn = []
        for i in range(self.num_ca_layers):
            residual = x
            x_normed = self.ca_pre_norms[i](x)

            ca_out, attn_w = self.cross_attn_layers[i](
                x_normed, token, mask=token_padding_mask, return_attn=True
            )
            attn.append(attn_w)

            x = residual + self.ca_layer_scales[i](ca_out)

            if t_mask is not None:
                x = x.masked_fill(~t_mask.unsqueeze(-1), 0)

        # ============ Output ============
        if self.use_out_norm:
            x = self.output_norm_x(x)
            token = self.output_norm_token(token)
        x_features = self.output_proj_x(x)
        token_features = self.output_proj_token(token)

        return x_features, token_features, attn


class EBFBackbone(nn.Module):
    """Single-stream decoder: input projection followed by stacked EBF layers."""

    def __init__(
            self, in_dim: int, out_dim: int,
            dim: int = 256,
            num_layers: int = 8,
            num_heads: int = 8,
            head_dim: int = 64,
            c_kernel_size: int = 31,
            m_kernel_size: int = 31,
            use_rope: bool = True,
            rope_cache: bool = True,
            dropout_attn: float = 0.0,
            out_drop: float = 0.0,
            c_out_drop: float = 0.1,
            c_latent_drop: float = 0.0,
            use_ls: bool = True,
            ffn_type: str = 'glu',
            ffn_latent_drop: float = 0.1,
            ffn_out_drop: float = 0.1,
            use_out_norm: bool = True,
            skip_first_ffn=True,
            skip_out_ffn=False,
    ):
        super().__init__()

        self.use_out_norm = use_out_norm

        self.input_proj = nn.Linear(in_dim, dim)

        self.layers = nn.ModuleList([
            EBF(dim=dim, num_heads=num_heads, head_dim=head_dim,
                c_kernel_size=c_kernel_size, m_kernel_size=m_kernel_size,
                use_rope=use_rope, rope_cache=rope_cache,
                dropout_attn=dropout_attn, out_drop=out_drop,
                c_out_drop=c_out_drop, c_latent_drop=c_latent_drop,
                use_ls=use_ls, ffn_type=ffn_type,
                ffn_latent_drop=ffn_latent_drop, ffn_out_drop=ffn_out_drop,
                skip_first_ffn=skip_first_ffn, skip_out_ffn=skip_out_ffn,
                )
            for _ in range(num_layers)
        ])

        if self.use_out_norm:
            self.output_norm = RMSNorm(dim)
        self.output_proj = nn.Linear(dim, out_dim)  # -> [B, T, C_out]

    def forward(self, x, mask=None):
        """
        Args:
            x: [B, T, in_dim] input tensor
            mask: [B, T] valid mask
        Returns:
            out: [B, T, C_out] output tensor
        """
        x = self.input_proj(x)

        for layer in self.layers:
            x = layer(x, mask=mask)

        if self.use_out_norm:
            x = self.output_norm(x)
        out = self.output_proj(x)
        return out
