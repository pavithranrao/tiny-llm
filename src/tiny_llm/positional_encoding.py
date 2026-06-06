import torch


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        assert dims % 2 == 0
        self.dims = dims
        self.seq_len = seq_len
        self.base = base
        self.traditional = traditional
        self.half_dims = dims // 2

        # θ_i = 1 / (base ^ (2i / d))
        pairs = torch.arange(0, self.half_dims, dtype=torch.float32)
        exponents = -2 * pairs / dims
        theta_i = torch.pow(base, exponents)
        positions = torch.arange(seq_len)
        angles = torch.outer(positions, theta_i)  # shape (seq_len, half_dims)

        self.cos_freqs = torch.cos(angles)
        self.sin_freqs = torch.sin(angles)

    def __call__(
        self,
        x: torch.Tensor,
        offset: list[slice] | slice | None = None,
    ) -> torch.Tensor:
        N, L, H, D = x.shape
        device = x.device
        cos_freqs = self.cos_freqs[:].to(device)
        sin_freqs = self.sin_freqs[:].to(device)

        if offset is not None:
            cos_basis = cos_freqs[offset]
            sin_basis = sin_freqs[offset]
        else:
            cos_basis = cos_freqs[:L]
            sin_basis = sin_freqs[:L]
        cos_basis = cos_basis.reshape(1, L, 1, self.half_dims)
        sin_basis = sin_basis.reshape(1, L, 1, self.half_dims)
        #    out1 = x1 * cos - x2 * sin
        #    out2 = x1 * sin + x2 * cos

        def rotation(x1, x2):
            real = x1 * cos_basis - x2 * sin_basis
            imag = x1 * sin_basis + x2 * cos_basis

            return real, imag

        if self.traditional:
            x = x.reshape(N, L, H, self.half_dims, 2)
            x1 = x[..., 0]
            x2 = x[..., 1]

            real, imag = rotation(x1, x2)
            y = torch.stack([real, imag], dim=-1)
        else:
            x1 = x[..., : self.half_dims]
            x2 = x[..., self.half_dims :]

            real, imag = rotation(x1, x2)
            y = torch.cat([real, imag], dim=-1)
        y = y.reshape(N, L, H, D)

        return y
