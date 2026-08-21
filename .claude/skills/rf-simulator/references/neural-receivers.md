# Neural Receivers and Learned PHY Components

## Contents

1. [Neural Demapper](#neural-demapper) -- Replace classical LLR computation with a network
2. [Neural Channel Estimator](#neural-channel-estimator) -- Learn channel estimation from pilots
3. [Neural Decoder](#neural-decoder) -- RNN/transformer replacing iterative decoding
4. [End-to-End Autoencoder](#end-to-end-autoencoder) -- Joint TX/RX optimization over a channel
5. [Training Loop Pattern](#training-loop-pattern) -- Train and evaluate with `sim_ber()`
6. [Common Pitfalls](#common-pitfalls) -- Shape, dtype, and sign convention traps

---

## Neural Demapper

When building a neural demapper, the input is received symbols + channel estimate, and the output must be LLRs (log-likelihood ratios). Forgetting the LLR sign convention (positive = bit is 0) causes the decoder to invert all decisions.

```python
import torch
import torch.nn as nn

class NeuralDemapper(nn.Module):
    """Replaces classical APP/MaxLog demapper with a learned network."""

    def __init__(self, num_bits_per_symbol=4):
        super().__init__()
        # Input: concatenated [Re(y), Im(y), Re(h), Im(h), noise_var]
        # = 2 + 2 + 1 = 5 features per symbol
        self.net = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_bits_per_symbol)  # Output: LLRs per bit
        )
        self.num_bits_per_symbol = num_bits_per_symbol

    def forward(self, y, h, no):
        """
        Args:
            y: Received symbols [batch, num_symbols], complex
            h: Channel estimates [batch, num_symbols], complex
            no: Noise variance, scalar or [batch]
        Returns:
            llr: [batch, num_symbols, num_bits_per_symbol], float
                 Positive = bit 0 more likely, Negative = bit 1 more likely
        """
        # Stack real-valued features
        features = torch.stack([
            y.real, y.imag,
            h.real, h.imag,
            no.expand_as(y.real) if no.dim() == 0 else no.unsqueeze(-1).expand_as(y.real)
        ], dim=-1)  # [batch, num_symbols, 5]

        return self.net(features)  # [batch, num_symbols, num_bits_per_symbol]
```

---

## Neural Channel Estimator

When replacing `LSChannelEstimator`, the network input is the received pilot symbols on the resource grid and the output is the full channel estimate across all resource elements.

```python
class NeuralChannelEstimator(nn.Module):
    """Learn channel estimation from pilot observations."""

    def __init__(self, num_ofdm_symbols=14, fft_size=72,
                 num_pilot_symbols=2):
        super().__init__()
        self.fft_size = fft_size
        self.num_ofdm_symbols = num_ofdm_symbols

        # Input: pilot observations on pilot OFDM symbols
        # Output: full channel estimate across all time-frequency slots
        input_size = num_pilot_symbols * fft_size * 2  # real + imag
        output_size = num_ofdm_symbols * fft_size * 2

        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )

    def forward(self, y_pilots):
        """
        Args:
            y_pilots: [batch, num_pilot_symbols, fft_size], complex
        Returns:
            h_hat: [batch, num_ofdm_symbols, fft_size], complex
        """
        batch = y_pilots.shape[0]
        # Flatten and split real/imag
        x = torch.cat([y_pilots.real, y_pilots.imag], dim=-1)
        x = x.reshape(batch, -1)

        out = self.net(x)
        out = out.reshape(batch, self.num_ofdm_symbols, self.fft_size, 2)
        return torch.complex(out[..., 0], out[..., 1])
```

---

## Neural Decoder

When replacing LDPC/Polar decoders with a neural network, the input is LLRs and the output is hard bit decisions or refined LLRs. The network must handle variable-length codewords -- use padding or fixed block sizes.

```python
class NeuralDecoder(nn.Module):
    """GRU-based decoder replacing iterative LDPC decoding.
    Input: LLRs [batch, code_length]. Output: bit probs [batch, info_length]."""

    def __init__(self, code_length=128, info_length=64, hidden_size=256):
        super().__init__()
        self.gru = nn.GRU(1, hidden_size, 2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.info_length = info_length

    def forward(self, llr):
        out, _ = self.gru(llr.unsqueeze(-1))
        logits = self.fc(out).squeeze(-1)
        return torch.sigmoid(logits[:, :self.info_length])
```

---

## End-to-End Autoencoder

When training an autoencoder, the TX encoder maps bit sequences to complex symbols, and the RX decoder maps received symbols back to bits. The channel is non-trainable but must be differentiable.

```python
import torch
import torch.nn as nn
from sionna.phy.channel import AWGN

class Encoder(nn.Module):
    """Maps k bits to n complex symbols. Must power-normalize output."""
    def __init__(self, k=4, n=1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(k, 64), nn.ReLU(), nn.Linear(64, 2 * n))
        self.n = n
    def forward(self, bits):
        out = self.net(bits.float())
        x = torch.complex(out[..., :self.n], out[..., self.n:])
        return x / torch.sqrt(torch.mean(torch.abs(x) ** 2))  # Normalize

class Decoder(nn.Module):
    """Maps received symbols [batch, n] complex to bit probs [batch, k]."""
    def __init__(self, k=4, n=1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2*n, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, k))
    def forward(self, y):
        return torch.sigmoid(self.net(torch.cat([y.real, y.imag], dim=-1)))

# Training loop
encoder, decoder = Encoder(k=4, n=1), Decoder(k=4, n=1)
awgn = AWGN()
opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
for step in range(5000):
    bits = torch.randint(0, 2, (256, 4)).float()
    loss = nn.BCELoss()(decoder(awgn(encoder(bits), no=0.1)), bits)
    opt.zero_grad(); loss.backward(); opt.step()
```

---

## Training Loop Pattern

When evaluating, use `model.eval()` + `torch.no_grad()`. Forgetting `model.eval()` keeps dropout/batchnorm in training mode, giving inconsistent BER results.

```python
model.eval()
with torch.no_grad():
    for snr_db in snr_range_db:
        no = 10 ** (-snr_db / 10)
        # Generate bits -> encode -> channel -> decode -> compute_ber()
model.train()
```

---

## Common Pitfalls

When interfacing neural components with Sionna layers, match these conventions:

| Interface | Shape | Dtype | Convention |
|-----------|-------|-------|------------|
| Mapper output | `[batch, num_symbols]` | `torch.complex64` | Unit average power |
| Demapper input | `[batch, num_symbols]` | `torch.complex64` | After channel + noise |
| LLR output | `[batch, num_symbols, bits_per_sym]` | `torch.float32` | Positive = bit 0 |
| Decoder input | `[batch, code_length]` | `torch.float32` | LLR values |
| Decoder output | `[batch, info_length]` | `torch.float32` | Hard bits or probs |

When training produces NaN loss, check: (1) division by zero in power normalization, (2) log of negative values in LLR computation, (3) exploding gradients in RNN decoders -- use `clip_grad_norm_`.

When BER does not improve during training, verify that the loss function matches the task: use BCE for bit-level, cross-entropy for symbol-level. MSE on complex symbols does not optimize BER.
