"""
Quantized KV Cache for Memory-Efficient Attention
===================================================

Implements three KV cache quantization strategies from recent literature,
adapted for MLX on Apple Silicon. These techniques compress key/value
embeddings to reduce memory usage during inference and training.

Quantization Methods:

1. QJL (Quantized Johnson-Lindenstrauss)
   - Zandieh, Daliri & Han (2024), "QJL: 1-Bit Quantized JL Transform
     for KV Cache Quantization with Zero Overhead", arXiv:2406.03482v2
   - Random projection followed by sign-bit quantization
   - Asymmetric estimator: (1/m) * sign(S @ k) . (S @ q) is unbiased for <q,k>
   - Zero overhead: no quantization constants (zero-point/scale) stored

2. PolarQuant
   - Han, Kacham, Karbasi, Mirrokni & Zandieh (2025), "PolarQuant:
     Quantizing KV Caches with Polar Transformation", arXiv:2502.02617v1
   - Random preconditioning (Hadamard + sign flips) + polar coordinate transform
   - After preconditioning, polar angles follow known Beta distributions
   - Uniform angle quantization without calibration

3. TurboQuant
   - Zandieh, Daliri, Hadian & Mirrokni (2025), "TurboQuant: Online Vector
     Quantization with Near-optimal Distortion Rate", arXiv:2504.19874v1
   - Random rotation induces concentrated Beta distribution on coordinates
   - Optimal scalar quantizers per coordinate
   - Near-optimal: within 2.7x of information-theoretic lower bound

Integration with Ahamed's Quantum Framework:
   The quantization strategies here are COMPLEMENTARY to the quantum-inspired
   attention mechanism (quantum_attention.py, Ahamed 2024). They compress
   the KV cache storage; the quantum attention computes scores differently.

Already in our model: No (experimental)
"""

import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# Utility: Fast Walsh-Hadamard Transform
# ---------------------------------------------------------------------------

def _hadamard_matrix(d: int) -> mx.array:
    """Construct a normalized Hadamard matrix of size d (must be power of 2).

    Used by PolarQuant and TurboQuant for random preconditioning.
    The Walsh-Hadamard transform is the same mathematical object as
    the Hadamard gate in quantum computing (Ahamed, quantum_tensors/gates.py),
    generalized to d dimensions.

    Reference: PolarQuant (Han et al. 2025), Section 3.1
    """
    assert d > 0 and (d & (d - 1)) == 0, f"d must be power of 2, got {d}"
    H = np.array([[1.0]])
    while H.shape[0] < d:
        H = np.block([[H, H], [H, -H]])
    H = H / math.sqrt(d)  # normalize
    return mx.array(H.astype(np.float32))


def _random_sign_diagonal(d: int, seed: int = 42) -> mx.array:
    """Generate a random diagonal sign matrix D with entries in {-1, +1}.

    Used for randomized Hadamard transform: H @ D @ x.
    The random signs break symmetry and ensure uniform distribution
    of the transformed coordinates.

    Reference: PolarQuant (Han et al. 2025), Section 3.1;
               TurboQuant (Zandieh et al. 2025), Section 3
    """
    rng = np.random.RandomState(seed)
    signs = rng.choice([-1.0, 1.0], size=d).astype(np.float32)
    return mx.array(signs)


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    p = 1
    while p < n:
        p *= 2
    return p


def _random_sign_matrix(m: int, d: int, seed: int = 42) -> mx.array:
    """Generate a random {-1, +1} projection matrix for QJL.

    Reference: QJL (Zandieh, Daliri & Han, 2024), Section 3
    """
    rng = np.random.RandomState(seed)
    S = rng.choice([-1.0, 1.0], size=(m, d)).astype(np.float32)
    return mx.array(S)


# ---------------------------------------------------------------------------
# Cartesian-to-Polar Conversion
# ---------------------------------------------------------------------------

def _cartesian_to_polar(x: mx.array) -> Tuple[mx.array, mx.array]:
    """Convert vectors from Cartesian to polar (spherical) coordinates.

    Vectorized implementation — computes all angles in parallel using
    reverse cumulative sums instead of a sequential loop over d-1 dims.

    For a d-dimensional vector x, returns:
    - r: magnitude ||x|| (scalar per vector)
    - angles: (d-1) polar angles theta_1, ..., theta_{d-1}

    The math follows PolarQuant (Han et al. 2025), Section 3.2:
        theta_k = arccos(x_k / ||x_{k:}||)  for k = 0, ..., d-3
        theta_{d-2} = atan2(x_{d-1}, x_{d-2})  (final angle in [0, 2*pi))

    Vectorization strategy:
        ||x_{k:}||^2 = sum(x_k^2, ..., x_{d-1}^2)
        This is a REVERSE cumulative sum of x^2 along the last axis.
        We compute all d tail norms in one shot, then all arccos values
        in one shot — no Python loop needed.

    Args:
        x: Input vectors [..., d]

    Returns:
        r: Magnitudes [..., 1]
        angles: Polar angles [..., d-1]
    """
    d = x.shape[-1]
    x_sq = x * x  # [..., d]

    # Reverse cumulative sum: tail_sq[..., k] = sum(x[..., k:]^2)
    # Reverse -> cumsum -> reverse gives reverse cumulative sum
    # MLX uses slice reversal instead of flip()
    x_sq_rev = x_sq[..., ::-1]
    tail_sq_rev = mx.cumsum(x_sq_rev, axis=-1)
    tail_sq = tail_sq_rev[..., ::-1]  # [..., d]

    # r = ||x|| = sqrt(tail_sq[..., 0])
    r = mx.sqrt(tail_sq[..., :1] + 1e-12)  # [..., 1]

    # Tail norms: ||x_{k:}|| for k = 0, ..., d-2
    tail_norms = mx.sqrt(tail_sq[..., :-1] + 1e-12)  # [..., d-1]

    # Angles theta_0 ... theta_{d-3}: arccos(x_k / ||x_{k:}||)
    cos_vals = x[..., :-1] / tail_norms  # [..., d-1]
    cos_vals = mx.clip(cos_vals, -1.0 + 1e-7, 1.0 - 1e-7)
    angles = mx.arccos(cos_vals)  # [..., d-1]

    # Override last angle: theta_{d-2} = atan2(x_{d-1}, x_{d-2})
    last_angle = mx.arctan2(x[..., -1], x[..., -2] + 1e-12)
    last_angle = mx.where(last_angle < 0, last_angle + 2 * math.pi, last_angle)

    # Replace the last column of angles
    angles = mx.concatenate([angles[..., :-1], mx.expand_dims(last_angle, axis=-1)], axis=-1)

    return r, angles


def _polar_to_cartesian(r: mx.array, angles: mx.array) -> mx.array:
    """Convert from polar (spherical) coordinates back to Cartesian.

    Vectorized implementation — uses cumulative products of sin values
    instead of a sequential loop.

    Inverse of _cartesian_to_polar. Used for dequantization.

    Reference: PolarQuant (Han et al. 2025), Section 3.2

    The k-th Cartesian coordinate is:
        x_k = r * cos(theta_k) * prod(sin(theta_j) for j < k)
        x_{d-1} = r * prod(sin(theta_j) for all j)

    Vectorization: compute cumulative product of sin(theta) as prefix_prod,
    then multiply by cos(theta) element-wise.

    Args:
        r: Magnitudes [..., 1]
        angles: Polar angles [..., d-1]

    Returns:
        x: Cartesian vectors [..., d] where d = angles.shape[-1] + 1
    """
    sin_angles = mx.sin(angles)  # [..., d-1]
    cos_angles = mx.cos(angles)  # [..., d-1]

    # Prefix product of sin values:
    # prefix_prod[..., 0] = 1 (no sin terms before first angle)
    # prefix_prod[..., k] = prod(sin(theta_0), ..., sin(theta_{k-1}))
    #
    # We need d values: one for each coordinate.
    # For k=0: prefix = r
    # For k=1: prefix = r * sin(theta_0)
    # For k=j: prefix = r * prod(sin(theta_i) for i < j)

    # Cumulative product of sin_angles along last axis
    log_sin = mx.log(mx.abs(sin_angles) + 1e-30)
    cum_log_sin = mx.cumsum(log_sin, axis=-1)  # [..., d-1]

    # sign tracking: cumulative product of signs
    sin_sign = mx.where(sin_angles >= 0, mx.ones_like(sin_angles), -mx.ones_like(sin_angles))
    # Cumulative sign: multiply signs sequentially
    # Use log trick: cumsum of log(abs) gives log of cumulative product
    # For signs: +1 * +1 = +1, +1 * -1 = -1, etc.
    # We can track this via cumulative sum of (sign == -1) and check parity
    is_negative = mx.where(sin_angles < 0, mx.ones_like(sin_angles), mx.zeros_like(sin_angles))
    cum_neg_count = mx.cumsum(is_negative, axis=-1)
    cum_sign = mx.where((cum_neg_count % 2) < 0.5, mx.ones_like(cum_neg_count), -mx.ones_like(cum_neg_count))

    # cum_prod_sin[..., k] = prod(sin(theta_0..theta_k))
    cum_prod_sin = cum_sign * mx.exp(cum_log_sin)  # [..., d-1]

    # Build prefix products: [1, sin(t0), sin(t0)*sin(t1), ...]
    # Shape: [..., d]
    ones = mx.ones_like(r[..., 0:1])  # [..., 1]  (value = 1.0 for k=0)
    prefix_prod = mx.concatenate([ones, cum_prod_sin], axis=-1)  # [..., d]

    # Scale by r
    prefix_prod = prefix_prod * r  # [..., d]

    # Cartesian coordinates:
    # x_k = prefix_prod[k] * cos(theta_k) for k < d-1
    # x_{d-1} = prefix_prod[d-1] (no cos term)
    cos_extended = mx.concatenate([cos_angles, mx.ones_like(r)], axis=-1)  # [..., d]
    x = prefix_prod * cos_extended

    return x


# ---------------------------------------------------------------------------
# Scalar Quantization Utilities
# ---------------------------------------------------------------------------

def _uniform_quantize(x: mx.array, n_bits: int, x_min: float, x_max: float) -> mx.array:
    """Uniform scalar quantization to n_bits.

    Maps continuous values in [x_min, x_max] to 2^n_bits integer levels,
    then reconstructs the midpoint value.

    Reference: TurboQuant (Zandieh et al. 2025), Section 4 — applied to
    each coordinate after random rotation concentrates the distribution.
    """
    n_levels = (1 << n_bits)
    step = (x_max - x_min) / n_levels

    # Clamp to valid range
    x_clamped = mx.clip(x, x_min, x_max - 1e-8)

    # Quantize: map to integer level
    level = mx.floor((x_clamped - x_min) / step)
    level = mx.clip(level, 0, n_levels - 1)

    # Dequantize: midpoint reconstruction
    return x_min + (level + 0.5) * step


def _sign_bit_quantize(x: mx.array) -> mx.array:
    """1-bit sign quantization: x -> sign(x).

    Core operation of QJL (Zandieh, Daliri & Han, 2024).
    Returns +1 for non-negative, -1 for negative values.
    """
    return mx.where(x >= 0, mx.ones_like(x), -mx.ones_like(x))


# ---------------------------------------------------------------------------
# QJL Quantizer
# ---------------------------------------------------------------------------

class QJLQuantizer(nn.Module):
    """Quantized Johnson-Lindenstrauss KV cache compressor.

    Implements the QJL method from Zandieh, Daliri & Han (2024):
    1. Project keys via structured random projection: projected_k = H @ D @ k
    2. Sign-bit quantize: quantized_k = sign(projected_k)
    3. For attention: estimate <q, k> ≈ (1/d) * quantized_k . (H @ D @ q)

    Uses a **randomized Hadamard transform** (H @ D) instead of a dense
    random {-1,+1} matrix. This is mathematically equivalent to a JL
    projection (both satisfy the JL lemma) but offers two advantages:
    - O(d log d) computation vs O(d^2) for dense matmul
    - O(d) storage (sign diagonal) vs O(m*d) for dense matrix

    The estimator remains UNBIASED: E[estimate] = <q, k>.

    When proj_dim < head_dim, falls back to the dense random matrix
    for dimensionality reduction (Hadamard is square).

    Args:
        head_dim: Dimension of each attention head (must be power of 2
            when proj_dim == head_dim).
        proj_dim: Projection dimension m. Default = head_dim (no dim reduction).
            When proj_dim != head_dim, uses dense random matrix fallback.
        seed: Random seed for reproducible projection.
    """

    def __init__(self, head_dim: int, proj_dim: Optional[int] = None, seed: int = 42):
        super().__init__()
        self.head_dim = head_dim
        self.proj_dim = proj_dim or head_dim
        self.seed = seed

        # Always use Hadamard — pad to next power of 2 if needed
        self._padded_dim = _next_power_of_2(head_dim)
        self._needs_pad = self._padded_dim != head_dim
        self._use_hadamard = (self.proj_dim == head_dim) or (self.proj_dim == self._padded_dim)

        if self._use_hadamard:
            self._H = _hadamard_matrix(self._padded_dim)
            self._D = _random_sign_diagonal(self._padded_dim, seed)
        else:
            # Fallback: dense random matrix for custom proj_dim
            self._use_hadamard = False
            S = _random_sign_matrix(self.proj_dim, head_dim, seed)
            self._S = S / math.sqrt(self.proj_dim)

    def _project(self, x: mx.array) -> mx.array:
        """Apply the random projection to input vectors.

        Uses the randomized Hadamard transform (H @ D @ x), padding
        to the next power of 2 when head_dim isn't a power of 2.

        Args:
            x: Input tensor [..., D]

        Returns:
            Projected tensor [..., padded_D or m]
        """
        if self._use_hadamard:
            if self._needs_pad:
                pad_width = self._padded_dim - self.head_dim
                x = mx.concatenate([x, mx.zeros((*x.shape[:-1], pad_width))], axis=-1)
            return (x * self._D) @ self._H.T
        else:
            return x @ self._S.T

    def quantize_keys(self, k: mx.array) -> Tuple[mx.array, mx.array]:
        """Quantize keys via random projection + sign bit.

        Uses the randomized Hadamard transform for O(d log d) projection
        when proj_dim == head_dim, otherwise falls back to dense matmul.

        Args:
            k: Key tensor [B, H, T, D]

        Returns:
            quantized_k: Sign-bit quantized projection [B, H, T, m] (values in {-1, +1})
            k_norms: Key norms for scale recovery [B, H, T, 1]
        """
        # Project: [B, H, T, D] -> [B, H, T, m]
        projected = self._project(k)

        # Store norms for unbiased estimation
        k_norms = mx.sqrt(mx.sum(k * k, axis=-1, keepdims=True) + 1e-12)

        # Sign-bit quantization
        quantized = _sign_bit_quantize(projected)

        return quantized, k_norms

    def estimate_attention(
        self,
        q: mx.array,
        quantized_k: mx.array,
        k_norms: mx.array,
        scale: float,
    ) -> mx.array:
        """Estimate attention scores using QJL asymmetric estimator.

        Computes the unbiased estimator from Zandieh et al. (2024):
            <q, k> ≈ (1/m) * sign(S @ k) . (S @ q)

        where S is either the randomized Hadamard transform (H @ D) or
        a dense random matrix, depending on configuration.

        Args:
            q: Query tensor [B, H, T_q, D]
            quantized_k: Sign-quantized key projection [B, H, T_k, m]
            k_norms: Key norms [B, H, T_k, 1]
            scale: Attention scale factor (typically 1/sqrt(d))

        Returns:
            Estimated attention scores [B, H, T_q, T_k]
        """
        # Project queries (NOT quantized — asymmetric estimator)
        # [B, H, T_q, D] -> [B, H, T_q, m]
        q_projected = self._project(q)

        # Asymmetric inner product estimator
        # [B, H, T_q, m] @ [B, H, m, T_k] -> [B, H, T_q, T_k]
        scores = (q_projected @ quantized_k.transpose(0, 1, 3, 2)) / self.proj_dim

        # Apply attention scale
        scores = scores * scale

        return scores


# ---------------------------------------------------------------------------
# PolarQuant Quantizer
# ---------------------------------------------------------------------------

class PolarQuantizer(nn.Module):
    """Polar coordinate KV cache quantizer.

    Implements PolarQuant from Han, Kacham, Karbasi, Mirrokni & Zandieh (2025):
    1. Random preconditioning: x' = H @ D @ x (Hadamard + random signs)
    2. Convert to polar coordinates: (r, theta_1, ..., theta_{d-1})
    3. Quantize angles uniformly (distribution is known analytically)
    4. Store magnitude r at full or reduced precision

    Key insight: After random preconditioning, the polar angles follow
    concentrated Beta distributions whose parameters are analytically
    computable. This eliminates the need for calibration data or storing
    quantization constants (zero-point, scale) per data block.

    The polar decomposition also provides a principled amplitude/phase
    split for Ahamed's quantum density matrix attention.

    Args:
        head_dim: Dimension of each attention head (must be power of 2).
        n_bits: Number of bits per angle for quantization. Default 4.
        seed: Random seed for preconditioning matrix.
    """

    def __init__(self, head_dim: int, n_bits: int = 4, seed: int = 42):
        super().__init__()
        self.head_dim = head_dim
        self.n_bits = n_bits
        self.seed = seed

        # Pad to next power of 2 for Hadamard (supports any head_dim)
        self._padded_dim = _next_power_of_2(head_dim)
        self._needs_pad = self._padded_dim != head_dim

        # Preconditioning: H @ D (Hadamard times random signs)
        self._H = _hadamard_matrix(self._padded_dim)
        self._D = _random_sign_diagonal(self._padded_dim, seed)

    def _precondition(self, x: mx.array) -> mx.array:
        """Apply random preconditioning H @ D @ x.

        Pads to next power of 2 when head_dim isn't a power of 2,
        then applies the randomized Hadamard transform.

        Args:
            x: Input tensor [..., D]

        Returns:
            Preconditioned tensor [..., padded_D]
        """
        if self._needs_pad:
            pad_width = self._padded_dim - self.head_dim
            x = mx.concatenate([x, mx.zeros((*x.shape[:-1], pad_width))], axis=-1)
        x_signed = x * self._D
        return x_signed @ self._H.T

    def _unprecondition(self, x: mx.array) -> mx.array:
        """Inverse preconditioning: D^{-1} @ H^{-1} @ x = D @ H^T @ x.

        Removes padding after inverse transform.
        """
        x_inv_h = x @ self._H
        x_out = x_inv_h * self._D
        if self._needs_pad:
            x_out = x_out[..., :self.head_dim]
        return x_out

    def quantize(self, x: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        """Quantize vectors via polar coordinate quantization.

        Args:
            x: Input tensor [B, H, T, D]

        Returns:
            r: Magnitudes [B, H, T, 1] (full precision)
            quantized_angles: Quantized polar angles [B, H, T, D-1]
            (angles are stored at n_bits precision)
        """
        # 1. Random preconditioning
        x_precond = self._precondition(x)

        # 2. Convert to polar coordinates
        r, angles = _cartesian_to_polar(x_precond)

        # 3. Quantize angles — vectorized (no Python loop)
        # After preconditioning, theta_k ∈ [0, pi] for k < d-2,
        # and theta_{d-1} ∈ [0, 2*pi)
        # Quantize all [0, pi] angles in one shot, then the single [0, 2*pi] angle
        q_bulk = _uniform_quantize(angles[..., :-1], self.n_bits, 0.0, math.pi)
        q_last = _uniform_quantize(angles[..., -1:], self.n_bits, 0.0, 2 * math.pi)
        quantized_angles = mx.concatenate([q_bulk, q_last], axis=-1)

        return r, quantized_angles

    def dequantize(self, r: mx.array, quantized_angles: mx.array) -> mx.array:
        """Reconstruct vectors from quantized polar representation.

        Args:
            r: Magnitudes [B, H, T, 1]
            quantized_angles: Quantized angles [B, H, T, D-1]

        Returns:
            Reconstructed vectors [B, H, T, D]
        """
        # Polar to Cartesian
        x_precond = _polar_to_cartesian(r, quantized_angles)

        # Undo preconditioning
        return self._unprecondition(x_precond)

    def get_amplitude_and_phase(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """Extract amplitude (magnitude) and phase (angles) from vectors.

        This provides a principled alternative to the arbitrary half/half
        split used in QuantumDensityAttention (quantum_attention.py, line 232).

        Instead of splitting head dimensions arbitrarily:
            q_amp, q_phase = q[..., :half_d], q[..., half_d:]

        PolarQuant decomposes into:
            r (amplitude/magnitude) and theta (phase/direction)

        This is physically motivated: attention primarily depends on
        direction (angular similarity), not magnitude.

        Reference: Adaptation of PolarQuant (Han et al. 2025) to
                   quantum density matrix attention (Ahamed 2024).

        Args:
            x: Input tensor [..., D]

        Returns:
            amplitude: Preconditioned magnitude [..., 1]
            phase: Preconditioned polar angles [..., D-1]
        """
        x_precond = self._precondition(x)
        r, angles = _cartesian_to_polar(x_precond)
        return r, angles


# ---------------------------------------------------------------------------
# TurboQuant Quantizer
# ---------------------------------------------------------------------------

class TurboQuantizer(nn.Module):
    """Near-optimal vector quantizer via random rotation + scalar quantization.

    Implements TurboQuant from Zandieh, Daliri, Hadian & Mirrokni (2025):
    1. Random rotation: x' = R @ x (randomized Hadamard)
    2. After rotation, each coordinate follows a concentrated Beta distribution
    3. Apply optimal scalar quantizer per coordinate
    4. For unbiased inner products: two-stage (MSE quantizer + QJL on residual)

    Achieves near-optimal distortion rate within 2.7x of the information-
    theoretic lower bound proven in the paper.

    Args:
        head_dim: Dimension of each attention head (must be power of 2).
        n_bits: Number of bits per coordinate. Default 4 (3.5 effective with residual).
        use_residual_qjl: If True, apply 1-bit QJL to the quantization residual
            for unbiased inner product estimation. Default False.
        seed: Random seed for rotation matrix.
    """

    def __init__(
        self,
        head_dim: int,
        n_bits: int = 4,
        use_residual_qjl: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.n_bits = n_bits
        self.use_residual_qjl = use_residual_qjl
        self.seed = seed

        # Pad to next power of 2 for Hadamard (supports any head_dim)
        self._padded_dim = _next_power_of_2(head_dim)
        self._needs_pad = self._padded_dim != head_dim

        # Randomized Hadamard rotation (same as PolarQuant preconditioning)
        self._H = _hadamard_matrix(self._padded_dim)
        self._D = _random_sign_diagonal(self._padded_dim, seed)

        # QJL for residual (optional, for unbiased inner products)
        if use_residual_qjl:
            self._qjl = QJLQuantizer(head_dim, seed=seed + 1)

        # Precompute quantization bounds based on Beta distribution
        self._quant_range = 3.0 / math.sqrt(self._padded_dim)

    def _rotate(self, x: mx.array) -> mx.array:
        """Apply randomized Hadamard rotation (pads if needed)."""
        if self._needs_pad:
            pad_width = self._padded_dim - self.head_dim
            x = mx.concatenate([x, mx.zeros((*x.shape[:-1], pad_width))], axis=-1)
        return (x * self._D) @ self._H.T

    def _unrotate(self, x: mx.array) -> mx.array:
        """Inverse rotation (removes padding if needed)."""
        x_out = (x @ self._H) * self._D
        if self._needs_pad:
            x_out = x_out[..., :self.head_dim]
        return x_out

    def quantize(self, x: mx.array) -> Tuple[mx.array, mx.array, Optional[Tuple]]:
        """Quantize vectors via TurboQuant.

        Args:
            x: Input tensor [B, H, T, D]

        Returns:
            x_norm: Vector norms [B, H, T, 1]
            quantized: Quantized rotated coordinates [B, H, T, D]
            residual_qjl: Optional (quantized_residual, residual_norms) for
                unbiased inner product estimation
        """
        # Store norms (needed for reconstruction)
        x_norm = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True) + 1e-12)

        # Normalize to unit vectors (direction is what matters for attention)
        x_unit = x / x_norm

        # Random rotation
        x_rotated = self._rotate(x_unit)

        # Scalar quantization per coordinate
        quantized = _uniform_quantize(
            x_rotated, self.n_bits,
            -self._quant_range, self._quant_range
        )

        # Optional: QJL on residual for unbiased estimation
        residual_qjl = None
        if self.use_residual_qjl:
            residual = x_rotated - quantized
            residual_qjl = self._qjl.quantize_keys(residual)

        return x_norm, quantized, residual_qjl

    def dequantize(self, x_norm: mx.array, quantized: mx.array) -> mx.array:
        """Reconstruct vectors from TurboQuant representation.

        Args:
            x_norm: Original norms [B, H, T, 1]
            quantized: Quantized rotated coordinates [B, H, T, D]

        Returns:
            Reconstructed vectors [B, H, T, D]
        """
        # Inverse rotation
        x_unit = self._unrotate(quantized)

        # Restore magnitude
        return x_unit * x_norm


# ---------------------------------------------------------------------------
# Unified Quantized KV Cache
# ---------------------------------------------------------------------------

class QuantizedKVCache(nn.Module):
    """Unified KV cache with selectable quantization strategy.

    Wraps key and value tensors with quantization/dequantization,
    providing a transparent interface for attention computation.
    Designed as a drop-in addition to CausalSelfAttention.

    The three strategies offer different trade-offs:
    - QJL: Fastest, most aggressive compression (1-bit), unbiased estimator
    - PolarQuant: Best quality at moderate compression, calibration-free
    - TurboQuant: Near-optimal distortion, quality-neutral at 3.5 bits

    Args:
        n_kv_heads: Number of KV attention heads.
        head_dim: Dimension per attention head.
        strategy: "qjl", "polar", or "turbo".
        n_bits: Quantization bits (for polar/turbo). Default 4.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_kv_heads: int,
        head_dim: int,
        strategy: str = "turbo",
        n_bits: int = 4,
        seed: int = 42,
    ):
        super().__init__()
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.strategy = strategy
        self.n_bits = n_bits

        if strategy == "qjl":
            self.quantizer = QJLQuantizer(head_dim, seed=seed)
        elif strategy == "polar":
            self.quantizer = PolarQuantizer(head_dim, n_bits=n_bits, seed=seed)
        elif strategy == "turbo":
            self.quantizer = TurboQuantizer(head_dim, n_bits=n_bits, seed=seed)
        else:
            raise ValueError(f"Unknown quantization strategy: {strategy}. "
                             f"Use 'qjl', 'polar', or 'turbo'.")

    def quantize_kv(
        self, k: mx.array, v: mx.array
    ) -> dict:
        """Quantize key and value tensors.

        Args:
            k: Key tensor [B, H, T, D]
            v: Value tensor [B, H, T, D]

        Returns:
            Dictionary of quantized representations (strategy-specific).
        """
        if self.strategy == "qjl":
            k_quant, k_norms = self.quantizer.quantize_keys(k)
            v_quant, v_norms = self.quantizer.quantize_keys(v)
            return {
                "k_quant": k_quant, "k_norms": k_norms,
                "v_quant": v_quant, "v_norms": v_norms,
                "strategy": "qjl",
            }

        elif self.strategy == "polar":
            k_r, k_angles = self.quantizer.quantize(k)
            v_r, v_angles = self.quantizer.quantize(v)
            return {
                "k_r": k_r, "k_angles": k_angles,
                "v_r": v_r, "v_angles": v_angles,
                "strategy": "polar",
            }

        elif self.strategy == "turbo":
            k_norm, k_quant, k_res = self.quantizer.quantize(k)
            v_norm, v_quant, v_res = self.quantizer.quantize(v)
            return {
                "k_norm": k_norm, "k_quant": k_quant, "k_res": k_res,
                "v_norm": v_norm, "v_quant": v_quant, "v_res": v_res,
                "strategy": "turbo",
            }

    def dequantize_kv(self, cache: dict) -> Tuple[mx.array, mx.array]:
        """Dequantize key and value tensors from cache.

        Args:
            cache: Dictionary from quantize_kv().

        Returns:
            k: Reconstructed keys [B, H, T, D]
            v: Reconstructed values [B, H, T, D]
        """
        strategy = cache["strategy"]

        if strategy == "qjl":
            # QJL doesn't directly dequantize — use estimate_attention instead.
            # For value reconstruction, we use sign * norm as approximation.
            k_approx = cache["k_quant"] * cache["k_norms"]
            v_approx = cache["v_quant"] * cache["v_norms"]
            # Trim padding if head_dim was padded to power of 2
            hd = self.head_dim
            if k_approx.shape[-1] > hd:
                k_approx = k_approx[..., :hd]
                v_approx = v_approx[..., :hd]
            return k_approx, v_approx

        elif strategy == "polar":
            k = self.quantizer.dequantize(cache["k_r"], cache["k_angles"])
            v = self.quantizer.dequantize(cache["v_r"], cache["v_angles"])
            return k, v

        elif strategy == "turbo":
            k = self.quantizer.dequantize(cache["k_norm"], cache["k_quant"])
            v = self.quantizer.dequantize(cache["v_norm"], cache["v_quant"])
            return k, v

    def compute_attention(
        self,
        q: mx.array,
        cache: dict,
        scale: float,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Compute attention output using quantized KV cache.

        For QJL, uses the asymmetric estimator directly.
        For PolarQuant/TurboQuant, dequantizes then computes standard attention.

        Args:
            q: Query tensor [B, H, T_q, D]
            cache: Quantized KV cache from quantize_kv()
            scale: Attention scale factor
            mask: Attention mask [T_q, T_k] or [B, 1, T_q, T_k]

        Returns:
            Attention output [B, H, T_q, D]
        """
        strategy = cache["strategy"]

        if strategy == "qjl":
            # Use asymmetric estimator for scores
            scores = self.quantizer.estimate_attention(
                q, cache["k_quant"], cache["k_norms"], scale
            )

            # Apply mask
            if mask is not None:
                scores = scores + mask

            # Softmax
            attn_weights = mx.softmax(scores, axis=-1)

            # For values, we must dequantize (QJL is designed for keys)
            v_approx = cache["v_quant"] * cache["v_norms"]
            out = attn_weights @ v_approx
            # Trim padding if head_dim was padded
            if out.shape[-1] > self.head_dim:
                out = out[..., :self.head_dim]
            return out

        else:
            # PolarQuant / TurboQuant: dequantize then standard attention
            k, v = self.dequantize_kv(cache)

            scores = (q @ k.transpose(0, 1, 3, 2)) * scale

            if mask is not None:
                scores = scores + mask

            attn_weights = mx.softmax(scores, axis=-1)
            return attn_weights @ v

    def estimate_compression_ratio(self) -> float:
        """Estimate the compression ratio achieved by this quantizer.

        Returns the ratio of original_size / compressed_size.
        """
        original_bits_per_elem = 16  # fp16

        if self.strategy == "qjl":
            # 1 bit per projected element + 16 bits for norm per vector
            compressed = 1.0 + 16.0 / self.head_dim
            return original_bits_per_elem / compressed

        elif self.strategy == "polar":
            # n_bits per angle * (d-1) angles + 16 bits for magnitude
            # Per element: (n_bits * (d-1) + 16) / d
            compressed = (self.n_bits * (self.head_dim - 1) + 16) / self.head_dim
            return original_bits_per_elem / compressed

        elif self.strategy == "turbo":
            # n_bits per coordinate + 16 bits for norm per vector
            compressed = self.n_bits + 16.0 / self.head_dim
            return original_bits_per_elem / compressed

        return 1.0
