# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for INT8 matrix multiplication accumulation (``ck.mm_int8``).

The contract is exact INT32 accumulation on every device. The MPS tests are the
regression guard for the fp32 fallback (aten::_int_mm has no MPS kernel), and
skip on machines without MPS; the CPU test covers the shape padding shared by
all devices.
"""

import pytest
import torch

import comfy_kitchen as ck

from .conftest import requires_mps

SEED = 42

# Dot products of biased operands at this K exceed FP32_EXACT_INT_LIMIT, the
# largest integer fp32 represents exactly. That is the regime where an fp32
# accumulator stops reproducing exact INT32 results.
WIDE_K = 8192
FP32_EXACT_INT_LIMIT = 2**24

# Shapes outside the 8-alignment the padding in _int8_matmul_accumulate targets.
UNALIGNED_M, UNALIGNED_K, UNALIGNED_N = 5, 7, 13

# Biased-positive operands: at WIDE_K every dot product lands above
# FP32_EXACT_INT_LIMIT (min 64 * 64 * WIDE_K = 33.5M > 16.7M).
BIAS_LOW = 64

INT8_MAX = torch.iinfo(torch.int8).max


def _random_pair(m: int, k: int, n: int, low: int, high: int, device: str):
    """Deterministic int8 pair, built on CPU so MPS and CPU see identical inputs."""
    gen = torch.Generator().manual_seed(SEED)
    a = torch.randint(low, high, (m, k), dtype=torch.int8, generator=gen)
    b = torch.randint(low, high, (k, n), dtype=torch.int8, generator=gen)
    return a.to(device), b.to(device)


def _cancelling_pair(m: int, k: int, n: int, device: str):
    """Dot products whose partial sums exceed FP32_EXACT_INT_LIMIT and then
    cancel to a small residual (the value depends on every half cancelling)."""
    a = torch.full((m, k), INT8_MAX, dtype=torch.int8)
    a[:, k // 2 :] = -INT8_MAX
    b = torch.full((k, n), INT8_MAX, dtype=torch.int8)
    b[-1, :] = 1
    return a.to(device), b.to(device)


def assert_exact_int32_mm(a: torch.Tensor, b: torch.Tensor) -> None:
    """mm_int8 must equal the exact int64 dot products, cell for cell."""
    # The reference is computed on CPU so it is exact and independent of the
    # device under test (an MPS-side reference could share its failure mode).
    expected = a.cpu().to(torch.int64) @ b.cpu().to(torch.int64)
    result = ck.mm_int8(a, b)
    assert result.dtype == torch.int32
    assert result.shape == expected.shape
    got = result.cpu().to(torch.int64)
    assert torch.equal(got, expected), (
        f"mm_int8 diverged from exact accumulation by up to {(got - expected).abs().max().item()}"
    )


def test_mm_int8_exact_on_cpu_unaligned_shapes():
    a, b = _random_pair(UNALIGNED_M, UNALIGNED_K, UNALIGNED_N, -INT8_MAX, INT8_MAX + 1, "cpu")
    assert_exact_int32_mm(a, b)


@requires_mps
@pytest.mark.parametrize(
    "case",
    ["zero_mean_unaligned", "accumulator_above_fp32_limit", "cancelling_partial_sums", "empty_k"],
)
def test_mm_int8_exact_on_mps(case):
    if case == "zero_mean_unaligned":
        a, b = _random_pair(
            UNALIGNED_M, UNALIGNED_K, UNALIGNED_N, -INT8_MAX, INT8_MAX + 1, "mps"
        )
    elif case == "accumulator_above_fp32_limit":
        a, b = _random_pair(4, WIDE_K, 4, BIAS_LOW, INT8_MAX + 1, "mps")
        # Self-check: this case is a regression guard only while its
        # accumulators really exceed the fp32 exact-integer limit.
        min_dot = (a.cpu().to(torch.int64) @ b.cpu().to(torch.int64)).min()
        assert min_dot > FP32_EXACT_INT_LIMIT
    elif case == "cancelling_partial_sums":
        a, b = _cancelling_pair(2, WIDE_K, 2, "mps")
    else:
        a, b = _random_pair(UNALIGNED_M, 0, UNALIGNED_N, -INT8_MAX, INT8_MAX + 1, "mps")
    assert_exact_int32_mm(a, b)
