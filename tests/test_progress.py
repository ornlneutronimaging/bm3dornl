"""Tests for progress reporting in bm3d_ring_artifact_removal."""

import unittest
from unittest.mock import patch

import numpy as np
from bm3dornl.bm3d import bm3d_ring_artifact_removal


def _make_stack(n=4, h=32, w=32):
    rng = np.random.RandomState(42)
    return rng.rand(n, h, w).astype(np.float32)


def _make_2d(h=32, w=32):
    rng = np.random.RandomState(42)
    return rng.rand(h, w).astype(np.float32)


class TestProgressDefault(unittest.TestCase):
    def test_default_no_progress(self):
        stack = _make_stack()
        result = bm3d_ring_artifact_removal(stack)
        self.assertEqual(result.shape, stack.shape)


class TestProgressCallable(unittest.TestCase):
    def test_callable_receives_correct_args(self):
        stack = _make_stack(n=4)
        calls = []
        result = bm3d_ring_artifact_removal(
            stack, progress=lambda c, t: calls.append((c, t))
        )
        self.assertEqual(result.shape, stack.shape)
        self.assertGreater(len(calls), 0)
        for current, total in calls:
            self.assertEqual(total, 4)
        self.assertEqual(calls[-1][0], calls[-1][1])


class TestProgressTqdmMissing(unittest.TestCase):
    def test_true_without_tqdm_raises(self):
        stack = _make_stack()
        with patch.dict("sys.modules", {"tqdm": None, "tqdm.auto": None}):
            with self.assertRaises(ImportError) as ctx:
                bm3d_ring_artifact_removal(stack, progress=True)
            self.assertIn("tqdm", str(ctx.exception))


class TestProgress2DIgnored(unittest.TestCase):
    def test_2d_with_progress_callable(self):
        sinogram = _make_2d()
        result = bm3d_ring_artifact_removal(sinogram, progress=lambda c, t: None)
        self.assertEqual(result.shape, sinogram.shape)


class TestProgressTqdmAvailable(unittest.TestCase):
    def test_true_with_tqdm(self):
        try:
            import tqdm  # noqa: F401
        except ImportError:
            self.skipTest("tqdm not installed")
        stack = _make_stack()
        result = bm3d_ring_artifact_removal(stack, progress=True)
        self.assertEqual(result.shape, stack.shape)


class TestProgressInvalidType(unittest.TestCase):
    def test_string_raises_type_error(self):
        stack = _make_stack()
        with self.assertRaises(TypeError) as ctx:
            bm3d_ring_artifact_removal(stack, progress="yes")
        self.assertIn("progress", str(ctx.exception))

    def test_int_raises_type_error(self):
        stack = _make_stack()
        with self.assertRaises(TypeError):
            bm3d_ring_artifact_removal(stack, progress=42)


class TestProgressMultiscale(unittest.TestCase):
    def test_multiscale_callable(self):
        stack = _make_stack(n=3)
        calls = []
        result = bm3d_ring_artifact_removal(
            stack,
            mode="streak",
            multiscale=True,
            progress=lambda c, t: calls.append((c, t)),
        )
        self.assertEqual(result.shape, stack.shape)
        self.assertGreater(len(calls), 0)
        for current, total in calls:
            self.assertEqual(total, 3)
        self.assertEqual(calls[-1][0], 3)


if __name__ == "__main__":
    unittest.main()
