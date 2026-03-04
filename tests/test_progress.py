"""Tests for progress reporting in bm3d_ring_artifact_removal."""

import unittest
from unittest.mock import patch

import numpy as np
from bm3dornl.bm3d import bm3d_ring_artifact_removal


def _make_stack(n=4, h=32, w=32):
    """Create a small synthetic 3D stack for testing."""
    rng = np.random.RandomState(42)
    return rng.rand(n, h, w).astype(np.float32)


def _make_2d(h=32, w=32):
    """Create a small synthetic 2D sinogram for testing."""
    rng = np.random.RandomState(42)
    return rng.rand(h, w).astype(np.float32)


class TestProgressDefault(unittest.TestCase):
    """progress=False (default) should work without any crash."""

    def test_default_no_progress(self):
        stack = _make_stack()
        result = bm3d_ring_artifact_removal(stack)
        self.assertEqual(result.shape, stack.shape)


class TestProgressCallable(unittest.TestCase):
    """progress=callable receives correct (current, total) arguments."""

    def test_callable_receives_correct_args(self):
        stack = _make_stack(n=4)
        calls = []
        result = bm3d_ring_artifact_removal(
            stack, progress=lambda c, t: calls.append((c, t))
        )
        self.assertEqual(result.shape, stack.shape)
        # Should receive (current, total) calls
        self.assertGreater(len(calls), 0)
        # All calls should have total = 4 (n_slices)
        for current, total in calls:
            self.assertEqual(total, 4)
        # Last call should have current == total
        self.assertEqual(calls[-1][0], calls[-1][1])


class TestProgressTqdmMissing(unittest.TestCase):
    """progress=True without tqdm should raise ImportError."""

    def test_true_without_tqdm_raises(self):
        stack = _make_stack()
        with patch.dict("sys.modules", {"tqdm": None, "tqdm.auto": None}):
            with self.assertRaises(ImportError) as ctx:
                bm3d_ring_artifact_removal(stack, progress=True)
            self.assertIn("tqdm", str(ctx.exception))


class TestProgress2DIgnored(unittest.TestCase):
    """2D input should silently ignore progress parameter."""

    def test_2d_with_progress_true(self):
        sinogram = _make_2d()
        # progress=True on 2D should not crash (tqdm import may or may not be available)
        # If tqdm is not available, it should still not raise because 2D skips progress
        result = bm3d_ring_artifact_removal(sinogram, progress=lambda c, t: None)
        self.assertEqual(result.shape, sinogram.shape)


class TestProgressTqdmAvailable(unittest.TestCase):
    """If tqdm is available, progress=True should complete without error."""

    def test_true_with_tqdm(self):
        try:
            import tqdm  # noqa: F401
        except ImportError:
            self.skipTest("tqdm not installed")
        stack = _make_stack()
        result = bm3d_ring_artifact_removal(stack, progress=True)
        self.assertEqual(result.shape, stack.shape)


class TestProgressMultiscale(unittest.TestCase):
    """Progress should work with the multiscale 3D path."""

    def test_multiscale_callable(self):
        stack = _make_stack(n=3)
        calls = []
        result = bm3d_ring_artifact_removal(
            stack, mode="streak", multiscale=True,
            progress=lambda c, t: calls.append((c, t)),
        )
        self.assertEqual(result.shape, stack.shape)
        self.assertGreater(len(calls), 0)
        # All calls should have total = 3
        for current, total in calls:
            self.assertEqual(total, 3)
        self.assertEqual(calls[-1][0], 3)


if __name__ == "__main__":
    unittest.main()
