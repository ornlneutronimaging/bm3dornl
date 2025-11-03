# JAX Migration Plan for BM3DORNL

## Executive Summary

This document outlines a comprehensive plan for migrating the BM3DORNL library from its current architecture (NumPy + Numba + CuPy) to JAX as the core computing library. JAX offers unified CPU/GPU execution, automatic differentiation, just-in-time compilation, and improved portability while maintaining NumPy-like syntax.

**Current Architecture:**
- **NumPy**: Base array operations and CPU computation
- **Numba**: JIT compilation for CPU performance optimization (primarily used in block matching and aggregation)
- **CuPy**: GPU acceleration for computationally intensive operations (collaborative filtering, FFT operations)

**Target Architecture:**
- **JAX**: Unified framework providing automatic CPU/GPU dispatch, XLA compilation, and functional programming paradigm

---

## 1. Motivation for Migration

### 1.1 Benefits of JAX

1. **Unified CPU/GPU Code** ([JAX Documentation](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html))
   - Single codebase works on both CPU and GPU without code duplication
   - Automatic device placement eliminates need for separate CuPy implementations
   - Solves current issue: "At present, the library requires a CUDA-enabled GPU" (from FAQ.md)

2. **High Performance Through XLA** ([JAX: Autograd and XLA, ICLR 2018](https://openreview.net/forum?id=SkluMSZ0W))
   - XLA (Accelerated Linear Algebra) compiler optimizes entire computation graphs
   - Often matches or exceeds hand-tuned CuPy/CUDA performance
   - Automatic fusion of operations reduces memory bandwidth requirements

3. **Functional Programming Paradigm** ([JAX Sharp Bits](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html))
   - Explicit pure functions enable better optimization
   - Easier to reason about and test
   - Better support for parallelization

4. **Automatic Differentiation** ([JAX Autodiff Cookbook](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html))
   - Native support for gradient computation
   - Enables potential future extensions (e.g., learned denoising parameters)
   - Facilitates hyperparameter optimization

5. **Active Development and Community Support**
   - Maintained by Google with strong community support
   - Used in production at DeepMind, Google Research, and other major institutions
   - Regular updates and performance improvements
   - Growing ecosystem of compatible libraries (Flax, Optax, etc.)

6. **Better Vectorization and SIMD** ([JAX `vmap` documentation](https://jax.readthedocs.io/en/latest/jax-101/03-vectorization.html))
   - `vmap` for automatic vectorization superior to manual loop optimization
   - Composable transformations (`jit`, `grad`, `vmap`, `pmap`)

### 1.2 Limitations to Consider

1. **Functional Programming Paradigm** ([JAX FAQ](https://jax.readthedocs.io/en/latest/faq.html))
   - Requires pure functions (no side effects)
   - In-place array modifications not allowed (arrays are immutable)
   - May require refactoring of existing stateful code

2. **Dynamic Shapes** ([JAX Sharp Bits: Shape Polymorphism](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#python-control-flow-+-JIT))
   - JIT compilation works best with static shapes
   - Dynamic shapes require special handling or may fall back to eager mode
   - Current code with variable patch counts may need adaptation

3. **Learning Curve**
   - Team needs training on functional programming concepts
   - Different debugging paradigm compared to imperative Python

4. **Dependencies**
   - Adds JAX as a new dependency (though removes CuPy and potentially Numba)
   - Need to maintain compatibility across JAX versions

---

## 2. Current Architecture Analysis

### 2.1 Module Breakdown

Based on code analysis of `/home/runner/work/bm3dornl/bm3dornl/src/bm3dornl/`:

| Module | Lines | Current Tech | Migration Complexity | Priority |
|--------|-------|--------------|---------------------|----------|
| `bm3d.py` | 866 | NumPy, imports from other modules | High | High |
| `block_matching.py` | 430 | **Numba** (`@njit`, `@njit(parallel=True)`) | **High** | **High** |
| `denoiser_gpu.py` | 254 | **CuPy**, GPU-specific | **High** | **High** |
| `phantom.py` | 241 | NumPy, SciPy | Medium | Low |
| `utils.py` | 152 | NumPy, SciPy, **Numba** (`@njit`) | Medium | Medium |
| `aggregation.py` | 103 | **Numba** (`@njit(parallel=True)`) | **High** | **High** |
| `noise_analysis.py` | 85 | NumPy | Low | Medium |
| `signal.py` | 31 | NumPy, SciPy | Low | High |
| `plot.py` | 24 | Matplotlib | Low | Low |

### 2.2 Key Operations to Migrate

#### High-Priority Operations (Core Algorithm)

1. **Block Matching** (`block_matching.py`)
   - Currently uses Numba JIT with `@njit` and `@njit(parallel=True)`
   - Operations: patch extraction, distance matrix computation, candidate patch identification
   - JAX equivalent: `jax.jit`, `jax.vmap`, `jax.lax.scan`

2. **GPU Denoising** (`denoiser_gpu.py`)
   - Currently uses CuPy for GPU operations
   - Operations: FFT shrinkage, Hadamard transforms, collaborative filtering
   - JAX equivalent: `jax.numpy.fft`, `jax.jit` (auto-GPU), native array ops

3. **Aggregation** (`aggregation.py`)
   - Currently uses Numba with parallel JIT
   - Operations: weighted patch aggregation back to image
   - JAX equivalent: `jax.vmap`, `jax.lax.scan`, or scatter operations

4. **Signal Processing** (`signal.py`)
   - FFT and Hadamard transforms
   - JAX equivalent: `jax.numpy.fft.fftn`, custom Hadamard implementation

#### Medium-Priority Operations

5. **Noise Analysis** (`noise_analysis.py`)
   - Variance estimation, PSD computation
   - Straightforward NumPy → JAX NumPy conversion

6. **Utilities** (`utils.py`)
   - Background estimation, up/downscaling
   - Mix of NumPy and Numba
   - JAX equivalent: Replace Numba with JAX JIT

#### Low-Priority Operations

7. **Phantom Generation** (`phantom.py`)
   - Test data generation
   - Can remain NumPy or be migrated last

8. **Plotting** (`plot.py`)
   - Visualization only
   - No migration needed (convert JAX arrays to NumPy when plotting)

### 2.3 Dependency Analysis

**Current Dependencies** (from `pyproject.toml`):
```python
dependencies = [
  "numpy",
  "scipy",
  "numba",          # → Remove or make optional
  "cupy-cuda12x",   # → Remove
  "scikit-image",
]
```

**Proposed Dependencies After Migration:**
```python
dependencies = [
  "jax[cuda12]",    # Includes GPU support
  "jaxlib",         # JAX linear algebra backend
  "numpy",          # For I/O and plotting interfaces
  "scipy",          # For some utilities (if not migrated)
  "scikit-image",   # Unchanged
]
```

**JAX Installation Considerations** ([JAX Installation Guide](https://github.com/google/jax#installation)):
- JAX supports CUDA 11.x and 12.x
- CPU-only installation: `pip install jax`
- GPU installation: `pip install jax[cuda12_pip]` or `pip install jax[cuda11_pip]`
- Conda: `conda install jax -c conda-forge`

---

## 3. Migration Strategy

### 3.1 Phased Approach

We recommend a **gradual, module-by-module migration** to minimize risk and allow for iterative testing.

#### Phase 1: Foundation and Infrastructure (Weeks 1-2)
**Goal:** Set up JAX infrastructure and migrate low-hanging fruit

- [ ] Install JAX alongside existing dependencies
- [ ] Create JAX utility module with helper functions
- [ ] Migrate `signal.py` (31 lines, pure NumPy/SciPy)
  - Replace `numpy.fft` with `jax.numpy.fft`
  - Replace `scipy.linalg.hadamard` with JAX implementation or custom code
- [ ] Migrate `noise_analysis.py` (85 lines, pure NumPy)
  - Straightforward numpy → jax.numpy conversion
- [ ] Set up dual testing framework (existing tests + JAX tests)
- [ ] Create performance benchmarking suite
- [ ] **Deliverable:** JAX installed, 2 modules migrated, tests passing

#### Phase 2: Core Numba Components (Weeks 3-5)
**Goal:** Replace Numba JIT compilation with JAX JIT

**2a. Block Matching (`block_matching.py`)**
- [ ] Replace `@njit` decorators with `@jax.jit`
- [ ] Refactor loops to use `jax.lax.scan` or `jax.vmap`
- [ ] Handle parallel operations with `jax.vmap` or `jax.pmap`
- [ ] Key functions to migrate:
  - `find_candidate_patch_ids`: Use vectorized operations
  - `get_signal_patch_positions`: Replace nested loops with `jax.vmap`
  - `compute_variance_weighted_distance_matrix`: Vectorize distance calculations
  - `form_hyper_blocks_from_distance_matrix`: Use advanced indexing

**2b. Aggregation (`aggregation.py`)**
- [ ] Replace `@njit(parallel=True)` with JAX parallelization
- [ ] Use `segment_sum` or scatter-add operations for aggregation
- [ ] Functions to migrate:
  - `aggregate_block_to_image`: Use `jax.ops.segment_sum` or scatter operations
  - `aggregate_denoised_block_to_image`: Similar approach

**2c. Utilities (`utils.py`)**
- [ ] Migrate `@njit` decorated functions to JAX
- [ ] `is_within_threshold`: Simple JAX conversion
- [ ] Keep SciPy-dependent functions as-is initially (can convert later if needed)

- [ ] **Deliverable:** All Numba code replaced with JAX, performance benchmarks show equivalent or better performance

#### Phase 3: GPU Operations (Weeks 6-7)
**Goal:** Replace CuPy with JAX (achieving CPU/GPU unification)

**3a. GPU Denoiser (`denoiser_gpu.py`)**
- [ ] Remove explicit `cp.asarray()` / `cp.asnumpy()` conversions
- [ ] Replace CuPy operations with JAX equivalents:
  - `cp.fft.fftn` → `jax.numpy.fft.fftn`
  - `cp.where` → `jax.numpy.where`
  - `cupyx.scipy.linalg.hadamard` → custom JAX Hadamard or port from SciPy
- [ ] Remove manual memory management (`memory_cleanup()`)
- [ ] Functions to migrate:
  - `shrinkage_fft`: Direct translation, remove GPU-specific code
  - `shrinkage_hadamard`: Implement Hadamard transform in JAX
  - `collaborative_wiener_filtering`: Pure JAX operations
  - `collaborative_hadamard_filtering`: Pure JAX operations

**3b. Main BM3D Module (`bm3d.py`)**
- [ ] Update imports to use JAX-migrated modules
- [ ] Ensure all functions use JAX arrays throughout pipeline
- [ ] Test CPU and GPU execution with same code

- [ ] **Deliverable:** Full CPU/GPU support with single codebase, CuPy dependency removed

#### Phase 4: Integration and Optimization (Weeks 8-9)
**Goal:** End-to-end testing and performance optimization

- [ ] Full integration testing with real neutron imaging data
- [ ] Performance benchmarking against original implementation
- [ ] Optimize JIT compilation (static argnums, donate_argnums)
- [ ] Profile and eliminate bottlenecks
- [ ] CPU-only execution testing (address FAQ requirement)
- [ ] Memory usage optimization
- [ ] **Deliverable:** Production-ready JAX implementation

#### Phase 5: Documentation and Deprecation (Weeks 10-11)
**Goal:** Update documentation and plan deprecation

- [ ] Update README with JAX installation instructions
- [ ] Update FAQ with JAX information
- [ ] Add migration guide for users
- [ ] Update API documentation
- [ ] Create performance comparison documentation
- [ ] Deprecation plan for Numba/CuPy version (if needed)
- [ ] **Deliverable:** Complete documentation, user migration path

### 3.2 Testing Strategy

#### Parallel Testing Approach
Maintain both implementations during migration:
```python
# Example testing structure
def test_signal_fft_parity():
    """Test that JAX implementation matches NumPy implementation."""
    x = np.random.randn(100, 100)

    # Original NumPy/SciPy
    result_numpy = fft_transform_numpy(x)

    # New JAX implementation
    result_jax = jax.device_get(fft_transform_jax(x))

    np.testing.assert_allclose(result_numpy, result_jax, rtol=1e-5)
```

#### Regression Testing
- [ ] Capture outputs from original implementation on test dataset
- [ ] Compare outputs from JAX implementation (numerical tolerance ~1e-5)
- [ ] Performance regression tests (new code should not be significantly slower)

#### Integration Testing
- [ ] End-to-end denoising pipeline tests
- [ ] Test with real neutron imaging data
- [ ] Test various sinogram sizes and properties
- [ ] CPU-only testing (verify FAQ requirement is met)
- [ ] GPU testing on available hardware

### 3.3 Performance Benchmarking

Create comprehensive benchmark suite:

```python
# Benchmark structure
def benchmark_comparison():
    """Compare performance of original vs JAX implementation."""
    test_cases = [
        ("small", (256, 512)),
        ("medium", (512, 1024)),
        ("large", (1024, 2048)),
    ]

    results = {}
    for name, shape in test_cases:
        # Original implementation
        time_original = time_function(original_impl, shape)

        # JAX implementation (CPU)
        time_jax_cpu = time_function(jax_impl_cpu, shape)

        # JAX implementation (GPU)
        time_jax_gpu = time_function(jax_impl_gpu, shape)

        results[name] = {
            "original": time_original,
            "jax_cpu": time_jax_cpu,
            "jax_gpu": time_jax_gpu,
        }

    return results
```

Expected performance outcomes:
- **CPU**: JAX should match or slightly exceed Numba performance after JIT warmup
- **GPU**: JAX should match or exceed CuPy performance due to XLA optimizations
- **Memory**: JAX may use more memory initially due to XLA compilation, but can be optimized

---

## 4. Implementation Guidelines

### 4.1 JAX Coding Patterns

#### Pattern 1: Replace Numba JIT with JAX JIT

**Before (Numba):**
```python
from numba import njit

@njit
def compute_distance(patch1, patch2):
    return np.sum((patch1 - patch2) ** 2)
```

**After (JAX):**
```python
import jax
import jax.numpy as jnp

@jax.jit
def compute_distance(patch1, patch2):
    return jnp.sum((patch1 - patch2) ** 2)
```

#### Pattern 2: Replace Loops with Vectorization

**Before (Numba with loops):**
```python
@njit
def process_patches(patches):
    results = np.zeros(len(patches))
    for i in range(len(patches)):
        results[i] = process_single_patch(patches[i])
    return results
```

**After (JAX with vmap):**
```python
@jax.jit
def process_single_patch(patch):
    # ... computation ...
    return result

# Vectorize over first axis
process_patches = jax.vmap(process_single_patch)
```

#### Pattern 3: Replace CuPy with JAX

**Before (CuPy):**
```python
import cupy as cp

def denoise_gpu(data):
    # Move to GPU
    data_gpu = cp.asarray(data)

    # Process on GPU
    result_gpu = cp.fft.fft2(data_gpu)
    result_gpu = cp.where(cp.abs(result_gpu) > threshold, result_gpu, 0)

    # Move back to CPU
    result = cp.asnumpy(result_gpu)

    # Manual cleanup
    del data_gpu, result_gpu
    cp.get_default_memory_pool().free_all_blocks()

    return result
```

**After (JAX - auto CPU/GPU):**
```python
import jax.numpy as jnp

@jax.jit  # Will run on GPU if available, CPU otherwise
def denoise_jax(data):
    result = jnp.fft.fft2(data)
    result = jnp.where(jnp.abs(result) > threshold, result, 0)
    return result  # JAX handles memory automatically

# Use it - same code for CPU or GPU
result = denoise_jax(data)
```

#### Pattern 4: Handling Dynamic Shapes

**Problem:**
```python
# This may not JIT well if num_patches varies
def process_variable_patches(patches):  # shape varies
    return jnp.sum(patches, axis=0)
```

**Solutions:**

a) **Padding approach:**
```python
@jax.jit
def process_patches_padded(patches, num_valid):
    """Process patches with padding for fixed shape."""
    # patches is padded to max_patches
    # num_valid indicates actual number of patches
    mask = jnp.arange(patches.shape[0]) < num_valid
    return jnp.sum(patches * mask[:, None, None], axis=0)
```

b) **Static argument approach:**
```python
@partial(jax.jit, static_argnums=(1,))
def process_patches_static(patches, num_patches):
    """num_patches is static, known at compile time."""
    return jnp.sum(patches[:num_patches], axis=0)
```

#### Pattern 5: Parallel Operations

**Before (Numba parallel):**
```python
@njit(parallel=True)
def parallel_process(data):
    result = np.zeros_like(data)
    for i in prange(data.shape[0]):
        result[i] = expensive_operation(data[i])
    return result
```

**After (JAX):**
```python
# Option 1: vmap (automatic parallelization)
@jax.jit
def parallel_process(data):
    return jax.vmap(expensive_operation)(data)

# Option 2: pmap (explicit multi-device parallelism)
@jax.pmap
def parallel_process_multi_gpu(data):
    return expensive_operation(data)
```

### 4.2 Key JAX Concepts

#### Immutability
JAX arrays are immutable. Use functional updates:
```python
# Don't do this:
# array[i] = new_value  # Error in JAX!

# Do this:
array = array.at[i].set(new_value)
```

#### Pure Functions
Functions must be pure (no side effects):
```python
# Don't do this:
global_state = []
def impure_function(x):
    global_state.append(x)  # Side effect!
    return x + 1

# Do this:
def pure_function(x, state):
    new_state = state + [x]
    return x + 1, new_state
```

#### Random Numbers
JAX uses explicit PRNG keys:
```python
# Don't do this:
# x = np.random.randn(100)  # Implicit global state

# Do this:
key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
x = jax.random.normal(subkey, (100,))
```

### 4.3 Performance Optimization Tips

1. **JIT Compilation** ([JAX JIT Compilation](https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html))
   - First call will be slow (compilation)
   - Subsequent calls are fast
   - Use `static_argnums` for arguments that don't change often

2. **Device Memory** ([JAX Memory Management](https://jax.readthedocs.io/en/latest/faq.html#how-to-use-jax-with-limited-memory))
   - Use `donate_argnums` to donate input buffers
   - JAX automatically manages memory (no manual cleanup needed)

3. **Batch Operations**
   - Use `vmap` instead of loops
   - Larger batches usually perform better

4. **Avoid Host-Device Transfers**
   - Keep data on device between operations
   - Only transfer final results back to host

---

## 5. Migration Checklist

### Pre-Migration
- [ ] Set up JAX development environment
- [ ] Create feature branch for JAX migration
- [ ] Document current performance baselines
- [ ] Establish test data set for validation

### Phase 1: Foundation (Weeks 1-2)
- [ ] Install JAX in CI/CD pipeline
- [ ] Create `jax_utils.py` with helper functions
- [ ] Migrate `signal.py` to JAX
- [ ] Migrate `noise_analysis.py` to JAX
- [ ] Add JAX tests for migrated modules
- [ ] Benchmark migrated modules

### Phase 2: Numba Components (Weeks 3-5)
- [ ] Migrate `block_matching.py` to JAX
  - [ ] `find_candidate_patch_ids`
  - [ ] `get_signal_patch_positions`
  - [ ] `get_patch_numba`
  - [ ] `compute_variance_weighted_distance_matrix`
  - [ ] `compute_distance_matrix_no_variance`
  - [ ] `form_hyper_blocks_from_distance_matrix`
  - [ ] `form_hyper_blocks_from_two_images`
- [ ] Migrate `aggregation.py` to JAX
  - [ ] `aggregate_block_to_image`
  - [ ] `aggregate_denoised_block_to_image`
- [ ] Migrate Numba functions in `utils.py`
- [ ] Tests pass for all migrated components
- [ ] Benchmarks show acceptable performance

### Phase 3: GPU Operations (Weeks 6-7)
- [ ] Migrate `denoiser_gpu.py` to JAX
  - [ ] `shrinkage_fft`
  - [ ] `shrinkage_hadamard`
  - [ ] `collaborative_wiener_filtering`
  - [ ] `collaborative_hadamard_filtering`
- [ ] Update `bm3d.py` to use JAX modules
- [ ] Test CPU execution (no GPU required)
- [ ] Test GPU execution
- [ ] Verify CuPy can be removed from dependencies

### Phase 4: Integration (Weeks 8-9)
- [ ] Full end-to-end tests pass
- [ ] Performance meets or exceeds original implementation
- [ ] Memory usage is acceptable
- [ ] CPU-only mode works correctly
- [ ] GPU mode works correctly
- [ ] All edge cases handled

### Phase 5: Documentation (Weeks 10-11)
- [ ] Update README.md
- [ ] Update docs/FAQ.md
- [ ] Update docs/developer.rst
- [ ] Add JAX migration guide
- [ ] Update API documentation
- [ ] Create performance comparison document
- [ ] Update installation instructions

### Post-Migration
- [ ] Remove Numba from dependencies
- [ ] Remove CuPy from dependencies
- [ ] Update conda recipe
- [ ] Create release notes
- [ ] Tag release
- [ ] Publish to PyPI and Conda

---

## 6. Risk Assessment

### High Risk Items
1. **Dynamic Shape Handling**
   - **Risk:** Variable number of patches per group may not JIT efficiently
   - **Mitigation:** Use padding or static compilation strategies
   - **Contingency:** Keep critical dynamic sections out of JIT

2. **Performance Regression**
   - **Risk:** JAX implementation may be slower than highly optimized Numba/CuPy
   - **Mitigation:** Extensive benchmarking and optimization in Phase 4
   - **Contingency:** Keep Numba/CuPy version as fallback for 1-2 releases

3. **Breaking API Changes**
   - **Risk:** JAX requires functional programming, may affect user API
   - **Mitigation:** Maintain backward-compatible wrapper functions
   - **Contingency:** Provide compatibility layer

### Medium Risk Items
4. **Learning Curve**
   - **Risk:** Team unfamiliar with JAX/functional programming
   - **Mitigation:** Training sessions, pair programming, incremental approach
   - **Contingency:** Extend timeline if needed

5. **Dependencies and Version Compatibility**
   - **Risk:** JAX version compatibility issues
   - **Mitigation:** Pin JAX version, test across versions
   - **Contingency:** Use specific JAX version until stable

6. **Test Coverage**
   - **Risk:** Tests may not catch all edge cases in migration
   - **Mitigation:** Extensive regression testing, real-world data testing
   - **Contingency:** Extended testing phase

### Low Risk Items
7. **Hadamard Transform Implementation**
   - **Risk:** SciPy Hadamard not in JAX
   - **Mitigation:** Implement custom Hadamard or use alternative
   - **Contingency:** Use SciPy version on CPU, implement custom for GPU

8. **Documentation**
   - **Risk:** Documentation becomes outdated
   - **Mitigation:** Update docs incrementally during migration
   - **Contingency:** Documentation sprint at end

---

## 7. Success Criteria

### Functional Requirements
- [ ] All existing features work correctly
- [ ] All existing tests pass with numerical tolerance
- [ ] CPU-only execution works (addresses FAQ requirement)
- [ ] GPU execution works on CUDA 11.x and 12.x

### Performance Requirements
- [ ] CPU performance: Within 20% of Numba implementation
- [ ] GPU performance: Matches or exceeds CuPy implementation
- [ ] Memory usage: No more than 50% increase over current implementation
- [ ] Compilation time: JIT compilation completes in reasonable time (<30s for full pipeline)

### Code Quality Requirements
- [ ] Test coverage maintained or improved
- [ ] Code follows JAX best practices
- [ ] Documentation complete and accurate
- [ ] No Numba or CuPy dependencies remain

### User Experience Requirements
- [ ] Installation simplified (single JAX dependency vs Numba+CuPy)
- [ ] API remains backward compatible or has clear migration path
- [ ] Performance improvement or parity documented
- [ ] Clear error messages for common issues

---

## 8. References and Resources

### Official Documentation
1. **JAX Documentation**: https://jax.readthedocs.io/
2. **JAX GitHub**: https://github.com/google/jax
3. **JAX Tutorials**: https://jax.readthedocs.io/en/latest/jax-101/index.html
4. **JAX API Reference**: https://jax.readthedocs.io/en/latest/jax.html

### Research Papers
5. **JAX: Composable transformations of Python+NumPy programs** (2018)
   - Bradbury et al., http://github.com/google/jax
   - Foundational paper introducing JAX

6. **Compiling machine learning programs via high-level tracing** (SysML 2018)
   - Roy Frostig et al., https://mlsys.org/Conferences/2019/doc/2019/146.pdf
   - Discusses XLA compilation in JAX context

### Tutorials and Guides
7. **JAX Quickstart**: https://jax.readthedocs.io/en/latest/notebooks/quickstart.html
8. **JAX 101 Tutorial Series**: https://jax.readthedocs.io/en/latest/jax-101/index.html
9. **From PyTorch to JAX**: https://www.datahubbs.com/from-pytorch-to-jax/
10. **JAX for the Impatient**: https://flax.readthedocs.io/en/latest/notebooks/jax_for_the_impatient.html

### Best Practices
11. **JAX Sharp Bits**: https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html
    - Critical reading for avoiding common pitfalls
12. **JAX FAQ**: https://jax.readthedocs.io/en/latest/faq.html
13. **How to Think in JAX**: https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html

### Performance
14. **JAX Benchmarks**: https://github.com/google/jax/tree/main/benchmarks
15. **XLA Performance Guide**: https://www.tensorflow.org/xla/performance

### Community Resources
16. **JAX Discussions**: https://github.com/google/jax/discussions
17. **JAX on Stack Overflow**: https://stackoverflow.com/questions/tagged/jax

### Comparable Migrations
18. **Neural Tangents (Google)**: https://github.com/google/neural-tangents
    - Large-scale library built entirely in JAX
19. **RLax (DeepMind)**: https://github.com/deepmind/rlax
    - Reinforcement learning library using JAX
20. **Flax (Google)**: https://github.com/google/flax
    - Neural network library demonstrating JAX best practices

### Related to BM3D
21. **Original BM3D Paper**: "Image Denoising by Sparse 3D Transform-Domain Collaborative Filtering"
    - Dabov et al., IEEE TIP 2007
    - https://ieeexplore.ieee.org/document/4271520

22. **scikit-image BM3D discussions**: Potential future JAX integration discussions
    - https://github.com/scikit-image/scikit-image/issues

---

## 9. Timeline Summary

| Phase | Duration | Key Milestones |
|-------|----------|----------------|
| Phase 1: Foundation | 2 weeks | JAX installed, signal.py and noise_analysis.py migrated |
| Phase 2: Numba Components | 3 weeks | block_matching.py, aggregation.py, utils.py migrated |
| Phase 3: GPU Operations | 2 weeks | denoiser_gpu.py migrated, CuPy removed |
| Phase 4: Integration | 2 weeks | Full testing, optimization, benchmarking complete |
| Phase 5: Documentation | 2 weeks | All docs updated, migration guide created |
| **Total** | **11 weeks** | **Production-ready JAX implementation** |

---

## 10. Conclusion

Migrating BM3DORNL to JAX is a strategic investment that will:

1. **Simplify the codebase**: One unified library instead of NumPy+Numba+CuPy
2. **Improve portability**: CPU and GPU with the same code
3. **Enable new capabilities**: Automatic differentiation for future research
4. **Maintain performance**: JAX/XLA matches or exceeds hand-tuned implementations
5. **Future-proof the library**: JAX is actively developed and widely adopted

The 11-week phased migration plan minimizes risk through:
- Incremental changes with continuous testing
- Parallel testing to ensure correctness
- Performance benchmarking at each phase
- Clear rollback points if issues arise

With careful execution and the detailed guidelines provided in this document, BM3DORNL can successfully transition to a modern, high-performance JAX-based architecture.

---

## Appendix A: Code Examples

### Example 1: Block Matching Migration

**Before (Numba):**
```python
@njit
def find_candidate_patch_ids(
    signal_patches: np.ndarray, ref_index: int, cut_off_distance: Tuple
) -> List[int]:
    num_patches = signal_patches.shape[0]
    ref_pos = signal_patches[ref_index]
    candidate_patch_ids = [ref_index]

    for i in range(ref_index + 1, num_patches):
        if (
            np.abs(signal_patches[i, 0] - ref_pos[0]) <= cut_off_distance[0]
            and np.abs(signal_patches[i, 1] - ref_pos[1]) <= cut_off_distance[1]
        ):
            candidate_patch_ids.append(i)

    return candidate_patch_ids
```

**After (JAX):**
```python
@jax.jit
def find_candidate_patch_ids(
    signal_patches: jnp.ndarray, ref_index: int, cut_off_distance: Tuple[int, int]
) -> jnp.ndarray:
    """
    Vectorized version using JAX.
    Returns array of candidate indices (with -1 for empty slots).
    """
    num_patches = signal_patches.shape[0]
    ref_pos = signal_patches[ref_index]

    # Vectorized distance check
    indices = jnp.arange(num_patches)
    within_row = jnp.abs(signal_patches[:, 0] - ref_pos[0]) <= cut_off_distance[0]
    within_col = jnp.abs(signal_patches[:, 1] - ref_pos[1]) <= cut_off_distance[1]
    after_ref = indices > ref_index

    # Combine conditions
    is_candidate = within_row & within_col & after_ref
    is_candidate = is_candidate.at[ref_index].set(True)  # Include reference

    # Return indices of candidates (pad with -1 for fixed size)
    candidate_indices = jnp.where(is_candidate, indices, -1)
    candidate_indices = candidate_indices[candidate_indices >= 0]

    return candidate_indices
```

### Example 2: GPU Shrinkage Migration

**Before (CuPy):**
```python
def shrinkage_fft(
    hyper_blocks: np.ndarray, variance_blocks: np.ndarray, threshold_factor: float = 3
) -> np.ndarray:
    # Send data to GPU
    denoised_hyper_blocks = cp.asarray(hyper_blocks)
    variance_blocks = cp.asarray(variance_blocks)

    # Compute the threshold for shrinkage
    threshold = threshold_factor * cp.sqrt(variance_blocks)

    # Transform the hyper blocks to the frequency domain
    denoised_hyper_blocks = cp.fft.fftn(denoised_hyper_blocks, axes=(-3, -2, -1))

    # Apply shrinkage (hard thresholding) in the frequency domain
    denoised_hyper_blocks = cp.where(
        np.abs(denoised_hyper_blocks) > threshold, denoised_hyper_blocks, 0
    )

    # Apply inverse 3D FFT to obtain the denoised hyper blocks
    denoised_hyper_blocks = cp.fft.ifftn(denoised_hyper_blocks, axes=(-3, -2, -1)).real

    # Send data back to CPU
    hyper_blocks = cp.asnumpy(denoised_hyper_blocks)

    # Clear memory cache
    del denoised_hyper_blocks, variance_blocks, threshold
    memory_cleanup()

    return hyper_blocks
```

**After (JAX):**
```python
@jax.jit  # Automatically runs on GPU if available
def shrinkage_fft(
    hyper_blocks: jnp.ndarray,
    variance_blocks: jnp.ndarray,
    threshold_factor: float = 3
) -> jnp.ndarray:
    """
    JAX version - works on CPU or GPU with same code.
    No manual memory management needed.
    """
    # Compute the threshold for shrinkage
    threshold = threshold_factor * jnp.sqrt(variance_blocks)

    # Transform the hyper blocks to the frequency domain
    denoised_hyper_blocks = jnp.fft.fftn(hyper_blocks, axes=(-3, -2, -1))

    # Apply shrinkage (hard thresholding) in the frequency domain
    denoised_hyper_blocks = jnp.where(
        jnp.abs(denoised_hyper_blocks) > threshold,
        denoised_hyper_blocks,
        0
    )

    # Apply inverse 3D FFT to obtain the denoised hyper blocks
    denoised_hyper_blocks = jnp.fft.ifftn(denoised_hyper_blocks, axes=(-3, -2, -1)).real

    return denoised_hyper_blocks
```

---

*Document created: 2025-10-29*
*Version: 1.0*
*Author: BM3DORNL Development Team*
