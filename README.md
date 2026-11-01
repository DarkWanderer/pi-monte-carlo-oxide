# mc_pi

Monte-Carlo π on the GPU with [cuda-oxide](https://nvidia.github.io/cuda-rust/cuda-oxide/): the
`#[kernel]` and the host code live in one Rust file (`src/main.rs`) and are built by `cargo oxide`.

The quarter circle's area is estimated with **stratified (jittered) sampling**. The unit square is
split into an M×M grid of cells with two random points per cell, and the fraction of points inside
the circle estimates π/4. Only cells on the circle's edge add variance, so the error falls as
N^(-3/4) instead of the N^(-1/2) of plain hit-or-miss:

| N               | plain σ | stratified σ | RTX 3090 Ti time |
| --------------- | ------- | ------------ | ---------------- |
| 2^32            | 2.5e-5  | 1.4e-7       | 14 ms            |
| 2^36            | 6.3e-6  | 1.8e-8       | 0.22 s           |
| 2^40            | 1.6e-6  | 2.3e-9       | 3.3 s            |
| 2^41.8 (12 s)   | 8.4e-7  | 8.9e-10      | 11.6 s           |

The estimate stays random and unbiased, integer-only, and bit-exact between GPU and CPU.

- `src/mc.rs`: pure logic shared by host and kernel.
  - Counter-based SplitMix64 PRNG. Pair `p` owns samples `2p` and `2p+1` and cell `(p % M, p / M)`;
    the PRNG output's two halves give the offsets inside the cell.
  - Integer-only midpoint hit test `x(x+1) + y(y+1) < R²`, i.e. `(x+½)² + (y+½)² < R²`. Testing the
    cell corner instead biases π̂ by +4/R ≈ 1.9e-9, which is larger than σ at 2^40.
  - `Grid`: `M` cells per side (any value, so N = 2M² can match a time budget finely) over a 2^31
    lattice; `RunConfig`: a grid within 2^18..=2^50 samples, split into launch `Chunk`s; `Budget`:
    the CLI argument.
  - `thread_stats`: per-thread hits and *disagreements* (pairs whose two samples differ).
- `src/main.rs`: the `count_hits` kernel and the host orchestration. The kernel samples per thread,
  reduces each warp with shuffles, and lane 0 of every warp adds the two counts into the block's
  `DeviceAtomicU32` slots (relaxed, no shared memory or barrier). The kernel has no `unsafe`; the
  only `unsafe` in the program is `kernels::load`. Blocks add into their slots, so the host zeroes
  the counters before every launch.

**Error estimate.** The expected number of disagreeing pairs D equals the variance of the hit count
exactly, so the standard error is measured as σ̂ = 4·√D / N: integer-only, no analytic formula.
The printed π̂ is truncated (not rounded) to the digits σ̂ leaves meaningful: σ̂ = 1e-7 keeps 6
decimals, σ̂ = 8.9e-10 keeps 9.

**Time budget.** The default run is sized by GPU time: after a warm-up, one 2^34-sample launch
measures throughput, then N ≈ throughput × budget × 0.95. A stratified estimate cannot be cut short
(stopping partway leaves part of the grid unsampled and biases the result), so there is no hard
cutoff; the reported GPU time can overrun slightly if the GPU throttles.

**Chunked launches.** Each launch covers at most 2^18 blocks (2^36 samples, about 0.22 s). WSL2 may
be subject to the ~2 s Windows TDR watchdog, which one long kernel would hit.

The run is checked two ways: the first 16 blocks are recomputed on the CPU and must match the GPU
exactly (hits and disagreements), and the estimate must be within 5σ̂ of π. The exit code is 1 on
failure. A single-threaded CPU run over 2^26 samples gives the speedup baseline.

## Prerequisites

Follow the [cuda-oxide installation guide](https://nvidia.github.io/cuda-rust/cuda-oxide/), then
check with `cargo oxide doctor`. The toolchain is pinned in `rust-toolchain.toml`: the cuda-oxide
backend is a rustc plugin and refuses to build on any other nightly.

## Run

```bash
cargo oxide run                 # 12 s of GPU time
cargo oxide run -- 15s          # 15 s of GPU time (fractions work: 0.5s)
cargo oxide run -- 36           # N ≈ 2^36 samples (log2 of the count, 18..=50); reproducible
```

## Test

```bash
cargo test
```

The unit tests are host-only. `cargo oxide test` also works but only with `-- --release`: a debug
build is rejected by device codegen with "atomic ordering must be a compile-time constant" for the
`fetch_add` calls in `count_hits`.

## Sample output

RTX 3090 Ti under WSL2, `cargo oxide run`:

```text
 INFO Calibration: 3.344e11 samples/s, so 12.0 s budget -> N = 2^41.79 (M = 1380582)
 INFO π ≈ 3.141592654  |error| = 8.612e-10  σ̂ = 8.907e-10  (plain MC σ at the same N: 8.411e-7, gain ×944)
 INFO GPU: N = 2^41.79 samples in 11.625 s (3.279e11 samples/s)
 INFO CPU: 1 thread, 2^26 samples (4.523e8 samples/s)  GPU speedup ×725
 INFO PASSED: GPU matches CPU on the first 16 blocks, estimate within 5σ̂
```
