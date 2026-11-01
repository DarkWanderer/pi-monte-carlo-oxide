mod mc;

use std::hint::black_box;
use std::io::IsTerminal;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use tracing::{error, info, warn};

use mc::{
    BLOCK_THREADS, Budget, Grid, MAX_CHUNK_BLOCKS, PAIRS_PER_THREAD, RunConfig, Stats, estimate_pi,
    pi_std_error, samples_for_time, stratified_std_error, thread_stats, truncate_to_error,
};

const SEED: u64 = 0x5EED_C0DE_2026;
const DEFAULT_BUDGET: Duration = Duration::from_secs(12);
/// Pays for module/JIT setup and clock ramp-up before anything is timed.
const WARM_UP_LOG2_SAMPLES: u32 = 30;
/// Long enough (~40 ms) to give a stable throughput, short enough to be free
/// next to a multi-second budget.
const CALIBRATION_LOG2_SAMPLES: u32 = 34;
const CPU_BASELINE_LOG2_SAMPLES: u32 = 26;
/// Blocks of chunk 0 recomputed on the CPU and compared bit-for-bit with the GPU.
const VERIFY_BLOCKS: usize = 16;
const SIGMA_TOLERANCE: f64 = 5.0;

fn warp_sum(mut v: u32) -> u32 {
    v += warp::shuffle_xor(v, 16);
    v += warp::shuffle_xor(v, 8);
    v += warp::shuffle_xor(v, 4);
    v += warp::shuffle_xor(v, 2);
    v += warp::shuffle_xor(v, 1);
    v
}

#[cuda_module]
mod kernels {
    use super::*;
    use crate::mc::{Grid, thread_stats};

    /// Each block writes the in-circle count and the disagreeing-pair count of
    /// its threads. `block_offset` is the launch's first block in the whole run;
    /// outputs are indexed by the launch-local block.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn count_hits(
        seed: u64,
        cells_per_side: u32,
        block_offset: u64,
        mut block_hits: DisjointSlice<u32>,
        mut block_disagreements: DisjointSlice<u32>,
    ) {
        static mut WARP_HITS: SharedArray<u32, 8> = SharedArray::UNINIT;
        static mut WARP_DISAGREEMENTS: SharedArray<u32, 8> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x();
        let block = thread::blockIdx_x();
        let gid = (block_offset + block as u64) * thread::blockDim_x() as u64 + tid as u64;

        let grid = Grid::from_validated(cells_per_side);
        let stats = thread_stats(seed, gid, grid, grid.pairs());
        let hits = warp_sum(stats.hits);
        let disagreements = warp_sum(stats.disagreements);

        if warp::lane_id() == 0 {
            // SAFETY: one lane-0 writer per warp slot (tid / 32 < 8).
            unsafe {
                WARP_HITS[(tid / 32) as usize] = hits;
                WARP_DISAGREEMENTS[(tid / 32) as usize] = disagreements;
            }
        }
        thread::sync_threads();

        if tid == 0 {
            let mut total_hits = 0;
            let mut total_disagreements = 0;
            let mut w = 0;
            while w < 8 {
                // SAFETY: all slots were written before the barrier above.
                unsafe {
                    total_hits += WARP_HITS[w];
                    total_disagreements += WARP_DISAGREEMENTS[w];
                }
                w += 1;
            }
            // SAFETY: only thread 0 of each block writes, at the block's own
            // index, and the launch grid has at most `block_hits.len()` blocks
            // (the host sizes both buffers for a full chunk).
            unsafe {
                *block_hits.get_unchecked_mut(block as usize) = total_hits;
                *block_disagreements.get_unchecked_mut(block as usize) = total_disagreements;
            }
        }
    }
}

/// Device state reused across launches.
struct Gpu {
    stream: Arc<CudaStream>,
    module: kernels::LoadedModule,
    // Sized for one full chunk, so a run never allocates per launch.
    block_hits: DeviceBuffer<u32>,
    block_disagreements: DeviceBuffer<u32>,
}

/// Totals of a whole run, plus the first blocks' counts for CPU verification.
struct GpuRun {
    hits: u64,
    disagreements: u64,
    /// Sum of per-launch kernel times; host copies between launches are excluded.
    secs: f64,
    head: Vec<Stats>,
}

impl Gpu {
    fn new() -> Result<Self> {
        let ctx = CudaContext::new(0).context("failed to initialise CUDA device 0")?;
        let stream = ctx.default_stream();
        let len = MAX_CHUNK_BLOCKS.get() as usize;
        let block_hits = DeviceBuffer::<u32>::zeroed(&stream, len)?;
        let block_disagreements = DeviceBuffer::<u32>::zeroed(&stream, len)?;
        // SAFETY: this package owns the embedded device bundle produced for the
        // kernels module above.
        let module = unsafe { kernels::load(&ctx)? };
        Ok(Self {
            stream,
            module,
            block_hits,
            block_disagreements,
        })
    }

    /// Runs every chunk of `cfg`. There is no mid-run cutoff: stopping early
    /// would leave part of the grid unsampled, which biases a stratified estimate.
    fn run(&mut self, cfg: RunConfig) -> Result<GpuRun> {
        let mut run = GpuRun {
            hits: 0,
            disagreements: 0,
            secs: 0.0,
            head: Vec::new(),
        };
        for chunk in cfg.chunks() {
            let blocks = chunk.blocks.get();
            let prepared =
                self.module
                    .prepare_count_hits(LaunchConfig1D::new(blocks, BLOCK_THREADS, 0))?;

            let start = self
                .stream
                .record_event(Some(CUevent_flags_enum_CU_EVENT_DEFAULT))?;
            self.module.count_hits(
                &self.stream,
                &prepared,
                SEED,
                cfg.grid().cells_per_side(),
                chunk.first_block,
                &mut self.block_hits,
                &mut self.block_disagreements,
            )?;
            let end = self
                .stream
                .record_event(Some(CUevent_flags_enum_CU_EVENT_DEFAULT))?;
            run.secs += start.elapsed_ms(&end)? as f64 / 1e3;

            // The buffers hold a full chunk; only the launched blocks are written.
            let hits = self.block_hits.to_host_vec(&self.stream)?;
            let disagreements = self.block_disagreements.to_host_vec(&self.stream)?;
            let launched = blocks as usize;
            run.hits += hits[..launched].iter().map(|&h| h as u64).sum::<u64>();
            run.disagreements += disagreements[..launched]
                .iter()
                .map(|&d| d as u64)
                .sum::<u64>();
            if chunk.first_block == 0 {
                run.head = hits
                    .iter()
                    .zip(&disagreements)
                    .take(launched.min(VERIFY_BLOCKS))
                    .map(|(&hits, &disagreements)| Stats {
                        hits,
                        disagreements,
                    })
                    .collect();
            }
        }
        Ok(run)
    }

    /// Measured throughput in samples/s.
    fn calibrate(&mut self) -> Result<f64> {
        let cfg = RunConfig::for_samples(1 << CALIBRATION_LOG2_SAMPLES)?;
        let run = self.run(cfg)?;
        Ok(cfg.total_samples().get() as f64 / run.secs)
    }
}

/// CPU reference for one block's counts. Chunk 0 starts at global block 0, so
/// block indices coincide.
fn cpu_block_stats(block: u64, cfg: RunConfig) -> Stats {
    let grid = cfg.grid();
    (0..BLOCK_THREADS as u64)
        .map(|t| thread_stats(SEED, block * BLOCK_THREADS as u64 + t, grid, grid.pairs()))
        .fold(Stats::default(), |acc, s| acc + s)
}

fn verify(cfg: RunConfig, head: &[Stats], pi_hat: f64, sigma: f64) -> Result<()> {
    for (block, got) in head.iter().enumerate() {
        let want = cpu_block_stats(block as u64, cfg);
        ensure!(
            *got == want,
            "block {block}: GPU counted {got:?}, CPU counted {want:?}"
        );
    }
    let error = (pi_hat - std::f64::consts::PI).abs();
    ensure!(
        error < SIGMA_TOLERANCE * sigma,
        "|π̂ - π| = {error:.3e} exceeds {SIGMA_TOLERANCE}σ̂ = {:.3e}",
        SIGMA_TOLERANCE * sigma
    );
    Ok(())
}

fn log2_samples(cfg: RunConfig) -> f64 {
    (cfg.total_samples().get() as f64).log2()
}

fn run(budget: Budget) -> Result<()> {
    let mut gpu = Gpu::new()?;
    gpu.run(RunConfig::for_samples(1 << WARM_UP_LOG2_SAMPLES)?)?;

    let cfg = match budget {
        Budget::Samples(cfg) => cfg,
        Budget::Time(budget) => {
            let rate = gpu.calibrate()?;
            let cfg = RunConfig::for_samples(samples_for_time(rate, budget))?;
            info!(
                "Calibration: {rate:.3e} samples/s, so {:.1} s budget -> N = 2^{:.2} (M = {})",
                budget.as_secs_f64(),
                log2_samples(cfg),
                cfg.grid().cells_per_side()
            );
            cfg
        }
    };
    let total = cfg.total_samples();

    let run = gpu.run(cfg)?;
    let pi_hat = estimate_pi(run.hits, total);
    let sigma = stratified_std_error(run.disagreements, total);
    let plain_sigma = pi_std_error(total);
    let gpu_rate = total.get() as f64 / run.secs;

    info!(
        "π ≈ {}  |error| = {:.3e}  σ̂ = {sigma:.3e}  (plain MC σ at the same N: {plain_sigma:.3e}, gain ×{:.0})",
        // σ̂ rather than the true error: the latter can be small by luck, and
        // digits kept on that basis would just be π leaking into the output.
        truncate_to_error(pi_hat, sigma),
        (pi_hat - std::f64::consts::PI).abs(),
        plain_sigma / sigma
    );
    info!(
        "GPU: N = 2^{:.2} samples in {:.3} s ({gpu_rate:.3e} samples/s)",
        log2_samples(cfg),
        run.secs
    );
    if let Budget::Time(budget) = budget
        && run.secs > budget.as_secs_f64()
    {
        // Thermal slowdown can make the run slower than the calibration launch.
        warn!(
            "GPU time overran the {:.1} s budget by {:.2} s",
            budget.as_secs_f64(),
            run.secs - budget.as_secs_f64()
        );
    }

    let cpu_grid = Grid::from_budget_samples(1 << CPU_BASELINE_LOG2_SAMPLES)?;
    let cpu_threads = cpu_grid.pairs().div_ceil(PAIRS_PER_THREAD as u64);
    let cpu_start = Instant::now();
    let cpu_stats = (0..cpu_threads)
        .map(|g| thread_stats(SEED, g, cpu_grid, cpu_grid.pairs()))
        .fold(Stats::default(), |acc, s| acc + s);
    black_box(cpu_stats);
    let cpu_rate = cpu_grid.total_samples().get() as f64 / cpu_start.elapsed().as_secs_f64();
    info!(
        "CPU: 1 thread, 2^{CPU_BASELINE_LOG2_SAMPLES} samples ({cpu_rate:.3e} samples/s)  GPU speedup ×{:.0}",
        gpu_rate / cpu_rate
    );

    verify(cfg, &run.head, pi_hat, sigma)
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_target(false)
        .without_time()
        .with_ansi(std::io::stdout().is_terminal())
        .with_writer(std::io::stdout)
        .init();

    // A bad CLI arg is a usage error, not a failed run, so it exits before FAILED reporting.
    let budget = match std::env::args().nth(1) {
        None => Budget::Time(DEFAULT_BUDGET),
        Some(arg) => arg.parse()?,
    };

    match run(budget) {
        Ok(()) => info!(
            "PASSED: GPU matches CPU on the first {VERIFY_BLOCKS} blocks, estimate within {SIGMA_TOLERANCE}σ̂"
        ),
        Err(e) => {
            error!("FAILED: {e:#}");
            std::process::exit(1);
        }
    }
    Ok(())
}
