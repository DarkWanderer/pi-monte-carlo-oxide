//! Stratified Monte-Carlo π logic shared by the host and the `count_hits` kernel.
//!
//! The unit square is a lattice of 2^31 × 2^31 points cut into an M × M grid
//! of cells; every cell gets two random points. Only cells on the circle's edge
//! contribute variance, so the error falls as N^(-3/4) instead of N^(-1/2).
//!
//! Everything here is integer-only so the GPU and the CPU agree bit-for-bit;
//! floating point appears only in the host-side estimate.

use anyhow::{Context, Result, ensure};
use std::num::{NonZeroU32, NonZeroU64};
use std::ops::Add;
use std::str::FromStr;
use std::time::Duration;

pub const BLOCK_THREADS: u32 = 256;
pub const SAMPLES_PER_THREAD: u32 = 1024;
/// Samples come in pairs sharing a cell, so a thread owns half as many pairs.
pub const PAIRS_PER_THREAD: u32 = SAMPLES_PER_THREAD / 2;
const PAIRS_PER_BLOCK: u64 = BLOCK_THREADS as u64 * PAIRS_PER_THREAD as u64;

/// Blocks per kernel launch (2^36 samples, about 0.16 s on a 3090 Ti). WSL2 may
/// enforce the ~2 s Windows TDR watchdog, which one long kernel would hit.
pub const MAX_CHUNK_BLOCKS: NonZeroU32 = nz32(1 << 18);

pub const MIN_LOG2_SAMPLES: u32 = 18;
pub const MAX_LOG2_SAMPLES: u32 = 50;

/// Lattice points per square side. Coordinates stay below 2^31, so the
/// midpoint test's `x(x+1) + y(y+1)` stays below 2^63 in `u64`.
const LATTICE: u32 = 1 << 31;

/// `M` range that `Grid::from_budget_samples` yields for the log2 bounds above.
/// Bounding `M` rather than `N = 2M²` keeps both bounds reachable: the floor in
/// `from_budget_samples` puts `N` just under 2^18 for `"18"`.
const MIN_CELLS_PER_SIDE: u64 = (1u64 << (MIN_LOG2_SAMPLES - 1)).isqrt();
const MAX_CELLS_PER_SIDE: u64 = (1u64 << (MAX_LOG2_SAMPLES - 1)).isqrt();

/// Fraction of the calibrated throughput a time budget is filled with: the
/// stratified estimate cannot be cut short, so the run must not overshoot.
const TIME_BUDGET_HEADROOM: f64 = 0.95;

const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// Const-context `NonZeroU32::new(n).unwrap()`; a zero fails the build.
const fn nz32(n: u32) -> NonZeroU32 {
    match NonZeroU32::new(n) {
        Some(n) => n,
        None => panic!("zero passed to nz32"),
    }
}

/// SplitMix64 step: the stateful generator's output for state `x`.
///
/// Used as a counter-based PRNG (`x = seed + sample_index`) so each sample is
/// independent of how samples are split across threads.
pub fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(GAMMA);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Midpoint hit test for lattice point `(x, y)`: `(x+½)² + (y+½)² < R²`, with
/// `r2 = R²`. Equivalent to `x(x+1) + y(y+1) < R²` for integers.
///
/// Testing the cell corner instead (`x² + y² < R²`) would bias π̂ by +4/R,
/// about 1.9e-9: six σ at 2^44 samples.
///
/// Requires `x, y < 2^31`. The operands are multiplied as widened `u32`s so the
/// GPU emits one `mul.wide.u32` instead of a multi-instruction 64-bit multiply.
pub fn in_quarter_circle(x: u32, y: u32, r2: u64) -> bool {
    let xx = x as u64 * (x + 1) as u64;
    let yy = y as u64 * (y + 1) as u64;
    xx + yy < r2
}

/// Scales 32 random bits to an offset in `[0, w)` (Lemire's multiply-shift).
/// Offsets are non-uniform by at most `w / 2^32`, which moves π̂ by about 1e-12.
fn offset(bits: u32, w: u32) -> u32 {
    ((bits as u64 * w as u64) >> 32) as u32
}

/// Lattice point drawn from PRNG output `r` inside the cell whose origin is
/// `(x0, y0)`: the high 32 bits place `x`, the low 32 bits place `y`. Stays
/// below `x0 + w ≤ R ≤ 2^31`, so `u32` cannot overflow.
pub fn sample_point(r: u64, x0: u32, y0: u32, w: u32) -> (u32, u32) {
    (x0 + offset((r >> 32) as u32, w), y0 + offset(r as u32, w))
}

/// An M × M grid over the 2^31 × 2^31 lattice, valid by construction.
///
/// Cells are `W = 2^31 / M` points wide, so the square side is `R = M·W ≤ 2^31`.
/// `M` need not be a power of two: that lets a time budget pick `N = 2M²` at
/// fine granularity instead of in steps of 2×.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Grid {
    m: NonZeroU32,
}

impl Grid {
    pub fn new(cells_per_side: NonZeroU32) -> Result<Self> {
        ensure!(
            cells_per_side.get() <= LATTICE,
            "{cells_per_side} cells per side exceeds the {LATTICE} lattice points per side"
        );
        Ok(Self { m: cells_per_side })
    }

    /// The largest grid with at most `samples` samples (`2M² ≤ samples`).
    pub fn from_budget_samples(samples: u64) -> Result<Self> {
        // `samples / 2 < 2^63`, so the root is below 2^32 and the cast is lossless.
        let m = NonZeroU32::new((samples / 2).isqrt() as u32);
        Self::new(m.with_context(|| format!("{samples} samples is too few for one cell"))?)
    }

    /// Rebuilds a grid inside the kernel from `cells_per_side()` of one that
    /// `new` accepted. The kernel cannot return errors, so it trusts the host's
    /// validation; a zero falls back to one cell rather than dividing by zero.
    pub fn from_validated(cells_per_side: u32) -> Self {
        Self {
            m: NonZeroU32::new(cells_per_side).unwrap_or(NonZeroU32::MIN),
        }
    }

    pub fn cells_per_side(&self) -> u32 {
        self.m.get()
    }

    /// Lattice points per cell side, `W`.
    pub fn cell_width(&self) -> u32 {
        LATTICE / self.m
    }

    /// `R = M·W ≤ 2^31`: the square's side and the circle's radius.
    pub fn side(&self) -> u32 {
        self.m.get() * self.cell_width()
    }

    /// `R²` for the hit test.
    pub fn r2(&self) -> u64 {
        self.side() as u64 * self.side() as u64
    }

    /// Pairs (one per cell), `M² ≤ 2^62`.
    pub fn pairs(&self) -> u64 {
        self.m.get() as u64 * self.m.get() as u64
    }

    /// `2M²`, at most `2^63`.
    pub fn total_samples(&self) -> NonZeroU64 {
        NonZeroU64::from(self.m)
            .saturating_mul(NonZeroU64::from(self.m))
            .saturating_mul(NonZeroU64::MIN.saturating_add(1))
    }
}

/// What a thread (or a block, or a whole run) reports.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Stats {
    pub hits: u32,
    /// Pairs whose two samples differ. `E[D] = Σ 2q(1-q) = Var(hits)` exactly,
    /// so `D` measures the estimator's own variance without any analytic model.
    pub disagreements: u32,
}

impl Add for Stats {
    type Output = Stats;

    fn add(self, rhs: Stats) -> Stats {
        Stats {
            hits: self.hits + rhs.hits,
            disagreements: self.disagreements + rhs.disagreements,
        }
    }
}

/// Stats of pairs `first..first + count`. Pair `p` owns samples `2p` and
/// `2p + 1` (PRNG counters `seed + 2p` and `seed + 2p + 1`) and cell
/// `(p % M, p / M)`. The range must lie within the grid: `first + count ≤ M²`.
pub fn range_stats(seed: u64, grid: Grid, first: u64, count: u64) -> Stats {
    let mut stats = Stats::default();
    if count == 0 {
        return stats;
    }
    let (w, side, r2) = (grid.cell_width(), grid.side(), grid.r2());

    // One 64-bit div/mod to find the starting cell (both indices are below
    // M ≤ 2^31); after that, origins advance by adding `w` and wrap at the row
    // end, which saves two multiplies per pair.
    let m = grid.cells_per_side() as u64;
    let mut x0 = (first % m) as u32 * w;
    let mut y0 = (first / m) as u32 * w;
    let mut i = 0;
    while i < count {
        let counter = seed.wrapping_add(2 * (first + i));
        let (ax, ay) = sample_point(splitmix64(counter), x0, y0, w);
        let (bx, by) = sample_point(splitmix64(counter.wrapping_add(1)), x0, y0, w);
        let a = in_quarter_circle(ax, ay, r2) as u32;
        let b = in_quarter_circle(bx, by, r2) as u32;
        stats.hits += a + b;
        stats.disagreements += a ^ b;

        x0 += w;
        if x0 == side {
            x0 = 0;
            y0 += w;
        }
        i += 1;
    }
    stats
}

/// Stats for the `PAIRS_PER_THREAD` consecutive pairs owned by thread `gid`.
/// Pairs at or beyond `pairs_total` are skipped, so the last block may be partial.
pub fn thread_stats(seed: u64, gid: u64, grid: Grid, pairs_total: u64) -> Stats {
    let first = gid * PAIRS_PER_THREAD as u64;
    let count = pairs_total
        .saturating_sub(first)
        .min(PAIRS_PER_THREAD as u64);
    range_stats(seed, grid, first, count)
}

pub fn estimate_pi(hits: u64, total: NonZeroU64) -> f64 {
    4.0 * hits as f64 / total.get() as f64
}

/// Standard deviation of plain hit-or-miss `estimate_pi` over `total` samples;
/// the yardstick that the stratified error is compared against.
pub fn pi_std_error(total: NonZeroU64) -> f64 {
    let p = std::f64::consts::FRAC_PI_4;
    4.0 * (p * (1.0 - p) / total.get() as f64).sqrt()
}

/// Measured standard error of the stratified `estimate_pi`: `4·√D / N`, with
/// `D` the total disagreement count (see `Stats::disagreements`).
pub fn stratified_std_error(disagreements: u64, total: NonZeroU64) -> f64 {
    4.0 * (disagreements as f64).sqrt() / total.get() as f64
}

/// Fractional digits an `f64` near π can carry; also what a zero error keeps.
const MAX_DECIMALS: i64 = 15;

/// Fractional digits of an estimate that `error` leaves meaningful: the digit
/// whose place value equals the error's order of magnitude is already noise, so
/// an error of 1e-7 keeps 6.
fn decimals_for_error(error: f64) -> usize {
    if error == 0.0 {
        return MAX_DECIMALS as usize;
    }
    // Read the exponent off `{:e}`: `log10().floor()` can land one below an
    // exact power of ten.
    format!("{error:e}")
        .split_once('e')
        .and_then(|(_, exp)| exp.parse::<i64>().ok())
        .map_or(0, |exp| (-exp - 1).clamp(0, MAX_DECIMALS) as usize)
}

/// `value` cut (not rounded) to the digits its `error` leaves meaningful, so
/// the text never claims precision the estimate lacks. Meant for `value >= 1`.
pub fn truncate_to_error(value: f64, error: f64) -> String {
    // Rust prints the exact binary expansion; below 2^-8 it would need more
    // than 60 digits, but for `value >= 1` it is at most 52, so nothing rounds.
    let exact = format!("{value:.60}");
    let decimals = decimals_for_error(error);
    match exact.split_once('.') {
        Some((int, _)) if decimals == 0 => int.to_string(),
        Some((int, frac)) => format!("{int}.{}", frac.get(..decimals).unwrap_or(frac)),
        None => exact,
    }
}

/// A grid whose sample count is within the supported range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunConfig {
    grid: Grid,
}

/// A kernel launch: global blocks `first_block..first_block + blocks`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Chunk {
    pub first_block: u64,
    pub blocks: NonZeroU32,
}

impl RunConfig {
    pub fn new(grid: Grid) -> Result<Self> {
        let m = grid.cells_per_side() as u64;
        ensure!(
            (MIN_CELLS_PER_SIDE..=MAX_CELLS_PER_SIDE).contains(&m),
            "{m} cells per side ({} samples) is outside the supported range of \
             2^{MIN_LOG2_SAMPLES}..=2^{MAX_LOG2_SAMPLES} samples \
             ({MIN_CELLS_PER_SIDE}..={MAX_CELLS_PER_SIDE} cells per side)",
            grid.total_samples()
        );
        Ok(Self { grid })
    }

    /// The largest valid config with at most `samples` samples.
    pub fn for_samples(samples: u64) -> Result<Self> {
        Self::new(Grid::from_budget_samples(samples)?)
    }

    pub fn grid(&self) -> Grid {
        self.grid
    }

    pub fn total_samples(&self) -> NonZeroU64 {
        self.grid.total_samples()
    }

    /// Blocks needed to cover every pair; the last one may be partial.
    pub fn total_blocks(&self) -> u64 {
        self.grid.pairs().div_ceil(PAIRS_PER_BLOCK)
    }

    /// Launches covering all blocks, each at most `MAX_CHUNK_BLOCKS`.
    pub fn chunks(&self) -> impl Iterator<Item = Chunk> {
        let total = self.total_blocks();
        let step = MAX_CHUNK_BLOCKS.get() as u64;
        (0..total).step_by(step as usize).filter_map(move |first| {
            // Never filtered out: `first < total`, so the remainder is non-zero.
            let blocks = NonZeroU32::new((total - first).min(step) as u32)?;
            Some(Chunk {
                first_block: first,
                blocks,
            })
        })
    }
}

/// Samples that should fill `budget` at `rate` samples/s.
pub fn samples_for_time(rate: f64, budget: Duration) -> u64 {
    (rate * budget.as_secs_f64() * TIME_BUDGET_HEADROOM) as u64
}

/// What the run is sized by: GPU seconds, or a fixed (reproducible) sample count.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Budget {
    Time(Duration),
    Samples(RunConfig),
}

impl FromStr for Budget {
    type Err = anyhow::Error;

    /// `"15s"` is GPU seconds; `"36"` is `N ≈ 2^36` samples.
    fn from_str(s: &str) -> Result<Self> {
        if let Some(secs) = s.strip_suffix('s') {
            let secs: f64 = secs
                .parse()
                .with_context(|| format!("time budget must look like \"12s\", got {s:?}"))?;
            let budget = Duration::try_from_secs_f64(secs)
                .with_context(|| format!("invalid time budget {s:?}"))?;
            ensure!(!budget.is_zero(), "time budget must be positive, got {s:?}");
            return Ok(Self::Time(budget));
        }

        let log2: u32 = s.parse().with_context(|| {
            format!("budget must be seconds like \"12s\" or log2(samples) like \"36\", got {s:?}")
        })?;
        ensure!(
            (MIN_LOG2_SAMPLES..=MAX_LOG2_SAMPLES).contains(&log2),
            "log2(samples) = {log2} is outside {MIN_LOG2_SAMPLES}..={MAX_LOG2_SAMPLES}"
        );
        Ok(Self::Samples(RunConfig::for_samples(1 << log2)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const R_MAX: u64 = LATTICE as u64;

    fn err_text<T>(r: Result<T>) -> String {
        match r {
            Ok(_) => String::from("<ok>"),
            Err(e) => format!("{e:#}"),
        }
    }

    fn grid(m: u32) -> Grid {
        Grid::from_validated(m)
    }

    /// Reference for the midpoint test in exact 128-bit integers.
    fn midpoint_reference(x: u64, y: u64, r: u64) -> bool {
        let (x, y, r) = (x as u128, y as u128, r as u128);
        (2 * x + 1).pow(2) + (2 * y + 1).pow(2) < 4 * r * r
    }

    /// The corner rule that the midpoint test replaced.
    fn corner_hit(x: u64, y: u64, r: u64) -> bool {
        x * x + y * y < r * r
    }

    /// Per-pair definition of `range_stats`, written without the row-stepping.
    fn brute_force(seed: u64, g: Grid, pairs: std::ops::Range<u64>) -> Stats {
        let (m, w, r2) = (g.cells_per_side() as u64, g.cell_width(), g.r2());
        pairs.fold(Stats::default(), |acc, p| {
            // Origins by multiplication, independent of the stepping in `range_stats`.
            let (x0, y0) = ((p % m) as u32 * w, (p / m) as u32 * w);
            let hit = |j: u64| {
                let (x, y) = sample_point(splitmix64(seed.wrapping_add(2 * p + j)), x0, y0, w);
                in_quarter_circle(x, y, r2) as u32
            };
            let (a, b) = (hit(0), hit(1));
            acc + Stats {
                hits: a + b,
                disagreements: a ^ b,
            }
        })
    }

    /// `(hits, disagreements)` over the whole grid, summed thread by thread.
    fn run_on_cpu(seed: u64, g: Grid) -> (u64, u64) {
        let threads = g.pairs().div_ceil(PAIRS_PER_THREAD as u64);
        (0..threads)
            .map(|gid| thread_stats(seed, gid, g, g.pairs()))
            .fold((0, 0), |(h, d), s| {
                (h + s.hits as u64, d + s.disagreements as u64)
            })
    }

    #[test]
    fn splitmix64_matches_reference_seed_0_sequence() {
        // Reference outputs of the stateful SplitMix64 seeded with 0: its state
        // after k steps is k * GAMMA.
        let expected = [
            0xE220_A839_7B1D_CDAF,
            0x6E78_9E6A_A1B9_65F4,
            0x06C4_5D18_8009_454F,
            0xF88B_B8A8_724C_81EC,
        ];
        for (k, want) in expected.into_iter().enumerate() {
            assert_eq!(
                splitmix64((k as u64).wrapping_mul(GAMMA)),
                want,
                "output {k}"
            );
        }
    }

    #[test]
    fn midpoint_test_on_axes_and_extremes() {
        let r2 = R_MAX * R_MAX;
        let top = LATTICE - 1;
        assert!(in_quarter_circle(0, 0, r2));
        assert!(in_quarter_circle(top, 0, r2));
        assert!(in_quarter_circle(0, top, r2));
        assert!(!in_quarter_circle(top, top, r2));
        assert!(!in_quarter_circle(top, top, 1));
        // R = 1: the only lattice point (0, 0) has midpoint (½, ½), inside.
        assert!(in_quarter_circle(0, 0, 1));
    }

    #[test]
    fn midpoint_test_on_the_diagonal_boundary() {
        // (x+½)·√2 < 2^31 ⇔ x < 1518500249.99...
        let r2 = R_MAX * R_MAX;
        assert!(in_quarter_circle(1_518_500_249, 1_518_500_249, r2));
        assert!(!in_quarter_circle(1_518_500_250, 1_518_500_250, r2));
    }

    #[test]
    fn midpoint_test_matches_u128_reference_near_the_boundary() {
        for r in [1, 2, 3, 7, 1000, R_MAX - 1, R_MAX] {
            let stride = (r / 997).max(1);
            let ys = (0..r).step_by(stride as usize).chain([r - 1]);
            let (mut hits, mut misses) = (0, 0);
            for y in ys {
                let edge = ((r * r - y * y) as f64).sqrt() as u64;
                for x in edge.saturating_sub(2)..=(edge + 2).min(r - 1) {
                    let got = in_quarter_circle(x as u32, y as u32, r * r);
                    assert_eq!(got, midpoint_reference(x, y, r), "x={x} y={y} r={r}");
                    if got {
                        hits += 1;
                    } else {
                        misses += 1;
                    }
                }
            }
            if r >= 2 {
                assert!(hits > 0 && misses > 0, "sweep missed the boundary at r={r}");
            }
        }
    }

    #[test]
    fn midpoint_lattice_count_beats_the_corner_rule() {
        let r = 1000u64;
        let count = |hit: fn(u64, u64, u64) -> bool| {
            (0..r)
                .map(|x| (0..r).filter(|&y| hit(x, y, r)).count() as u64)
                .sum::<u64>()
        };
        let pi = |count: u64| 4.0 * count as f64 / (r * r) as f64;
        let midpoint = (pi(count(|x, y, r| {
            in_quarter_circle(x as u32, y as u32, r * r)
        })) - std::f64::consts::PI)
            .abs();
        let corner = (pi(count(corner_hit)) - std::f64::consts::PI).abs();
        // The corner rule is biased by ~4/R; the midpoint rule's lattice error is
        // O(R^(-4/3)).
        assert!(
            midpoint < 2.0 * (r as f64).powf(-4.0 / 3.0),
            "midpoint = {midpoint}"
        );
        assert!(corner > 0.9 * 4.0 / r as f64, "corner = {corner}");
    }

    #[test]
    fn grid_new_checks_lattice_limits() -> Result<()> {
        for m in [1, 2, 3, 7, 1000, 1 << 10, (1 << 31) - 1, 1 << 31] {
            let g = Grid::new(nz32(m))?;
            let (w, r) = (g.cell_width() as u64, m as u64 * g.cell_width() as u64);
            assert_eq!(w, R_MAX / m as u64, "m = {m}");
            assert!(w >= 1 && r <= R_MAX, "m = {m}");
            assert_eq!(g.side() as u64, r, "m = {m}");
            assert_eq!(g.r2(), r * r, "m = {m}");
            assert_eq!(g.pairs(), m as u64 * m as u64, "m = {m}");
            assert_eq!(g.total_samples().get(), 2 * g.pairs(), "m = {m}");
        }
        assert_eq!(Grid::new(nz32(1))?.cell_width(), LATTICE);
        // 2^31 is not divisible by 3, so the square shrinks to R = 3·715827882.
        assert_eq!(Grid::new(nz32(3))?.cell_width(), 715_827_882);
        assert_eq!(Grid::new(nz32(3))?.r2(), 2_147_483_646u64.pow(2));
        assert!(err_text(Grid::new(nz32((1 << 31) + 1))).contains("exceeds"));
        assert!(err_text(Grid::new(nz32(u32::MAX))).contains("exceeds"));
        Ok(())
    }

    #[test]
    fn grid_from_validated_round_trips() -> Result<()> {
        let g = Grid::new(nz32(12_345))?;
        assert_eq!(Grid::from_validated(g.cells_per_side()), g);
        Ok(())
    }

    #[test]
    fn grid_from_budget_samples_takes_the_floor_root() -> Result<()> {
        // 2^35 = 34359738368; floor(sqrt) = 185363 since 185363² = 34359441769.
        assert_eq!(
            Grid::from_budget_samples(1 << 36)?.cells_per_side(),
            185_363
        );
        assert_eq!(Grid::from_budget_samples(2)?.cells_per_side(), 1);
        assert_eq!(Grid::from_budget_samples(8)?.cells_per_side(), 2);
        assert_eq!(Grid::from_budget_samples(17)?.cells_per_side(), 2);
        assert_eq!(Grid::from_budget_samples(18)?.cells_per_side(), 3);
        assert_eq!(
            Grid::from_budget_samples(1 << 63)?.cells_per_side(),
            1 << 31
        );
        assert!(err_text(Grid::from_budget_samples(0)).contains("too few"));
        assert!(err_text(Grid::from_budget_samples(1)).contains("too few"));
        assert!(err_text(Grid::from_budget_samples(u64::MAX)).contains("exceeds"));
        Ok(())
    }

    #[test]
    fn offset_stays_below_cell_width_and_is_near_uniform() {
        for w in [1, 2, 3, 1000, 1 << 11, LATTICE] {
            for bits in [0, 1, 0x7FFF_FFFF, 0x8000_0000, 0xFFFF_FFFF] {
                assert!(offset(bits, w) < w, "bits = {bits}, w = {w}");
            }
        }
        // A power-of-two width keeps the top bits exactly.
        assert_eq!(offset(0xDEAD_BEEF, 1 << 11), 0xDEAD_BEEF >> 21);

        let w = 10;
        let mut buckets = [0u32; 10];
        for k in 0..1u64 << 20 {
            buckets[offset((splitmix64(k) >> 32) as u32, w) as usize] += 1;
        }
        let expected = (1u32 << 20) / 10;
        for (i, &n) in buckets.iter().enumerate() {
            assert!(n.abs_diff(expected) < expected / 100, "bucket {i}: {n}");
        }
    }

    #[test]
    fn both_samples_of_a_pair_land_in_their_cell() {
        for m in [1u32, 7, 1000] {
            let g = grid(m);
            let (m, w) = (m as u64, g.cell_width());
            for p in (0..g.pairs().min(5000)).chain([g.pairs() - 1]) {
                let (x0, y0) = ((p % m) as u32 * w, (p / m) as u32 * w);
                for j in 0..2 {
                    let r = splitmix64(99u64.wrapping_add(2 * p + j));
                    let (x, y) = sample_point(r, x0, y0, w);
                    assert!((x0..x0 + w).contains(&x), "p = {p}, j = {j}");
                    assert!((y0..y0 + w).contains(&y), "p = {p}, j = {j}");
                }
            }
        }
    }

    #[test]
    fn range_stats_matches_per_pair_definition() {
        // M = 40: ranges wrap many rows; M = 1000: a start inside a row.
        for (m, first, count) in [
            (40, 0, 1600),
            (40, 37, 600),
            (1000, 631_808, 512),
            (1, 0, 1),
        ] {
            let g = grid(m);
            assert_eq!(
                range_stats(7, g, first, count),
                brute_force(7, g, first..first + count),
                "m = {m}, first = {first}"
            );
        }
    }

    #[test]
    fn thread_stats_matches_per_pair_definition_with_row_wrap_and_truncation() {
        let g = grid(40);
        let total = g.pairs(); // 1600 pairs: threads 0..=2 are full, 3 has 64 pairs, 4 has none
        let per_thread = PAIRS_PER_THREAD as u64;
        for gid in 0..5 {
            let first = (gid * per_thread).min(total);
            let last = ((gid + 1) * per_thread).min(total);
            assert_eq!(
                thread_stats(42, gid, g, total),
                brute_force(42, g, first..last),
                "gid = {gid}"
            );
        }
        assert_eq!(thread_stats(42, 4, g, total), Stats::default());
        // Far past the end: no pairs, and no overflow from the large start.
        assert_eq!(thread_stats(42, 1 << 40, g, total), Stats::default());
    }

    #[test]
    fn range_stats_is_independent_of_partitioning() {
        let g = grid(40);
        for (first, n) in [(0, 100), (13, 250), (800, 400)] {
            let whole = range_stats(5, g, first, 2 * n);
            let halves = range_stats(5, g, first, n) + range_stats(5, g, first + n, n);
            assert_eq!(whole, halves, "first = {first}");
        }
    }

    #[test]
    fn stats_depend_on_seed() {
        let g = grid(40);
        assert_ne!(range_stats(1, g, 0, 1600), range_stats(2, g, 0, 1600));
    }

    #[test]
    fn estimate_pi_is_four_times_hit_rate() {
        assert_eq!(estimate_pi(1, NonZeroU64::MIN.saturating_add(3)), 1.0);
        assert_eq!(estimate_pi(0, NonZeroU64::MIN.saturating_add(9)), 0.0);
        assert_eq!(estimate_pi(10, NonZeroU64::MIN.saturating_add(9)), 4.0);
    }

    #[test]
    fn std_errors() {
        let a = pi_std_error(NonZeroU64::MIN.saturating_add((1 << 20) - 1));
        let b = pi_std_error(NonZeroU64::MIN.saturating_add((1 << 22) - 1));
        assert!((a / b - 2.0).abs() < 1e-12);

        let n = NonZeroU64::MIN.saturating_add(1599);
        assert!((stratified_std_error(100, n) - 4.0 * 10.0 / 1600.0).abs() < 1e-15);
        assert_eq!(stratified_std_error(0, n), 0.0);
    }

    #[test]
    fn decimals_follow_the_error_exponent() {
        assert_eq!(decimals_for_error(1e-7), 6);
        assert_eq!(decimals_for_error(9.99e-8), 7);
        assert_eq!(decimals_for_error(8.907e-10), 9);
        assert_eq!(decimals_for_error(1.0), 0);
        assert_eq!(decimals_for_error(0.3), 0);
        assert_eq!(decimals_for_error(0.01), 1);
    }

    #[test]
    fn decimals_are_exact_on_every_power_of_ten() {
        for k in 1..=15 {
            let error: f64 = format!("1e-{k}").parse().unwrap_or(f64::NAN);
            assert_eq!(decimals_for_error(error), k - 1, "error = 1e-{k}");
        }
    }

    #[test]
    fn decimals_are_capped_and_degenerate_errors_keep_none() {
        assert_eq!(decimals_for_error(1e-30), 15);
        assert_eq!(decimals_for_error(0.0), 15);
        assert_eq!(decimals_for_error(5.0), 0);
        assert_eq!(decimals_for_error(f64::NAN), 0);
        assert_eq!(decimals_for_error(f64::INFINITY), 0);
    }

    #[test]
    fn truncation_cuts_instead_of_rounding() {
        let pi = 3.141592654451;
        assert_eq!(truncate_to_error(pi, 1e-7), "3.141592");
        assert_eq!(truncate_to_error(pi, 8.907e-10), "3.141592654");
        assert_eq!(truncate_to_error(pi, 1e-2), "3.1");
        // Rounding would give 3.15 / 4.
        assert_eq!(truncate_to_error(3.1999999, 1e-3), "3.19");
        assert_eq!(truncate_to_error(3.9999999, 1.0), "3");
    }

    #[test]
    fn truncation_handles_an_exact_value_and_a_huge_error() {
        assert_eq!(truncate_to_error(3.5, 1e-3), "3.50");
        assert_eq!(truncate_to_error(3.5, 0.0), "3.500000000000000");
        assert_eq!(truncate_to_error(3.5, 7.0), "3");
    }

    #[test]
    fn stratified_estimate_is_within_five_sigma_and_far_tighter_than_plain() -> Result<()> {
        let g = Grid::new(nz32(1 << 10))?;
        let (hits, d) = run_on_cpu(0xC0FFEE, g);
        let n = g.total_samples();
        let err = (estimate_pi(hits, n) - std::f64::consts::PI).abs();
        let sigma = stratified_std_error(d, n);
        assert!(err < 5.0 * sigma, "err = {err}, σ̂ = {sigma}");
        // Expected gain is about 26× at this size.
        assert!(sigma < pi_std_error(n) / 10.0, "σ̂ = {sigma}");
        Ok(())
    }

    #[test]
    fn disagreements_are_an_unbiased_variance_estimate() {
        let g = grid(1 << 7);
        // SplitMix64 outputs for adjacent counters are unrelated, but adjacent
        // *seeds* would reuse the same draws one slot over, so spread them out.
        let runs: Vec<(f64, f64)> = (0..64)
            .map(|i| {
                let (hits, d) = run_on_cpu(splitmix64(i), g);
                (hits as f64, d as f64)
            })
            .collect();
        let n = runs.len() as f64;
        let mean_hits = runs.iter().map(|r| r.0).sum::<f64>() / n;
        let var_hits = runs.iter().map(|r| (r.0 - mean_hits).powi(2)).sum::<f64>() / (n - 1.0);
        let mean_d = runs.iter().map(|r| r.1).sum::<f64>() / n;
        let ratio = var_hits / mean_d;
        assert!((0.5..2.0).contains(&ratio), "Var(hits) / mean(D) = {ratio}");
    }

    #[test]
    fn config_bounds_follow_the_log2_range() -> Result<()> {
        let min = Grid::new(nz32(MIN_CELLS_PER_SIDE as u32))?;
        let max = Grid::new(nz32(MAX_CELLS_PER_SIDE as u32))?;
        assert_eq!(MIN_CELLS_PER_SIDE, 362);
        assert!(RunConfig::new(min).is_ok());
        assert!(RunConfig::new(max).is_ok());
        assert!(err_text(RunConfig::new(grid(361))).contains("outside the supported range"));
        assert!(err_text(RunConfig::new(grid(MAX_CELLS_PER_SIDE as u32 + 1))).contains("outside"));
        assert!(err_text(RunConfig::new(grid(1 << 31))).contains("outside"));

        // The bounds are exactly what the extreme budgets produce.
        assert_eq!(RunConfig::for_samples(1 << MIN_LOG2_SAMPLES)?.grid(), min);
        assert_eq!(RunConfig::for_samples(1 << MAX_LOG2_SAMPLES)?.grid(), max);
        Ok(())
    }

    #[test]
    fn config_total_blocks_rounds_up() -> Result<()> {
        // 362² = 131044 pairs fit in one block of 2^17 pairs; 363² = 131769 do not.
        assert_eq!(RunConfig::new(grid(362))?.total_blocks(), 1);
        assert_eq!(RunConfig::new(grid(363))?.total_blocks(), 2);
        assert_eq!(
            RunConfig::new(grid(362))?.total_samples().get(),
            2 * 362 * 362
        );
        // The floor root drops a little under 2^k samples, so these sit just
        // below 2^18 and 2^22 blocks.
        assert_eq!(
            RunConfig::for_samples(1 << 36)?.total_blocks(),
            (1 << 18) - 2
        );
        assert_eq!(
            RunConfig::for_samples(1 << 40)?.total_blocks(),
            (1 << 22) - 2
        );
        Ok(())
    }

    #[test]
    fn chunks_tile_the_blocks_exactly() -> Result<()> {
        for cfg in [
            RunConfig::new(grid(362))?,
            RunConfig::for_samples(1 << 36)?,
            // 185364² pairs need 262145 blocks: one full chunk plus 1 block.
            RunConfig::new(grid(185_364))?,
            RunConfig::for_samples(1 << 41)?,
        ] {
            let chunks: Vec<Chunk> = cfg.chunks().collect();
            let mut next = 0;
            for c in &chunks {
                assert_eq!(c.first_block, next);
                assert!(c.blocks <= MAX_CHUNK_BLOCKS);
                next += c.blocks.get() as u64;
            }
            assert_eq!(next, cfg.total_blocks());
        }
        let tail: Vec<u32> = RunConfig::new(grid(185_364))?
            .chunks()
            .map(|c| c.blocks.get())
            .collect();
        assert_eq!(tail, [1 << 18, 1]);
        Ok(())
    }

    #[test]
    fn time_budget_leaves_headroom() {
        let n = samples_for_time(4e11, Duration::from_secs(12));
        assert_eq!(n, (4e11 * 12.0 * 0.95) as u64);
        assert_eq!(samples_for_time(0.0, Duration::from_secs(12)), 0);
    }

    #[test]
    fn budget_parses_seconds_and_log2_samples() -> Result<()> {
        assert_eq!(
            "15s".parse::<Budget>()?,
            Budget::Time(Duration::from_secs(15))
        );
        assert_eq!(
            "0.5s".parse::<Budget>()?,
            Budget::Time(Duration::from_millis(500))
        );
        assert_eq!(
            "36".parse::<Budget>()?,
            Budget::Samples(RunConfig::new(grid(185_363))?)
        );
        assert!("18".parse::<Budget>().is_ok());
        assert!("50".parse::<Budget>().is_ok());
        Ok(())
    }

    #[test]
    fn budget_rejects_bad_input() {
        for bad in [
            "0s", "-1s", "-0s", "abc", "", "s", "infs", "nans", "1e400s", "12 s", "-5",
        ] {
            assert!(bad.parse::<Budget>().is_err(), "{bad:?} was accepted");
        }
        assert!(err_text("0s".parse::<Budget>()).contains("positive"));
        assert!(err_text("17".parse::<Budget>()).contains("outside 18..=50"));
        assert!(err_text("51".parse::<Budget>()).contains("outside 18..=50"));
        assert!(err_text("abc".parse::<Budget>()).contains("\"abc\""));
    }
}
