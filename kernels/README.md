# Kernel track: what was measured, what failed

Five single-GPU problems, each a piece of the Muse Spark Open forward pass, timed on an H200 against SGLang's own kernel. Submissions ran from Oct 22 to Nov 4, 2026, Pacific time (the track closed at 07:59 UTC on Nov 5). This project recorded **623 judged runs** between Oct 25 and Nov 4 (419 checks, 109 scored runs, 95 whose kind was not recorded).

## Where it ended

| problem | this project's own kernels, best | public base, first run | board entry | final rank | #1 | #10 | #50 |
|---|---|---|---|---|---|---|---|
| NVFP4 MoE (`muse_moe_w4a16`) | 2.130× | 2.365× | 2.411× | 164 of 1,423 | 2.844× | 2.679× | 2.527× |
| Decode layer | 2.010× | 2.267× | 2.273× | 172 of 1,297 | 2.614× | 2.499× | 2.372× |
| Attention decode | 2.299× | 2.640× | 2.609× | 324 of 1,380 | 3.339× | 3.007× | 2.809× |
| BF16 projections | 1.526× | 1.581× | 1.570× | 218 of 1,741 | 1.728× | 1.670× | 1.605× |
| Prefill layer | 1.757× | 1.903× | 1.797× | 262 of 896 | 2.147× | 2.033× | 1.938× |

Speedups are over SGLang's kernel, as the judge reports them. Ranks were read from the final public boards on 2026-11-26.

**How to read the columns.** Until the evening of Nov 2 the project ran its own kernel lines (494 judged runs). At 22:07 UTC that day it rebased every problem onto the public Apache-2.0 code of team keell0renz (commit `890809c`), which itself builds on vjeux, alygg77 and others, because that code was 4 to 15% faster on every problem. The board entries are that base plus this project's changes. They carry the original attribution and are not redistributed here.

**Most of each board score belongs to the public code.** Measured against the first run on that base, this project's changes are worth +1.9% on the MoE and +0.3% on the decode layer, and they cost 1.2%, 0.7% and 5.6% on attention, BF16 projections and the prefill layer. This project's own contribution to the kernel track is the left column, the experiments below and the measurements.

On three problems the board entry is below the first run on the public base. That is not noise: see [Rules read too strictly](#rules-read-too-strictly).

## Progress by day

Best judged speedup at the end of each day, over every recorded run.

| day (UTC) | MoE | decode layer | attention | BF16 proj | prefill layer |
|---|---|---|---|---|---|
| Oct 25 to 27 | | | | 1.245 → 1.292 | |
| Oct 29 | | | 1.585 | 1.396 | |
| Oct 30 | 1.547 | 1.640 | 2.010 | 1.486 | 1.702 |
| Oct 31 | 1.882 | 1.884 | 2.255 | 1.509 | 1.740 |
| Nov 1 | 2.096 | 1.957 | 2.299 | 1.513 | 1.740 |
| Nov 2, own kernels | 2.130 | 2.010 | 2.299 | 1.526 | 1.757 |
| Nov 2, public base | 2.393 | 2.267 | 2.674 | 1.581 | 1.903 |
| Nov 3 | 2.431 | 2.273 | 2.687 | 1.581 | 1.903 |

## The limit model

Each component had a lower bound derived from the hardware (weight bytes that must cross the memory bus, tensor-core cycles, launch costs) and a measured best time. On Nov 2 the project's kernels reached this share of their own floor (time at the floor divided by measured time): decode layer 52%, MoE 48%, attention 34%, BF16 projections 75%, prefill layer 55%. The top decode entry on the board stood at about 64% by the same model.

## What did not work

**MoE (4-bit expert GEMMs)**

- **Prefetching weights into L2:** 3 to 13% slower.
- **Three designs taken from the public base's notes** (fused slice, a second warpgroup-specialized layout, a publisher variant): 5 to 20% slower.
- **Per-chunk weight copies:** neutral within ±2%. The bound was memory bandwidth per SM; the variants were never re-measured with that bound removed.
- **Register rebalancing between producer and consumers:** about +1% per shape, and already at the ceiling for that thread geometry.
- **Four redesigns on the last day:** a 384-thread skinny path lost on all three shapes (with two warps per SM sub-partition the tensor-core latency is no longer hidden); a dual-tile stage was closed on a modelled +0.31% without a check; a ping-pong weight ring lost at its first check; a scheduler change was closed after one check.
- **Never tried:** a different warpgroup geometry. Every variant treated the base's 640 threads and four consumer warpgroups as fixed.

**Decode layer**

- **One persistent kernel for the whole layer:** closed. Handing work between SMs costs 1.4 µs with memory idle and 6.3 µs under load.
- **A cluster-scope boundary at 32 tokens:** 1.34 to 2.02 µs slower. **Routing in two parts:** 0.61 to 1.25 µs slower.
- **One variant won inside the layer** (0.38 to 1.15 µs beyond the A/A spread) and was still excluded, because the falsifier written before the run looked at the MoE alone, where it measured 1.0008. It was left open.

**Attention decode**

- **Split-KV with a cluster-scope combine through distributed shared memory:** +1.60 µs, worse than combining through L2.

**BF16 projections**

- **Compressing BF16 weights below 12 bits:** an 11.3-bit codec is worth at most about 2.5% in bytes, and variable-length decoding costs more than that.

**Prefill layer**

- **Mixing a polynomial `exp` with the hardware one.** In a microbenchmark a 3:1 mix raised throughput by 8.5% (16.7 against 15.4 elements per clock per SM). Inside the attention kernel, moving 25% of the exponentials to the polynomial made it 13 to 15% slower.
- **An own FP8 attention kernel:** on par with FlashAttention-3, 1.3 to 2.5% behind depending on shape.

## Method errors

These cost more than any single failed idea.

- **The judge was the development bench.** Twenty checks a day per problem, with about 1.5% noise between nodes, which is larger than many of the changes being tested.
- **Until Nov 1 the diagnostic builds flushed L2 by writing instead of reading.** On small kernels that inflated differences by up to a factor of two.
- **Instrumented builds used 149 registers against 128** and distorted the timelines they were meant to explain.
- **Some decisions rested on single profiles (±35%), truncated logs or unpaired A/B runs.**
- **Lines of work were closed after a fixed number of checks**, not when the model said the ceiling was reached. Redesigns were stopped before they matured.
- **During the track the rented H200 was never calibrated against the judge.** The sign of a change always transferred; its size ranged from 0.5 to 1.3 times the judge's. A calibration attempted after the track failed, with timings off by -4.5% to +8.5%.

## Rules read too strictly

On Nov 3 the organizers clarified that approximate math accurate to BF16 precision, FP16 accumulation of attention scores, and BF16 or FP16 partial sums were all allowed. This project's internal rules forbade every one of them, some by its own decision of Nov 2, and the update was not read before the track closed on Nov 5.

- **BF16 projections:** removing the public base's BF16 split-K reductions cost 3.6% of the score at once (1.581 → 1.524; 35% on one cell, 25% on another). Later work recovered most of it: the board entry is 1.570.
- **Attention decode:** replacing the base's FP16 score kernels with FP32 paths cost 6.9% on one cell. The structural consequence was never tried either: FP16 accumulators halve the accumulator registers, which was exactly the resource that tied the geometries.
- **Prefill layer:** the base's FP16 attention path was replaced with FP32 accumulators. The unchanged base measured 1.903 in its first check; the board entry scored 1.797.
- **Other constraints were self-imposed, not rules:** fixed summation order and bit-identity until Nov 1; identical machine code and zero register spills used as design constraints rather than checks.

The lesson carried into the serving track: every working day opened by reading the change log at the top of the rules.

## After the track: the MoE kernel inside SGLang

For the serving track the MoE kernel was rebuilt inside SGLang's tree, with stock's rounding (see [serving/README.md](../serving/README.md)). Measured against the fastest public kernel on the same data:

- **On the kernel board's own metric it scores 2.14**, against 2.41 for the board entry, 2.84 for the leader and 2.93 for the floor model.
- **It takes 1.81, 1.34, 1.22, 1.20, 1.19 and 1.17 times as long** as the public kernel at 1, 8, 16, 32, 128 and 256 tokens.
- **The weight stream and the decode are equal.** The difference is the consumer: theirs costs as much as decoding alone (7.17 µs per work item at 8 columns), this one 1.9 µs more (9.07). Their kernel runs at the pace of the stream (8.51 µs per item); this one at 9.9.
- **Three short routes to close the gap failed:** fixed launch costs (11 to 21% slower), per-warp multiplies in place of narrow `wgmma` (about 1% reachable), and a balanced work queue (slower, 2.075 against 2.14). Closing it needs a rewrite of the consumer's row layout, which was not done.

The MoE work alone accounts for 41 experiment cards: 20 confirmed, 19 falsified, 2 never run.

## Data

- [results/judged_runs.csv](results/judged_runs.csv): all 623 recorded runs with time, problem, internal label, kind, job id, speedup and its confidence interval. The record stops at 16:05 UTC on Nov 4, so the last hours of the track, including the MoE board entry, are not in it.
- [results/board_entries_per_shape.csv](results/board_entries_per_shape.csv): per-shape timings (SGLang's kernel and the board entry, in microseconds) for the four board entries the database covers.
