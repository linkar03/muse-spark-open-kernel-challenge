# Serving track: the patches

Each file in [patches/](patches/) is the complete diff against SGLang commit `297e54fd105e974cdf401e5b4d20253a22c16f90` that one scored run used. They are cumulative: every patch contains the ones before it. The sha256 of each file is the one the judge recorded for that job; `tools/verify_patches.sh` checks them.

| # | Patch | Job | Score | Files | Lines | Launch flags |
|---|---|---|---|---|---|---|
| 1 | `01_nvfp4_expert_kernel.diff` | `a79b1b4be2ef44fa` | 1.409 | 6 | +1,812 −8 | none |
| 2 | `02_memory_fix_split_prefill.diff` | `83300888fd6b44eb` | 1.700 | 11 | +1,999 −42 | `--enable-layernorm-sp` |
| 3 | `03_kv_writeback_spans_prefix_reuse.diff` | `ca898a4bd3f04762` | 2.159 | 13 | +2,237 −64 | `--enable-layernorm-sp` |
| 4 | same patch as 3 | `39d68d7c5e264f07` | 2.239 | 13 | | adds `--swa-full-tokens-ratio 0.3` |
| 5 | `05_fused_collectives.diff` | `7ecffc2b96af444f` | 2.316 | 13 | +2,246 −67 | as 4 |
| 6 | `06_column_split_pooler.diff` | `91bf2b180d444050` | 2.365 | 16 | +2,343 −78 | as 4 |
| 7 | `07_draft_context_projection_per_rank.diff` | `19289d345d3c4640` | **2.437** | 18 | +2,416 −80 | as 4 |
| 8 | `08_final_comment_only.diff` | `d3b554bca0f145cf` | 2.432 | 18 | +2,416 −80 | as 4 |

All paths below are under `python/sglang/`.

## 1. NVFP4 expert kernel for Hopper

**Files:** `kernels/jit/csrc/muse_spark/nvfp4_experts_sm90.cuh` (new, 1,404 lines), `kernels/ops/moe/muse_spark_nvfp4_experts_sm90.py` (new, 247 lines), and the hooks that select it per layer.

Stock SGLang runs the model's routed 4-bit experts on H100/H200 through the generic Marlin W4A16 path: two GEMMs with an activation between them. This kernel does both GEMMs and the SwiGLU in one persistent kernel per GPU.

- **Three kernels chained with programmatic dependent launch:** route positions (above 256 tokens), a gather that applies the pre-expert RMSNorm and scatters rows into per-expert tiles, and the expert kernel itself.
- **The expert kernel streams the packed weights with bulk copies**, dequantizes NVFP4 to BF16 in registers and feeds them to `wgmma` as the register operand. Accumulation is FP32.
- **Rounding follows the Marlin path stage by stage**, so the two agree except for the order of the FP32 sums. That is what keeps the judge's agreement metrics at stock's own level.
- **It falls back to Marlin** when a checkpoint's block scales are outside what the kernel decodes, or when the build fails at load.

**Provenance.** This kernel builds on public kernel-track code. What was taken, why, and what was added are in [Provenance and credits](../README.md#provenance-and-credits).

**Outcome.** Score 1.409 against a forecast of 1.64. The shortfall was a memory bug in the weight repack, fixed in the next step.

## 2. Memory fix and split prefill

**Files:** `srt/layers/muse_spark/muse_spark_nvfp4_moe.py`, `srt/layers/layernorm_sp.py`, `srt/arg_groups/layernorm_sp_hook.py`, `srt/models/muse_spark.py`, `srt/arg_groups/fields/parallel.py`, `srt/environ.py`.

- **Memory fix.** SGLang's post-load staging keeps the previous layer's native weights alive until its hook returns. Repacking a layer before returning those blocks split them and stranded about 0.2 GB per layer. The fix returns the allocator cache at the head of each layer's repack. Weights went from 82.7 back to 69.8 GB per GPU (stock: 69.85), the KV cache from 28.0 back to 37.6 GB, and its capacity from 1.23 to 2.14 million tokens.
- **Sequence-parallel prefill on Hopper.** SGLang's LayerNorm sequence parallelism splits a long prefill's rows across the GPUs, but required an all-to-all expert backend that Hopper does not have. Here the experts run on all-gathered tokens with the same call as the tensor-parallel path, then reduce-scatter their partial sums back to the shards. Prefills below a size threshold stay replicated.

**Outcome.** 1.700. First-turn time to first token at 1 user fell from 6.37 s to 5.01 s.

## 3. KV write-back by span and prompt-prefix reuse

**Files:** `kernels/jit/csrc/kvcacheio/staged_write_back.cuh` (+15 −4), `srt/entrypoints/openai/parallel_prompt_encode.py` (+221 −18), `srt/environ.py` (+2).

- **KV write-back.** With the hierarchical cache on, staged KV pages were copied back one call per page. Staging pages are packed, so pages with consecutive host indices form one contiguous span on both sides; one copy per span replaces one call per page. A check run with only this change measured 27,171 tokens/s per GPU at 32 users against 21,570 before.
- **Prompt-prefix reuse.** A multi-turn chat renders the whole conversation again on every turn. The encoder keeps the ids of recent prompts and encodes only the text after the shared prefix, cutting at added-token boundaries. Its own check showed no measurable gain on the judge (later-turn time to first token 0.163 s against 0.160 s). It is correct and harmless, and it is the part of the final patch that earned nothing measured.

**Outcome.** 2.159, the largest single step. At 32 users the write-back change accounts for it: the check with only that change (27,171 tokens/s per GPU) is within 0.4 % of the scored run (27,270), inside the ±0.66 % noise seen between runs at that load. At the other loads the two changes were not measured separately.

## 4. KV pool ratio (flag only)

`--swa-full-tokens-ratio 0.3` instead of the default 0.8. The sliding-window layers need far less KV than the full-attention layers at this context length, so the default left the full pool short: 23 of 32 conversations resident, the rest re-prefilled every turn.

**Outcome.** 2.239. KV capacity 1.98 → 3.12 million tokens; later-turn time to first token at 32 users 1.69 s → 0.57 s; GPU memory unchanged.

## 5. Fused collectives

**Files:** `srt/layers/layernorm_sp.py`, `srt/models/muse_spark.py`.

In the split prefill, the all-gather before a GEMM and the reduce-scatter after it use torch's fused symmetric-memory forms (`fused_all_gather_matmul`, `fused_matmul_reduce_scatter`), which overlap the transfer with the multiply. They start at 8 ranks on Hopper with every prefill graph bucket.

**Outcome.** 2.316 against a forecast of 2.35. First-turn time to first token at 1 user 3.87 s → 3.62 s.

## 6. Column-split pooler and unpooler

**Files:** `kernels/ops/communication/muse_spark_all_gather.py`, `kernels/ops/communication/muse_spark_all_gather_norm_residual.py`, `kernels/ops/layernorm/muse_spark_norm_residual_jit.py`, `srt/models/muse_spark.py`.

In unsharded forwards of 1 to 768 tokens (the decode and verify steps), every GPU computed the same 4096- and 8192-wide projections. They are now split by columns across the GPUs and exchanged through the SM90+ multicast push plane, removing replicated work from each decode step. The server logs one static line at startup stating whether the path is on.

**Outcome.** 2.365 against a forecast of 2.375. Raw throughput at 16 users rose 2.7 % with 1.0 % spread between repetitions; the ratio to stock at that load rose 1.0 %.

## 7. Draft context projection per rank

**Files:** `srt/speculative/dflash_worker_v2.py` (+53 −2), `srt/arg_groups/fields/spec.py` (+12), `srt/models/muse_spark.py` (+8).

Under the split prefill, each rank applies the draft model's context projection (a linear layer and its norm, row-wise) to its own rows of the captured target hidden states before the gather. The gather then moves the draft's hidden size instead of every captured layer, and the projection is not repeated on every GPU. New option `speculative_dflash_split_context_projection`, on by default.

**Outcome.** 2.437 against a forecast of 2.384. An isolated check at 32 users had shown +0.50 % throughput, inside the ±0.66 % noise, and 2.8 s less first-token wait, so the forecast was modest. The scored run came in above its interval and the second sample (2.432) agrees with it. Peak GPU memory came in 2.0 GB under stock, against +0.1 GB for run 6, although the two runs report identical memory at startup. The difference is in peak use during the run; its cause was not isolated.

## 8. Final

Run 7 with three comment lines reworded in `muse_spark_all_gather.py`. Same code, new sample: 2.432.

## Results files

- [results/scored_runs.csv](results/scored_runs.csv): one row per scored run (score, patch hash, flags, agreement metrics, memory, KV capacity).
- [results/scored_runs_per_load.csv](results/scored_runs_per_load.csv): one row per run and load (throughput, per-user speed, time to first token, accepted tokens per step, spread between the three repetitions, stock's numbers on the same node, the ratio).
- [results/receipts/](results/receipts/): the judge's text output for each scored job, unedited.
- [results/check_runs.csv](results/check_runs.csv) and [results/check_receipts/](results/check_receipts/): the 17 check runs (one load each, unscored), with what each one tested. Most of them ran experimental patches that are not in this repository; their hashes are in the table.

Columns worth a note:

- `ours_tok_s_per_gpu_at_stock_speed` is the patched server's throughput read at stock's per-user speed; `ratio_vs_stock` is that divided by `stock_tok_s_per_gpu`. These are the numbers the score is made of.
- `tok_s_per_gpu` and `tok_s_per_user` are the patched server as measured at that concurrency.
- `excluded_from_frontier` is non-empty once: the 16-user rung of run 1, for a time-to-first-token limit.
