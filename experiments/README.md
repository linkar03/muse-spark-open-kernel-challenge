# Experiment registry

Every experiment in this project was written down before it was measured. [registry_it.md](registry_it.md) is the index of all 128 cards as the project's own tool printed it. It is the working file, so titles are in Italian.

## What a card holds

| field | meaning |
|---|---|
| hypothesis | what is expected to happen and why |
| predictions | one or more numbers, each with a minimum, a maximum and the model it came from |
| falsifier | the outcome that would prove the hypothesis wrong, fixed in advance |
| bench | where it is measured: the judge, a rented GPU, the local GPU |
| outcomes | the measured values, added afterwards; the predictions are not edited |
| verdict and decision | confirmed, falsified or inconclusive, and what follows |

## Counts

- **128 cards:** 73 serving, 41 MoE kernel, 13 rest-of-layer, 1 bench calibration.
- **88 closed:** 48 confirmed, 34 falsified, 6 inconclusive, not measurable or mixed.
- **40 never closed:** planned or still open when the track ended.
- **556 numeric predictions** on closed cards were compared with a measurement; **381 (69 %)** fell inside their stated interval. The index shows 12 more predicted and measured pairs on cards that were never closed.

## One card, translated

**SRV-034, scored run with KV pool ratio 0.3 (flag only).**

- **Hypothesis.** On the judge, the already verified patch launched with the one extra flag `--swa-full-tokens-ratio 0.3` keeps all 32 conversations resident: full-layer capacity rises to about 3.13 million tokens, evictions and reloads at 32 users disappear, the ratio at 32 users rises above 2.18 and the score above 2.1586, with no change at the other loads or in GPU memory.
- **Falsifier.** The reported KV capacity does not change (the flag is inert); or the ratio at 32 users stays at 2.18 or below; or later-turn time to first token at 32 users stays at 1.69 s or above; or the server does not start, or agreement, memory or time-to-first-token limits fail at any load.
- **Predicted and measured.** Score 2.23 → 2.2386 as written on the card (the judge's record reads 2.239). Ratio at 32 users 2.31 → 2.3174. KV capacity 3,127,000 → 3,117,563 tokens.
- **Verdict.** Confirmed: all seven predictions inside their intervals, no falsifier fired.

## Reading the index

Columns: id, state, component, type, title, predicted → measured (one pair per prediction, with the error in percent), verdict.

| Italian | English |
|---|---|
| chiusa / in_corso / pianificata | closed / open / planned |
| confermata / falsificata / inconcludente | confirmed / falsified / inconclusive |
| misura / taratura | measurement / calibration |
| corsa a punteggio / check | scored run / check run |
| giudice / banco | judge / bench |
| bozzatore | draft model |
| prefill diviso | split (sequence-parallel) prefill |
| scambi fusi | fused collectives |
| cache gerarchica | hierarchical KV cache |

A row without a measured value shows only the prediction and its interval in square brackets.
