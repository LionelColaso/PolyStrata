# Community benchmark on AMD Radeon

Measured on 2026-11-10 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon, 12 GiB VRAM
- CPU: AMD Ryzen 5 3600 6-Core Processor (6 cores, 10 threads); installed RAM: 56.0 GiB; storage: not measured
- OS: Linux 7.0.14-26-pve; GPU driver: not read; CUDA: not read
- PolyStrata: project 0.1.45, commit 83a7a332; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: qwen3.6-35b-a3b-ud-q4_k_m
- Context: 65536 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata-poly --model Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --kv-whole --max-context 65536 --expert-profile qwen36-ud-q4_k_m-official.txt --conversation-cache-mib 0 --threads 8 --vram-reserve-mib 1024 --tail-skip 0
```

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a PolyStrata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 78.9 (66.1-89.4) | 0.41 (0.39-0.47) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1037.5 (1021.0-1039.5) | not measured | 3.89 (3.89-3.95) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1050.9 (1037.5-1052.4) | not measured | 15.32 (15.27-15.51) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 833.4 (828.3-834.5) | not measured | 38.52 (38.43-38.76) |
| agent session, follow-up turns | 15005 (14993-15010) at the last turn | see below | up to 64 | 3 | see below | 80.9 (67.1-87.9) | 4.53 (3.28-5.29) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 68.6 % (68.6-68.6); earlier-turn tokens read again per session: 248 (236-253) (earlier turns that the cache did not hold; the 14183 (14183-14183) new tokens of the files are read once); first turn (nothing cached) first token: 1.35 s (1.33-1.41).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 442.8 | 1.412 | 80.2 |
| 1 | 4053 | 602 | 3451 | 818.5 | 4.229 | 74.8 |
| 2 | 5688 | 4046 | 1642 | 502.6 | 3.29 | 67.1 |
| 3 | 9112 | 5681 | 3431 | 685.8 | 5.028 | 80.1 |
| 4 | 12075 | 9105 | 2970 | 658.7 | 4.529 | 81.1 |
| 5 | 15005 | 12068 | 2937 | 558.5 | 5.28 | 74.9 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 291 s.
