# Community benchmark on AMD Radeon

Measured on 2026-11-10 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon, 12 GiB VRAM
- CPU: AMD Ryzen 5 3600 6-Core Processor (6 cores, 10 threads); installed RAM: 56.0 GiB; storage: not measured
- OS: Linux 7.0.14-26-pve; GPU driver: not read; CUDA: not read
- PolyStrata: project 0.1.45, commit 83a7a332; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: gemma-4-26b-a4b-ud-q4_k_m
- Context: 65536 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata-poly --model gemma-4-26B-A4B-it-UD-Q4_K_M.gguf --draft-model mtp-gemma-4-26B-A4B-it-Q8_0.gguf --kv-whole --max-context 65536 --expert-profile gemma4-ud-q4_k_m-official.txt --conversation-cache-mib 0 --threads 8 --vram-reserve-mib 1024
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
| short chat | 48 (45-53) | 0 | 100 (100-100) | 9 | not measured | 72.5 (60.7-92.1) | 0.65 (0.61-0.67) |
| 4K prompt, fresh | 4101 (4101-4101) | 0 (0-0) | 24 cap | 3 | 977.9 (960.4-984.1) | not measured | 4.33 (4.32-4.36) |
| 16K prompt, fresh | 16390 (16390-16390) | 0 (0-0) | 24 cap | 3 | 984.5 (974.9-984.7) | not measured | 16.85 (16.83-17.09) |
| 32K prompt, fresh | 32774 (32774-32774) | 0 (0-0) | 24 cap | 3 | 819.0 (815.0-819.4) | not measured | 40.44 (40.23-40.71) |
| agent session, follow-up turns | 15278 (15266-15287) at the last turn | see below | up to 64 | 3 | see below | 67.8 (58.6-80.5) | 4.45 (3.21-5.66) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 67.9 % (67.9-67.9); earlier-turn tokens read again per session: 259 (247-268) (earlier turns that the cache did not hold; the 14658 (14658-14658) new tokens of the files are read once); first turn (nothing cached) first token: 1.12 s (1.07-1.20).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 396 | 0 | 396 | 390.9 | 1.204 | 77.1 |
| 1 | 4014 | 389 | 3625 | 877.6 | 4.166 | 68.3 |
| 2 | 5722 | 4007 | 1715 | 561.2 | 3.211 | 68.0 |
| 3 | 9217 | 5715 | 3502 | 718.0 | 5.041 | 58.6 |
| 4 | 12223 | 9210 | 3013 | 676.4 | 4.629 | 67.8 |
| 5 | 15266 | 12216 | 3050 | 579.6 | 5.422 | 64.0 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 302 s.
