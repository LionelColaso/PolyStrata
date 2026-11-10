# Community benchmark on AMD Radeon

Measured on 2026-11-10 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD Radeon, 12 GiB VRAM
- CPU: AMD Ryzen 5 3600 6-Core Processor (6 cores, 10 threads); installed RAM: 56.0 GiB; storage: not measured
- OS: Linux 7.0.14-26-pve; GPU driver: not read; CUDA: not read
- PolyStrata: project 0.1.45, commit 83a7a332; engine 0.1.45
- Background workloads and power limits: not recorded

## Model and configuration

- Model and quantization: tiel-coder-35b-a3b-ud-q3_k_xl
- Context: 65536 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

```text
strata-poly --model Tiel-Coder-35B-A3B-MTP-UD-Q3_K_XL.gguf --kv-whole --max-context 65536 --expert-profile tiel-ud-q3_k_xl-official.txt --conversation-cache-mib 0 --threads 8 --vram-reserve-mib 1024
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
| short chat | 224 (221-229) | 0 | 100 (100-100) | 9 | not measured | 82.2 (68.1-93.6) | 0.66 (0.63-0.69) |
| 4K prompt, fresh | 4025 (4025-4025) | 0 (0-0) | 24 cap | 3 | 1192.8 (1168.0-1196.9) | not measured | 3.40 (3.38-3.46) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 1089.7 (1076.1-1092.2) | not measured | 14.78 (14.72-14.96) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 847.5 (846.2-851.1) | not measured | 37.88 (37.69-37.94) |
| agent session, follow-up turns | 15210 (15206-15272) at the last turn | see below | up to 64 | 3 | see below | 75.2 (64.6-100.2) | 3.86 (2.06-4.76) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.8 % (69.8-69.9); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 14203 (14203-14204) new tokens of the files are read once); first turn (nothing cached) first token: 1.25 s (1.23-1.28).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 835 | 0 | 835 | 671.5 | 1.281 | 71.9 |
| 1 | 4272 | 844 | 3428 | 892.8 | 3.852 | 73.5 |
| 2 | 5911 | 4314 | 1597 | 765.1 | 2.111 | 67.3 |
| 3 | 9324 | 5952 | 3372 | 736.1 | 4.605 | 72.9 |
| 4 | 12281 | 9359 | 2922 | 761.4 | 3.863 | 72.8 |
| 5 | 15206 | 12317 | 2889 | 707.0 | 4.11 | 64.6 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load and power limits are not recorded. Total time of the suite: 271 s.
