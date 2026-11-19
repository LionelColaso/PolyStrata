# How to Use PolyStrata

A comprehensive guide to installing, configuring, and using PolyStrata — run large AI models on your own PC.

---

## Table of Contents

1. [What is PolyStrata?](#what-is-polystrata)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Setup Options](#setup-options)
5. [Running Models](#running-models)
6. [API Server & Web App](#api-server--web-app)
7. [Connecting Your Apps](#connecting-your-apps)
8. [Model Selection Guide](#model-selection-guide)
9. [Configuration & Tuning](#configuration--tuning)
10. [Multi-GPU & Advanced Setup](#multi-gpu--advanced-setup)
11. [Troubleshooting](#troubleshooting)
12. [MCP Server](#mcp-server)
13. [Updating & Maintenance](#updating--maintenance)

---

## Engine Comparison: Strata vs Strata-Poly

PolyStrata has **two completely separate engines** with different architectures, model formats, and feature sets. **This is the most important distinction to understand.**

| Aspect | 🟦 Strata Engine | 🟪 Strata-Poly Engine |
| -------- | ------------------ | ---------------------- |
| **Executable** | `strata.exe` / `strata` | `strata-poly.exe` / `strata-poly` |
| **Model Format** | Native packed experts (`.bin` packs) | Single GGUF file (read in place) |
| **Model Families** | `qwen`, `swift`, `coder`, `unsloth`, `nvfp4`, `qwen36-strata` | `glm`, `deepseek`, `lfm25`, `flint`, `kolibri`, `clover`, `qwen36`, `ornith`, `gemma4`, `tiel`, `minicpmv`, `ornith122` |
| **Architecture** | Custom MoE (Qwen3.8-Flash-Next variants) | Multiple: `qwen4exp`, `glm5-next`, `qwen35moe`, `gemma4`, `deepseek4`, `lfm2moe`, `flint`, `kolibri1`, `clover_moe` |
| **Context Window** | 8K–262K tokens | 8K–131K tokens |
| **RoPE Scaling** | ✅ `none`, `linear`, `yarn` (extend past 262K) | ❌ Fixed contexts only |
| **KV Streaming** | ✅ Setup: `--kv-streaming on`; engine: `--kv-resident 32768` | ✅ Opt-in for `qwen35moe` only; setup: `--kv-streaming on`, engine: `--kv-resident 32768` |
| **Low-RAM Mode** | ✅ `--low-ram` (mmap experts from file) | ❌ Not implemented |
| **Resident Budget** | ✅ `--resident-budget-gib` (Unsloth models) | ❌ Not available |
| **Calibration** | ✅ `--calibrate` (auto-tune engine) | ❌ Not available |
| **Draft Vocab** | ✅ `--draft-vocab` (cjk/en/cyrillic/fr) | ❌ Not applicable |
| **Speed Projection** | ✅ `--experimental-speed-projection` | ❌ Not available |
| **Remote Expert Opt** | ✅ `--no-remote-expert-opt` | ❌ Not applicable |
| **Vision Support** | ✅ All families | ⚠️ Only 6 families (see Vision table) |
| **Multi-GPU** | Layer split + remote expert opt | Layer split only (no remote opt) |
| **Linux/Windows** | Both | Both (GLM/DeepSeek: Linux only) |
| **Build System** | Prebuilt CUDA 13 / compile from source | Compile from source (no prebuilt for all) |

### Quick Decision Guide

| If you want... | Use Engine | Example Families |
| ---------------- | ------------ | ------------------ |
| Qwen3.8-Flash-Next (best quality/speed) | 🟦 Strata | `qwen`, `swift`, `coder`, `unsloth` |
| NVFP4 quantization | 🟦 Strata | `nvfp4` |
| Longest context (262K) | 🟦 Strata | All Strata families |
| KV streaming (RAM savings at 64K+) | 🟦 Strata | All Strata families |
| KV streaming on a supported GGUF | 🟪 Poly | `qwen36`, `ornith`, `ornith122`, `tiel`; opt-in, context >= 64K |
| Low-RAM mode (big GPU, little RAM) | 🟦 Strata | All Strata families |
| GLM 5.3 Flash / DeepSeek V4 Flash | 🟪 Poly | `glm`, `deepseek` |
| Qwen3.6, Ornith, Gemma 4 | 🟪 Poly | `qwen36`, `ornith`, `gemma4` |
| Single GGUF file, no packing | 🟪 Poly | All Poly families |
| AMD GPU (ROCm/HIP) | Both | Both support `hip` backend |
| Single model file you already have | 🟪 Poly | `--gguf-dir` / `--custom-gguf` |

### Correction note: [`docs/BACKENDS.md`](docs/BACKENDS.md)

`docs/BACKENDS.md` is the broad reference for model families and backends, but its current Strata-Poly KV-streaming paragraph (the one beginning “It also takes `--kv q8_0`” and saying “strata-poly also supports `--kv-resident N`”) is too broad. Read it with these corrections:

- **Incorrect or misleading as written:** it can be read as if every Strata-Poly family, including Kolibri 1, supports `--kv-resident`.
- **Correct engine scope in this build:** `--kv-resident N` is accepted only for the `qwen35moe` architecture. The engine rejects it for Kolibri 1, Flint, LFM2.5, Gemma 4, GLM and DeepSeek.
- **Correct setup scope:** `--kv-streaming on` is wired for the `qwen36`, `ornith`, `ornith122` and `tiel` setup families. It is opt-in, requires a context of at least 65,536, and setup keeps it off in WSL. Custom GGUFs that setup resolves as `qwen36` use this path.
- **Correct mechanics:** the newest `N` rows stay in VRAM; the context's rows are also kept in pinned host RAM, and older rows are streamed back through VRAM in stages. `--kv-whole` takes precedence over `--kv-resident`, and parallel engine batch slots take precedence too.
- **Run-config correction:** `--kv-streaming` is a setup option, not an engine argument. A config stores the engine's `--kv-resident N` flag.

The checked-in measurements and their limits are in [`docs/BENCHMARK_UNIFIED_REPORT.md`](docs/BENCHMARK_UNIFIED_REPORT.md). The implementation checks are in `setup.py` (setup-family scope), `src/poly/engine.cpp` (accepted model architecture) and `src/poly/main.cpp` (option precedence).

---

## What is PolyStrata?

PolyStrata runs large AI models (that usually need server-grade hardware) on a normal PC with a single NVIDIA or AMD graphics card plus system RAM. It serves an **OpenAI- and Anthropic-compatible API** on `http://127.0.0.1:8080` and includes a web app with chat, live monitor, and settings.

### Supported Models (Two Engines)

**Engine 1: `strata` (Qwen family)**

- **Qwen3.8-Flash-Next** (original, Coder, Swift 1.5, Unsloth variants)
- Reads images (vision)
- Windows & Linux, NVIDIA & AMD

**Engine 2: `strata-poly` (GGUF models)**

- **GLM 5.3 Flash** (largest: 79-114 GB)
- **DeepSeek V4 Flash** (91 GB)
- **LFM2.5-24B-A2B** (11-25 GB)
- **Flint 40B-A2.5B** (18-42 GB)
- **Qwen3.6-35B-A3B** (13-23 GB, reads images)
- **Clover-35B-A3B-Preview** (13-21 GB)
- **Ornith 1.5** 35B & 122B (14-132 GB, reads images)
- **Gemma 4 26B-A4B** (10-17 GB)
- **Tiel Coder 35B-A3B** (13-23 GB)
- **MiniCPM-V 4.7 35B-A3B** (15-28 GB, reads images)
- **Kolibri 1** (28-47 GB)
- Linux & Windows (except GLM & DeepSeek: Linux only)

---

## System Requirements

### For Qwen3.8-Flash-Next (Main Model)

| Component | Requirement |
| ----------- | ------------- |
| **GPU** | NVIDIA RTX 20/30/40/50 series (12 GB+ VRAM; 8 GB works slowly) • AMD Radeon RX 7900 XT/XTX, RX 7800 XT/7700 XT, RX 9060 XT, RX 9070/9070 XT, AI PRO R9700, RX 6800/6900 series (12 GB+ VRAM) |
| **RAM** | 64 GB recommended (see [Model Selection](#model-selection-guide)) |
| **CPU** | x86-64 with AVX2 (last ~8 years); AVX-512 faster |
| **Disk** | 70-80 GB free for model + 6 GB MTP layer (+1 GB with images) |
| **OS** | Windows 10/11, Linux (Ubuntu 22.04/24.04, Debian 12/13) |
| **Driver** | NVIDIA 580+ • AMD: Linux kernel amdgpu / Windows Adrenalin |

> **GLM 5.3 Flash** needs Linux, one NVIDIA card, and far more RAM (see [GLM.md](docs/GLM.md)).  
> **Qwen3.6, Ornith, Gemma 4** need less RAM (see [BACKENDS.md](docs/BACKENDS.md)).
---

## Installation

### Quick Start (Windows)

1. **Download** the project: [GitHub ZIP](https://github.com/VecSzn/PolyStrata/archive/refs/heads/main.zip) or `git clone https://github.com/VecSzn/PolyStrata`
2. **Unzip** (or clone) to a folder
3. **Double-click** `START-HERE.bat`

### Quick Start (Linux)

```bash
git clone https://github.com/VecSzn/PolyStrata
cd PolyStrata
./setup.sh
```

### First Run

The first run asks **4 questions** (press Enter for recommended defaults):

1. **Which model?** Qwen3.8-Flash-Next (original) or Swift 1.5 (thinks shorter)
2. **Which size?** Q2_0, IQ2_XS, or IQ3_XXS (recommended for your RAM)
3. **How much context?** 8K–256K tokens (recommended for your VRAM)
4. **Images?** yes / no (vision encoder)

Then it:

- Installs Python 3.12 (user account, no admin)
- Creates `.venv/` virtual environment
- Downloads ready-made engine (or compiles if needed)
- Downloads model (~66-76 GB, resumable)
- Prepares model + fetches MTP draft layer (~5 GB)
- Writes `run-<model>.bat` / `run-<model>.sh`
- **Starts the server** → opens browser at `http://127.0.0.1:8080`

### Subsequent Runs

Just double-click `START-HERE.bat` (or `./setup.sh`) — it skips straight to running the model.

---

## Setup Options

All options work with both `START-HERE.bat` (Windows) and `./setup.sh` (Linux).

**⚠️ Critical:** PolyStrata has **two completely separate engines** with different executables, argument names, and features. The setup script (`setup.py`) handles both, but **the engine arguments themselves are different**.

| Aspect | 🟦 Strata Engine (`strata.exe`) | 🟪 Strata-Poly Engine (`strata-poly.exe`) |
| -------- | ---------------------------------- | ------------------------------------------ |
| **Executable** | `strata.exe` / `strata` | `strata-poly.exe` / `strata-poly` |
| **Model Format** | Native packed experts (`.bin` packs) | Single GGUF file (read in place) |
| **Model Families** | `qwen`, `swift`, `coder`, `unsloth`, `nvfp4`, `qwen36-strata` | `glm`, `deepseek`, `lfm25`, `flint`, `kolibri`, `clover`, `qwen36`, `ornith`, `gemma4`, `tiel`, `minicpmv`, `ornith122` |
| **Context Window** | 8K–262K tokens | 8K–131K tokens |
| **RoPE Scaling** | ✅ `none`, `linear`, `yarn` | ❌ Fixed contexts only |
| **KV Streaming** | ✅ Setup: `--kv-streaming`; engine: `--kv-resident` | ✅ Opt-in for `qwen35moe` only; setup: `--kv-streaming on`, engine: `--kv-resident` |
| **Low-RAM Mode** | ✅ `--low-ram` / `--mmap-experts` | ❌ Not implemented |
| **Vision** | ✅ All families | ⚠️ Only 6 families |

### Vision / Images

| Option | Strata Engine | Strata-Poly Engine | Description |
| -------- | --------------- | ------------------- | ------------- |
| `--vision` | 🟦 All families | 🟪 Only: `qwen36`, `ornith`, `ornith122`, `gemma4`, `tiel`, `minicpmv` | `yes`, `no`, `none`, `gpu`, `cpu` (auto = gpu on 16 GB+ NVIDIA) |
| `--vision-tokens` | 🟦 All families | 🟪 Only with vision families | Max image tokens (default: 1024) |

> **Not available:** `glm`, `deepseek`, `lfm25`, `flint`, `kolibri`, `clover` (no vision support)

### Hardware & Performance

---

## Quick Reference: What Works on Each Engine

| Feature | 🟦 Strata Engine | 🟪 Strata-Poly Engine |
| --------- | ------------------ | ---------------------- |
| Context up to 262K | ✅ | ❌ (max 131K) |
| RoPE scaling (extend past trained) | ✅ | ❌ |
| KV Streaming (`--kv-resident`) | ✅ | ✅ Opt-in for `qwen35moe` only |
| Low-RAM / mmap experts | ✅ | ❌ |
| Resident budget (Unsloth) | ✅ | ❌ |
| Calibration (`--calibrate`) | ✅ | ❌ |
| Draft vocab selection | ✅ | ❌ |
| MTP draft depth (`--spec`) | ✅ | Via GGUF file |
| Remote expert optimization | ✅ | ❌ |
| Tail skip (`--tail-skip`) | ✅ | ✅ |
| Vision | All families | 6 families only |
| Multi-GPU layer split | ✅ | ✅ |
| Conversation cache | Via server | ✅ `--conversation-cache-mib` |
| Prebuilt engine | CUDA 13 (NVIDIA) | Must compile (no prebuilt) |

---

## Vision Support by Family

| Engine | Families with Vision | Notes |
| -------- | --------------------- | ------- |
| **Strata** | `qwen`, `swift`, `coder`, `unsloth`, `nvfp4`, `qwen36-strata` | All support `--vision yes` |
| **Poly** | `qwen36`, `ornith`, `ornith122`, `gemma4`, `tiel`, `minicpmv` | 6 of 12 families |
| **Poly (No Vision)** | `glm`, `deepseek`, `lfm25`, `flint`, `kolibri`, `clover` | `--vision` ignored/refused |

---

## Examples

```bash
# Strata Engine: Quick non-interactive install (Qwen3.8 IQ2_XS)
START-HERE.bat --yes

# Strata Engine: Swift 1.5 with 128K context and KV streaming
START-HERE.bat --setup --family swift --model IQ2_XS --context 131072 --kv-streaming on

# Strata Engine: Coder with vision and calibration
START-HERE.bat --setup --family coder --vision yes --calibrate

# Strata Engine: Unsloth UD-Q4_K_XL with resident budget
START-HERE.bat --setup --family unsloth --model UD-Q4_K_XL --resident-budget-gib 40

# Poly Engine: GLM 5.3 Flash (Linux only, no vision)
./setup.sh --family glm --model 3.0BIT --context 65536

# Poly Engine: Qwen3.6 with vision, 128K context, conversation cache
./setup.sh --family qwen36 --model UD-Q4_K_M --vision yes --context 131072

# Poly Engine: Qwen3.6 with opt-in KV streaming (off by default)
./setup.sh --family qwen36 --model UD-Q4_K_M --vision yes --context 131072 --kv-streaming on

# Poly Engine: Gemma 4 with draft model and conversation cache
./setup.sh --family gemma4 --model UD-Q3_K_M --vision yes --context 65536

# Poly Engine: DeepSeek V4 Flash (Linux only)
./setup.sh --family deepseek --model UD-IQ2_XXS --context 32768

# Poly Engine: Using existing GGUF file
./setup.sh --family ornith --custom-gguf /path/to/Ornith-1.5-35B-A3B-APEX-I-Compact.gguf

# Enable network access (both engines)
START-HERE.bat --host 0.0.0.0 --api-key "your-long-random-secret"

# Check PC compatibility only
START-HERE.bat --check

# Force AMD backend (both engines)
START-HERE.bat --backend hip

# Low-RAM mode (Strata engine only)
START-HERE.bat --low-ram auto
```

---

## Running Models

### After Installation

**Windows:** Double-click `run-<model>.bat` (e.g., `run-qwen38-IQ2_XS.bat`)  
**Linux:** Run `./run-<model>.sh`

Or simply run `START-HERE.bat` / `./setup.sh` again — it detects installed models and asks which to start.

### Switching Models

With multiple models installed:

- `START-HERE.bat` shows a menu to pick one
- Web app: Chat page → model dropdown → switch (no restart)
- API: Set `"switch_on_request": true` in config, then request with `"model": "model-name"`

### Stopping

Close the server window, or press `Ctrl+C` in the terminal.

---

## API Server & Web App

### Server Endpoints

| Endpoint | Description |
| ---------- | ------------- |
| `GET /health` | Health check (`{"status": "ok", "model": "...", "loaded": true}`) |
| `GET /v1/models` | List available models |
| `POST /v1/chat/completions` | OpenAI-compatible chat completions |
| `POST /v1/messages` | Anthropic-compatible messages |
| `POST /v1/responses` | OpenAI Responses API (stateless, for Codex CLI) |
| `GET /v1/status` | Detailed server status |
| `GET /` | Web app (Chat, Monitor, About) |

### Web App Tabs

1. **Chat** — Streaming answers, thinking display, code copy, images, sampling/thinking settings
2. **Monitor** — Real-time GPU/CPU/RAM/VRAM, tokens/s, context usage, recent requests
3. **About** — Settings, model info, API addresses, config editor

### Authentication

- Default: no API key required (localhost only)
- With `--api-key KEY`: add header `Authorization: Bearer KEY`
- **Never expose without `--api-key`** when using `--host 0.0.0.0`

---

## Connecting Your Apps

### OpenAI-Compatible Apps

```text
Base URL: http://127.0.0.1:8080/v1
API Key: any string (or your --api-key value)
Model: any name (e.g., "strata")
```

### Anthropic-Compatible Apps

```text
Base URL: http://127.0.0.1:8080/v1/messages
API Key: any string (or your --api-key value)
```

### Codex CLI / Responses API

```text
Base URL: http://127.0.0.1:8080/v1/responses
```

**Codex `config.toml`:**

```toml
[model_providers.polystrata]
name = "polystrata"
base_url = "http://127.0.0.1:8080/v1"
api_key = "anything"
```

### Claude Code

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8080
export ANTHROPIC_MODEL=claude-3-5-sonnet-20241022  # any known name
export ANTHROPIC_AUTH_TOKEN=your-api-key  # or any string
```

Add to model config (`strata-<model>.json`):

```json
"fit_max_tokens": true
```

### Thinking Level (All APIs)

```json
"reasoning_effort": "none" | "low" | "medium" | "high"  // default: high
```

### Parallel Requests

Enable batching in config or `--parallel N`:

```json
"parallel": 4
```

For a Strata-Poly config that uses `--kv-resident`, keep `"parallel": 1`. Parallel engine batch slots take precedence over KV residency, so the engine turns `--kv-resident` off when more than one batch slot is requested.

---

## Model Selection Guide

### Qwen3.8-Flash-Next Sizes (Main Engine)

| Your RAM | Recommended | Why |
| ---------- | ------------- | ----- |
| 32 GB | **Coder** (IQ1_M) | Fits 32 GB; best at code, weaker elsewhere |
| 48 GB | **IQ2_XS** (or Q2_0) | Larger sizes don't fit |
| 64 GB | **IQ2_XS** (recommended), IQ3_XXS, IQ3_S | All fit |
| 96 GB+ | **IQ3_S**, **UD-IQ4_XS**, **UD-Q4_K_XL** | Room for larger sizes |
| 128-192 GB | **UD-Q5_K_XL**, **UD-Q6_K_XL**, **Q8_0** | All experts in RAM |

> **Not sure? Take IQ2_XS.** It's the best balance.

### Speed Comparison (RTX 5070 12 GB, 64 GB RAM)

| Size | Short Chat | 128K Context | Prompt Reading (32K) |
| ------ | ------------ | -------------- | --------------------- |
| Q2_0 | 94 tok/s | 76 tok/s | 2,650 tok/s |
| IQ2_XS | 79 tok/s | 63 tok/s | 2,090 tok/s |
| IQ3_XXS | 62 tok/s | 49 tok/s | 1,750 tok/s |
| IQ3_S | 53 tok/s | 46 tok/s | 1,620 tok/s |
| Coder | 55 tok/s | 43 tok/s | 2,180 tok/s |

### Model Variants

| Variant | Description | Command |
| --------- | ------------- | --------- |
| **Original** | Qwen3.8-Flash-Next GSQ-RCO | `--family qwen` |
| **Coder** | 256/512 experts, code-optimized | `--family coder` |
| **Swift 1.5** | Fine-tune, thinks 63% less | `--family swift --model IQ2_XS` |
| **Unsloth** | 4-8 bit quantizations | `--family unsloth --model UD-Q5_K_XL` |

### Strata-Poly Models (GGUF Engine)

| Model | Family | Size Range | Notes |
| ------- | -------- | ------------ | ------- |
| GLM 5.3 Flash | `glm` | 79-114 GB | Linux only, huge RAM |
| DeepSeek V4 Flash | `deepseek` | 91 GB | Linux only |
| Qwen3.6-35B-A3B | `qwen36` | 13-23 GB | Reads images |
| Ornith 1.5 35B | `ornith` | 14-24 GB | Reads images |
| Ornith 1.5 122B | `ornith122` | 44-132 GB | Reads images |
| Gemma 4 26B-A4B | `gemma4` | 10-17 GB | |
| Tiel Coder | `tiel` | 13-23 GB | Code-focused |
| MiniCPM-V 4.7 | `minicpmv` | 15-28 GB | Reads images |
| Flint 40B | `flint` | 18-42 GB | Long context efficient |
| LFM2.5-24B | `lfm25` | 11-25 GB | Commercial use < $10M |
| Kolibri 1 | `kolibri` | 28-47 GB | |
| Clover 35B | `clover` | 13-21 GB | Block generation |

---

## Configuration & Tuning

### Model Config File

Each model creates `strata-<model>.json` in the PolyStrata folder. Key settings:

```json
{
  "engine": "strata",
  "model_path": "engine/strata.exe",
  "args": [
    "--model", "models/IQ2_XS",
    "--ctx-size", "32768",
    "--prefill", "auto",
    "--draft-vocab", "cjk"
  ],
  "api_key": null,
  "parallel": 1,
  "switch_on_request": false,
  "fit_max_tokens": false,
  "reasoning_effort": "high",
  "env_to_user": false,
  "return_progress": false,
  "tool_call_recovery": false
}
```

`--kv-streaming` is a setup option, not an engine argument. Setup translates it into the engine's `--kv-resident N` argument in the generated config; do not put `--kv-streaming` in the config's `args` array.

### Common Tweaks

| Setting | File Location | Effect |
| --------- | --------------- | -------- |
| `parallel` | `strata-<model>.json` | Concurrent requests (uses more VRAM) |
| `switch_on_request` | `strata-<model>.json` | Allow model switching via API |
| `fit_max_tokens` | `strata-<model>.json` | Required for Claude Code |
| `reasoning_effort` | `strata-<model>.json` / request | Thinking level |
| `args` → `--vram-reserve-mib` | `strata-<model>.json` | Reserve VRAM for other apps |
| `args` → `--prefill` | `strata-<model>.json` | Prefill chunk size (`auto`, `4096`, `8192`, etc.) |
| `args` → `--kv-resident` | `strata-<model>.json` | Engine-level KV residency; Strata or `qwen35moe` Poly only |
| setup → `--kv-streaming on` | setup command only | Opt in to KV residency; context >= 64K; Poly setup supports `qwen36`, `ornith`, `ornith122`, `tiel` |

### Strata-Poly KV Streaming (`qwen35moe` only)

- **Default:** off. A normal Poly config has no `--kv-resident` argument and keeps the attention cache in VRAM as the conversation grows.
- **Setup opt-in:** use `--kv-streaming on` with a supported family and a context of at least 65,536. Setup then writes `--kv-resident 32768` into the generated config.
- **Direct config edit:** add `"--kv-resident", "32768"` to `args`; `--kv-streaming` itself does not belong in `args`.
- **Memory behavior:** the newest 32,768 rows remain in VRAM, while the context's rows are also kept in pinned host RAM. Attention streams older rows back through VRAM in stages.
- **Overrides:** `--kv-whole` takes precedence and keeps the full context resident in VRAM. More than one parallel engine batch slot also takes precedence and disables KV residency.
- **Environment:** setup does not enable Poly KV streaming in WSL because the host copy must be pinned.
- **KV format:** `--kv q8_0` and `--kv q4_0` are available for `qwen35moe`. The local configs use `q8_0`; the checked-in PR #180 report did not measure Q4 answer quality, so Q4 needs separate quality evidence before use.

For this PC's custom GGUFs, Groot, KAT Coder, MiniCPM-V, Sage, Tiel and both Qwen3.6 files use the `qwen35moe` architecture and can use KV residency. Setup resolves those custom files through its `qwen36` family path, which is why their generated names begin with `strata-qwen36-`. Flint uses the `flint` architecture and must not receive `--kv-resident`.

**This PC's local override:** context `131072`; KV format `q8_0`; conversation cache `1280` MiB; `"parallel": 1`; images off; `--tail-skip 0` for the `qwen35moe` models; `--kv-resident 32768` for those seven models and no KV residency for Flint. The configs store `--vram-reserve-mib 0`, but the engine enforces at least 256 MiB. Use `--vram-reserve-mib 700` instead when the GTX 1650 also serves the desktop or games.

### Calibration

Run `START-HERE.bat --calibrate` to auto-tune engine settings for your PC (15-30 min).

---

## Multi-GPU & Advanced Setup

### Multiple GPUs (Same Model)

```bash
# Use GPUs 0 and 1
START-HERE.bat --gpus 0,1

# Explicit layer split (e.g., 20 layers on GPU 0, 20 on GPU 1)
START-HERE.bat --gpus 0,1 --layer-split 20,20
```

### Replicas (Independent Engines)

```bash
# 2 replicas on 4 GPUs = two 2-GPU engines
START-HERE.bat --gpus 0,1,2,3 --replicas 2
```

Each replica holds its own model copy; requests spread across replicas.

### AMD GPU (HIP Backend)

Auto-detected if no usable NVIDIA card. On Windows: ready-made HIP engine. On Linux: compiled.

```bash
# Force AMD backend
START-HERE.bat --backend hip
```

See [AMD_HIP.md](docs/AMD_HIP.md) for details.

### Intel Arc (SYCL - Experimental)

```bash
START-HERE.bat --backend sycl
```

See [INTEL_ARC.md](docs/INTEL_ARC.md).

### Low-RAM Mode (Big GPU, Little RAM)

```bash
START-HERE.bat --low-ram auto
```

Reads experts from file instead of copying to RAM. Options: `auto`, `on`, `off`, `resident`, `mmap`.

### Conversation Cache (GLM/Poly Models)

```bash
# Set conversation cache size (MiB)
START-HERE.bat --setup --family glm --conv-cache 8192
```

---

## Troubleshooting

### Common Issues

| Symptom | Fix |
| --------- | ----- |
| PC froze on first start | Normal while loading model. Wait 10 min. Restart PC, close other apps, or pick smaller size. |
| Download/install stopped | Re-run `START-HERE.bat` / `./setup.sh` — it resumes. |
| Very slow, disk blinking, "engine stopped" | Not enough free RAM. Close browser/apps, or use smaller size (`--model Q2_0` or `IQ2_XS`). |
| "Port 8080 already in use" | PolyStrata already running. Check `/health` or use `--port 8081`. |
| "NVIDIA driver too old" | Update driver (NVIDIA App or nvidia.com/drivers), restart, re-run setup. |
| "ExpertCache: cudaMalloc failed: out of memory" (Windows) | Page file off/tiny. Set fixed size (e.g., 65536 MB), restart. |
| "Prompt exceeds context" | Re-run setup with `--setup --context <larger>`. |
| No AMD GPU found (Linux) | Check amdgpu driver, card support. Integrated GPUs not supported. |
| Python/build tools failed | Install what message names, then re-run. |

### Logs

- Engine log: `strata-<model>.log` in PolyStrata folder
- Setup log: `.strata-mcp/setup.log` (if using MCP)
- Server window prints `ready: http://127.0.0.1:8080/v1` when ready

### Health Check

```bash
curl http://127.0.0.1:8080/health
# {"status": "ok", "model": "...", "max_context": ..., "loaded": true, ...}

curl http://127.0.0.1:8080/v1/models
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{"model": "strata", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 64, "reasoning_effort": "none"}'

```

### More Help

- [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

---

## MCP Server

Let your AI assistant (Claude Code, Cursor, VS Code, Codex, etc.) manage PolyStrata via tools.

### Add to Your Assistant

**Claude Code:**

```bash
claude mcp add strata -- python C:\Users\you\PolyStrata\tools\strata_mcp.py  # Windows
claude mcp add strata -- python3 /home/you/PolyStrata/tools/strata_mcp.py     # Linux
```

**Claude Desktop / Cursor / VS Code / Codex:** Add to config:

```json
{
  "mcpServers": {
    "strata": {
      "command": "python",  // or "python3" on Linux
      "args": ["C:\\Users\\you\\PolyStrata\\tools\\strata_mcp.py"]
    }
  }
}
```

### Available Tools

| Tool | Description |
| ------ | ------------- |
| `strata_status` | Check if running, VRAM usage, model info |
| `strata_models` | List installable models for this PC |
| `strata_install` | Install PolyStrata (asks for confirmation) |
| `strata_start` | Start the model |
| `strata_stop` | Stop the model (force option) |
| `strata_logs` | Get setup/engine/server logs |
| `strata_benchmark` | Run short speed test |

### Example Usage

> "Install PolyStrata for this PC."  
> "Start PolyStrata."  
> "Is PolyStrata running? How much VRAM does it use?"  
> "Which PolyStrata model fits my PC?"  
> "PolyStrata didn't start - what does the log say?"

See [MCP_SERVER.md](docs/MCP_SERVER.md) for full details.

---

## Updating & Maintenance

### Update Engine & Server

```bash
START-HERE.bat --update
# or
./setup.sh --update
```

Keeps your model files; updates engine binary and Python server.

### Rollback Engine

```bash
START-HERE.bat --rollback-engine
```

### Add Another Model

```bash
START-HERE.bat --setup --family <family> --model <size>
# or
SETUP.bat --setup --family <family> --model <size>
```

Shared files (vision encoder, shard 2) are not re-downloaded.

### Where Files Are Stored

| Location | Contents |
| ---------- | ---------- |
| `PolyStrata/` (project folder) | Code, `.venv/`, `engine/`, `third_party/`, config JSONs, run scripts |
| `PolyStrata-data/` (beside project) | Model files (`models/`, `packs/`, `mtp/`) — 70-120 GB |
| `%APPDATA%\PolyStrata\settings.json` (Win) / `~/.config/polystrata/settings.json` (Linux) | Data directory location |

### Uninstall

Delete the `PolyStrata/` folder and `PolyStrata-data/` folder.

---

## Key Files Reference

| File | Purpose |
| ------ | --------- |
| `START-HERE.bat` / `setup.sh` | One-click installer & launcher |
| `setup.py` | Core setup logic (all options) |
| `run-<model>.bat` / `run-<model>.sh` | Direct model launch scripts |
| `strata-<model>.json` | Per-model configuration |
| `serve/server.py` | API server (OpenAI/Anthropic compatible) |
| `serve/frontend.py` | Web app (Chat, Monitor, About) |
| `tools/strata_mcp.py` | MCP server for AI assistants |
| `polystrata.py` | Module entry point |

---

## Documentation Links

| Document | Description |
| ---------- | ------------- |
| [README.md](README.md) | Project overview, quick install |
| [INSTALL.md](docs/INSTALL.md) | Detailed installation guide |
| [AI_SETUP.md](docs/AI_SETUP.md) | AI-agent installation instructions |
| [MODELS.md](docs/MODELS.md) | Model selection by RAM/speed |
| [DETAILS.md](docs/DETAILS.md) | Technical details, API, all settings |
| [BACKENDS.md](docs/BACKENDS.md) | All model families & backends; see the KV-streaming correction note above |
| [GLM.md](docs/GLM.md) | GLM 5.3 Flash specifics |
| [ORNITH.md](docs/ORNITH.md) | Ornith 1.5 specifics |
| [DEEPSEEK.md](docs/DEEPSEEK.md) | DeepSeek V4 Flash specifics |
| [AMD_HIP.md](docs/AMD_HIP.md) | AMD GPU build & validation |
| [MULTI_GPU.md](docs/MULTI_GPU.md) | Multi-GPU configuration |
| [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Common problems & fixes |
| [MCP_SERVER.md](docs/MCP_SERVER.md) | MCP server setup & tools |
| [QUANTS.md](docs/QUANTS.md) | Quantization quality comparison |
| [BATCHING.md](docs/BATCHING.md) | Parallel request batching |

---

## Security Notes

- **Never expose the server beyond `127.0.0.1` without `--api-key`**
- Use `--host 0.0.0.0 --api-key "long-random-secret"` for LAN access
- No telemetry; nothing leaves your PC
- All model weights stay local

---

## License

PolyStrata is open source under the **MIT License**. Individual models have their own licenses (see [HOW_IT_WORKS.md#license](docs/HOW_IT_WORKS.md#license)).

---

*Generated from PolyStrata documentation. For the latest version, see the [GitHub repository](https://github.com/VecSzn/PolyStrata).*
