# Measurement log

Notes from bringing this fork up on one machine and tuning it. **None of this
is a property of the repo** — different hardware, driver, or image digest will
give different numbers, and one of the findings below is that some numbers are
not even reproducible on the same machine.

The scripts deliberately carry no measurements. They explain *why the code does
what it does*; this file records *what happened when it ran here*.

Dates: 2026-09-05 / 06.

## Machine

| | |
|---|---|
| GPU | NVIDIA GB10, sm_121, driver 580.126.09, CUDA 13.0 |
| Memory | 119 GiB unified, 16 GiB swapfile |
| OS | Ubuntu, kernel 6.17.0-1008-nvidia, aarch64 |
| Docker | 29.1.3, NVIDIA Container Toolkit 1.19.0 |
| Image | `lmsysorg/sglang:qwen38flashnext` (arm64), torch 2.13.0+cu130 |
| Checkpoint | `RadixArk/Qwen3.8-Flash-Next-NVFP4`, 206 safetensors, 135.25 GB |

Download ran at ~11 MB/s (link-limited; 8 parallel streams gave the same
aggregate as 1), so the checkpoint took ~3.5 h.

## 1. The flash_attn patch was mounted over a path the image does not use

The most consequential finding, and it would have been silent.

`install-3.8flash.sh` mounts the sm_121 flash-attention patch over
`/usr/local/lib/python3.12/dist-packages/flash_attn/cute/flash_fwd.py`. In this
image SGLang installs into its own prefix and that directory, while it exists,
is **not on `sys.path`**. Docker creates a missing bind target rather than
erroring, so the mount "succeeds" and nothing ever imports the patched file.

Evidence for the correct path, `/opt/sglang/lib/python3.12/site-packages/...`:

* `import flash_attn.cute.flash_fwd` resolves there
* `qwen_sparse_attn_backend.py:83` imports top-level `flash_attn.cute.*`
* the patch differs from that file by **one line**, and from SGLang's vendored
  copy (`sglang/kernels/ops/attention/flash_attn/...`) by 508

The one line is the sm_121 TMA-store guard:

```python
-        self.use_tma_O = self.arch >= Arch.sm_90
+        self.use_tma_O = self.arch >= Arch.sm_90 and mCuSeqlensQ is None
```

Without it the server boots cleanly, benchmarks plausibly, and returns quietly
wrong tokens — which is what the upstream README means by "stock kernels
silently corrupt on this chip".

`setup.sh` now remaps this and verifies all four bind targets against the image
both before installing and after the launcher is generated.

## 2. Concurrency is capped by the mamba pool, not by `--max-running-requests`

With upstream's `--max-running-requests 8`, the scheduler never ran more than
**4**; the rest queued (`#queue-req` reached 8). Admission reserves
`3 (base) + 2 (overlap) = 5` mamba slots per request, and
`--max-mamba-cache-size 24 / 5 = 4`.

Constants are in `mem_cache/kv_cache_configurator.py`
(`MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO` and friends). This also explains
a bimodal benchmark pattern — 8 submitted streams split 4 fast / 4 slow at
almost exactly 2× the latency.

`SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1` drops the base ratio by 1 (→ 6 concurrent
from the same 24 slots). Not tested here.

## 3. Tuning `MAX_RUNNING` and `MAMBA_CACHE`

Mamba state is ~57 MB per slot and **fixed regardless of context length** (36
of 48 layers are Gated DeltaNet, carrying recurrent state instead of KV). Slots
sized for 8 concurrent requests are dead weight if you allow 2, and the freed
memory goes to the KV pool nearly 1:1:

```
mamba ssm  1.32 -> 0.69 GB   (24 -> 12 slots)
KV         1.02 -> 1.66 GB
```

Measured effect on the KV pool:

| config | pool |
|---|---:|
| `8 / 24` (upstream) | 55,360 |
| `2 / 24` | 89,664 |
| `2 / 12` | 97,152 – 146,816 (see §5) |

Throughput cost: **none** single-stream (40–46 tok/s across all three);
peak aggregate drops ~27%, which only matters above 2 concurrent requests.

Do not set `MAMBA_CACHE` below 12: `10/5 = 2.0` sits exactly on the boundary,
and slots also serve mamba *prefix caching* between turns (`evict_mamba()` in
`mamba_radix_cache.py`), which multi-turn work depends on.

## 4. The KV pool is the real context ceiling, and the API misreports it

`/v1/models` advertises `max_model_len = context_length`. The server actually
rejects on pool size:

```
Input length (174054 tokens) exceeds the maximum allowed length (144890 tokens).
```

144,890 ≈ `max_total_num_tokens`. So the advertised 200,000 was never servable.

**Lowering `CTX` does not fix this.** At `CTX=130000` the server advertised
130,000 and rejected above 110,714 — the gap moved down rather than closing,
because `context_len` also feeds pool sizing. Configure the *client's* context
window; leave `CTX` unset.

Long-context behaviour at a 144,896 pool (needle at 70% depth, all correct):

| prompt | % of pool | wall | prefill |
|---:|---:|---:|---:|
| 54,303 | 37% | 21.9 s | ~2,500 tok/s (cold) |
| 114,191 | 79% | 32.4 s | — |
| 138,142 | 95% | 26.3 s | faster: prefix-cache hit |

Below the hard limit the failure mode is **preemption**, not an error: SGLang
evicts and recomputes prefill, a silent 30–50 s stall. Leaving ~20% headroom
avoids it. Prefix caching (`disable_radix_cache: False`) shares the same pool,
so a session at 95% has nothing left to retain its prefix between turns.

## 5. The KV pool is not reproducible across boots

Five boots, **identical settings** (`2 / 12`, `CTX` unset):

```
97,152  101,952  117,760  144,896  146,816
```

Spread 1.51×, mean 121,715.

**Cause (found 2026-11-22): the pool is a ~2 GB remainder of a ~112 GB
measurement, and that measurement is noisy.** On an integrated GPU, SGLang's
`get_available_gpu_memory()` (`srt/utils/common.py`) skips
`torch.cuda.mem_get_info()` and returns `psutil.virtual_memory().available`,
i.e. the kernel's `MemAvailable` estimate. The pool is then sized in
`_profile_available_bytes()` (`srt/mem_cache/kv_cache_configurator.py`):

```
KV = avail_now − pre_model_load_avail × (1 − mem_fraction_static) − mm_reserve − mamba
```

Worked through for the 146,816 boot (from its startup log):

```
avail before target load   111.56 GB  → slack 0.10 × 111.56 = 11.16 GB
avail after draft load      14.60 GB
mm reserve                   0.10 GB
mamba (12 slots)             1.36 GB  (conv 0.03 + ssm 0.69 + intermediate 0.63 + 0.01)
rest                         1.98 GB  → KV allocated 1.82 GB = 146,816 tokens
```

That is ~12.4 KB per token (target + draft KV), so **each 1 GB of error in
`MemAvailable` at profile time moves the pool by ~80K tokens.** The whole
97K–147K spread is about 0.6 GB of wobble in a ~112 GB reading.

`MemAvailable` has plenty of room to wobble that much. It is an estimate that
shifts with page cache (loading streams the whole 135 GB checkpoint through
it) and with how much anonymous memory the kernel has swapped out by the time
of profiling, which varies per boot. The draft numbers above are the same
effect: the "allocation" is a `MemAvailable` delta, not a count of bytes
allocated. The draft weights are identical every boot, yet the delta varied
2.06–2.73 GB. That 0.67 GB spread alone is worth ~54K tokens, the size of the
observed gap. It correlated only loosely (r = −0.60) because it is one noise
source among several, not the cause. The earlier check of `MemAvailable` after
the *target* load (16.95 vs 16.89 GB) was taken at the wrong point: the
profile reads it after the draft load, one noisy step later.

What follows:
* No setting raises the floor. `--max-total-tokens` is a cap only (Dead
  ends), and raising `MEM_FRACTION` above 0.90 trades directly against the
  runtime reserve. At 0.90 the box idles at ~9 GB available.
* Booting from a clean state should narrow the spread but won't remove it:
  run `sync; echo 3 > /proc/sys/vm/drop_caches` and `swapoff -a && swapon -a`
  first. Untested. Loading itself still churns page cache and swap.

Practical consequence: **read the pool at startup, do not assume it.**

```
curl -s localhost:30000/get_server_info | grep -o '"max_total_num_tokens":[0-9]*'
```

A low draw is worth one restart if you need the headroom. Even the floor is
1.75× the upstream default.

## 6. Mode 2 (`SGLANG_QWEN4_PLE_OFF`)

The n-gram table is a 320M-row hashed **trigram** memory (`ngram_size 3`,
16 heads × 160 dims, fp8 = 51.2 GB), injected additively into the residual
stream at **layer 2 only**. Mode 1 already runs it lossy: HashK R=4 collapses
320,001,446 rows into 80,000,369 — a 4:1 collision rate — for 12.8 GB.

Mode 2 zeroes the injection entirely, freeing that 12.8 GB for the KV pool.
Upstream measures ~2 points on tool-eval-bench, "within run noise". That is the
benchmark closest to agentic coding, so it is the relevant one — but it is also
the *only* measurement, and the model was trained with the injection, so
switching it off is out-of-distribution. Not evaluated here.

## 7. The model is multimodal, and vision is active

Not obvious from the upstream README. `vision_config` (27-layer ViT), 333
vision tensors in the checkpoint, and `--language-only` does **not** disable it
— that flag is about disaggregated encoder serving
(`"Encoders are expected to register dynamically via the EncoderBootstrapServer"`).

Verified: solid colours (blue/green/yellow) and a two-tone spatial test
("left half red, right half blue") all answered correctly.

Cost — one token per 32×32 pixel block, `(px/32)²`, floored at 64:

| image | tokens | TTFT | decode |
|---|---:|---:|---:|
| none | 0 | 0.17 s | 43.3 tok/s |
| 448² | 196 | 0.41 s | 38.5 |
| 1344² | 1,764 | 1.8 s | 36.5 |

Prefill is where it hurts (~9× TTFT); decode costs ~15%.

## Throughput

`sgbench.sh`, temperature 0.0. Single-stream figures match the upstream README
(within its stated run-to-run noise).

| workload | 1 stream | 8 submitted (4 running) |
|---|---:|---:|
| Code | 39.8–41.4 | 84.3–85.4 |
| JSON | 44.2 | 93.0–95.1 |
| Math | 41.5 | 94.2–99.5 |
| LongCode | 43.1–43.8 | 81.5–81.9 |
| Q&A | 36.2–36.4 | *ragged, not comparable* |

Batching returns only ~2×, not 8× — MoE activates a wider union of experts as
the batch grows. Q&A is excluded because its streams end at natural lengths, so
the batch drains and the machine idles through the tail.

Speculative decoding accepts ~2.76 of 4 draft tokens. **Throughput is strongly
sampling-dependent**: at `temperature 0.7` the same workload gave ~26 tok/s
versus ~43 at greedy, because the NEXTN draft agrees with the target far more
often when sampling is greedy. Benchmark at the temperature you actually use.

## Dead ends

Recorded so they are not retried.

* **`--max-total-tokens`** cannot raise the pool. It is a cap only:
  `token_capacity = min(token_capacity, user_limit)`, and it warns and ignores
  you if you ask for more.
* **Persisting Triton/Inductor kernel caches** across boots (mounting
  `TRITON_CACHE_DIR` / `TORCHINDUCTOR_CACHE_DIR`) changed nothing: boot time
  612 s warm vs 598 s cold, and pool variance unchanged. Reverted.
* **`req_to_token`** is *not* a large buffer. It is
  `(max_running_requests + 1) × context_len × 4` bytes — about **7 MB**, not
  gigabytes. It is not why lowering `MAX_RUNNING` helps.
* ~~Host swap/page-cache pressure does not explain the pool variance.~~ Wrong:
  it does, through the `MemAvailable` reading the pool is sized from. See §5.
* **HiCache** offloads KV to "host" memory, which on unified memory is the same
  physical RAM. Not attempted.
* **fp4 KV cache** (`--kv-cache-dtype nvfp4`) would halve bytes/token and is
  supported by this torch build, but SGLang warns it "might lead to an accuracy
  drop" and compatibility with the QSA sparse path plus the sm_121 patches is
  unverified. Deliberately not used.

## Operational notes

* Boot takes ~10 min, most of it silent weight loading. `Found local HF
  snapshot` appearing twice is normal — the second is the NEXTN draft worker.
* The container logs in UTC; scripts stamp local time.
* The launcher's low-swap warning fires on a 16 GiB swapfile because
  `SwapTotal` is a few KB under 16 GiB and the check truncates with `%d`.
  Harmless; `start.sh` says so before the launcher pauses.
* `sgbench.sh` hardcodes `max_tokens: 512` in its parallel path, so `LongCode`
  at N>1 is not actually long, and no parallel run stresses the KV pool
  (`full token usage` peaked at 0.03).
