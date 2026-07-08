# Run: DeepSeek-V4 KV-offload agentic benchmark (MI355X, agentx-v1.0)

Run the agentic trace-replay benchmark for DeepSeek-V4-Pro and compare KV-offload
modes, using the **official `Amd/agentx-v1.0`** recipe.

**Script:** `benchmarks/single_node/agentic/dsv4_fp4_mi355x_vllm.sh` (FP4, MI355X).
**Model:** `deepseek-ai/DeepSeek-V4-Pro`. **Image:** per `amd-master.yaml`
(ROCm vLLM). This branch = `Amd/agentx-v1.0` + these docs and the trace/cache
analysis under `scratch/`.

Offload is selected by env (official interface):
- `KV_OFFLOADING=none` — no offload (GPU KV only), baseline.
- `KV_OFFLOADING=dram` + `KV_OFFLOAD_BACKEND=<native|mooncake|lmcache|hicache>` — offload.

---

## 1. Setup

```bash
cd ~/InferenceX
git fetch myfork && git checkout feat/dsv4-mi355x-vllm-lmcache   # = Amd/agentx-v1.0 + docs
git submodule update --init utils/aiperf utils/agentic-benchmark
ls benchmarks/single_node/agentic/dsv4_fp4_mi355x_vllm.sh        # official script
```

## 2. Storage prerequisites (bare box — this bites)

Two separate needs:
- **Container images** (~50 GB vLLM image + snapshots) MUST live on a **local** FS
  (ext4/xfs). Docker's overlay store (`overlay2` **and** the containerd `overlayfs`
  snapshotter) **cannot run on NFS** → `Input/output error`. They live under
  `/var/lib/docker` / `/var/lib/containerd` on `/`.
- **HF weights** (~865 GB) can live on **NFS** (plain file reads). Mount as HF cache.

If `/` is small/full (`no space left`, `mkdir ~/.docker` fails):
```bash
df -h /;  lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT
# common: root LV claims only part of the NVMe -> grow it into free VG space:
sudo lvextend -l +100%FREE /dev/ubuntu-vg/ubuntu-lv && sudo resize2fs /dev/ubuntu-vg/ubuntu-lv
```
Don't move the image store to NFS. Watch NFS **write quotas** too: a directory
SmartQuota (`df` hides it) can EIO large writes; and HF **Xet / hf_transfer** do
mmap/parallel writes NFS may reject — set `HF_HUB_DISABLE_XET=1
HF_HUB_ENABLE_HF_TRANSFER=0` for a plain sequential download if you see EIO.

## 3. Launch the container (docker)

```bash
docker run -it --rm --network=host --ipc=host --shm-size=64g \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v "$PWD":/workspace \
  -v /path/to/hf-cache:/root/.cache/huggingface \
  -w /workspace --entrypoint /bin/bash \
  <vllm-rocm-image-from-amd-master.yaml>
```
`rocm-smi` inside must show 8 GPUs → `TP=8`.

## 4. Run — baseline FIRST, then offload

```bash
export INFMAX_CONTAINER_WORKSPACE=/workspace MODEL_PREFIX=dsv4 \
  MODEL=deepseek-ai/DeepSeek-V4-Pro HF_HUB_CACHE=/root/.cache/huggingface \
  PORT=8000 TOTAL_CPU_DRAM_GB=2500 TP=8 EP_SIZE=1 DP_ATTENTION=false
SCRIPT=benchmarks/single_node/agentic/dsv4_fp4_mi355x_vllm.sh

# baseline (no offload) — proves the model serves first:
CONC=8 DURATION=1800 KV_OFFLOADING=none \
  RESULT_DIR=/workspace/results/none_c8 bash $SCRIPT

# LMCache DRAM offload — the comparison:
CONC=8 DURATION=1800 KV_OFFLOADING=dram KV_OFFLOAD_BACKEND=lmcache \
  RESULT_DIR=/workspace/results/lmcache_c8 bash $SCRIPT
```
- Other backends: `KV_OFFLOAD_BACKEND=native | mooncake | hicache`.
- Quick smoke: `CONC=4 DURATION=600 AIPERF_UNSAFE_OVERRIDE=true`.
- `MAX_NUM_SEQS` is auto (`2×CONC`). Parallelism: pure TP (`DP_ATTENTION=false EP_SIZE=1`);
  DEP via `DP_ATTENTION=true EP_SIZE=8`.
- The official recipe keeps the **hybrid KV-cache manager ON**
  (`--no-disable-hybrid-kv-cache-manager`) → efficient DSv4 (MLA + sparse-attn) KV
  sizing, so the model's 1M default context fits without capping. If you still hit
  `estimated maximum model length ... larger than available KV cache`, the KV/token
  is inflated — add `--max-model-len` (e.g. 131072) or raise `--gpu-memory-utilization`.

## 5. Metrics — where to find them

Per `RESULT_DIR`:
- `$RESULT_FILENAME.json` (from `process_agentic_result.py`) — **mean / p75 / p90 / p95** in **seconds**: `mean_ttft`, `p90_ttft`, `mean_e2el`, `mean_intvty` (interactivity = tok/s/user = 1/ITL), `total_tput_tps`, `output_tput_tps`. **No median.**
- `aiperf_artifacts/.../profile_export_aiperf.json` — full distribution (**avg/p50/p90/p99**, in **ms**): `time_to_first_token`, `inter_token_latency`, `request_latency`, `output_token_throughput`, `output_token_throughput_per_user`.
- `metrics_plots.png` — server KV usage + prefix-cache-hit time series. `benchmark.log` — phase progress + `errors=`.

Your 6 metrics:
| want | field | file |
|---|---|---|
| total tok/s/gpu | `total_tput_tps` or `output_tput_tps` **÷ TP** | agg JSON |
| interactivity (tok/s/user) | `mean_intvty` | agg JSON |
| **median** interactivity | `output_token_throughput_per_user` p50 | aiperf export |
| e2e latency | `mean_e2el`/`p90_e2el` (s) or `request_latency` (ms) | agg JSON / aiperf |
| p90 TTFT | `p90_ttft` | agg JSON |
| **median** TTFT | `time_to_first_token` p50 | aiperf export |

```bash
f=$(find $RESULT_DIR/aiperf_artifacts -name profile_export_aiperf.json | head -1)
jq '{ttft_p50:.time_to_first_token.p50, ttft_p90:.time_to_first_token.p90,
     e2e_p50:.request_latency.p50, interactivity_p50:.output_token_throughput_per_user.p50,
     interactivity_avg:.output_token_throughput_per_user.avg,
     out_tok_per_s:.output_token_throughput.avg}' "$f"   # tok/s/gpu = out_tok_per_s / TP
```

**What proves offload works:** `KV_OFFLOADING=dram+lmcache` vs `none` at the **same CONC** —
lower **TTFT** (mean + p90/median), higher sustained **prefix-cache hit rate**, higher
**tok/s/gpu**, with **TPOT/ITL not regressing**. Gap widens as CONC rises. Units differ
(agg = s, aiperf = ms) — don't mix in one plot.

## 6. Trace / cache analysis (context)

`scratch/weka_traces/` — analysis of the replay corpus (`semianalysisai/cc-traces-weka-061526`):
prefill-heavy (ISL:OSL ≈ 211:1), **98.4% prefix-reusable** (upper bound), per-conversation
KV footprint, and a finite-LRU **LMCache cache-simulator** sweep (`FINDINGS.md` / `FINDINGS.html`)
showing GPU-only hit rate cratering under concurrency — the quantitative case for offload.
Reproduce with `analyze_weka_cache.py` / `make_figs.py` / `to_lookup_jsonl.py`.

## Troubleshooting

| symptom | fix |
|---|---|
| `no space left` / `/var/lib/containerd` huge | grow local `/` (`lvextend`+`resize2fs`); don't move images to NFS. §2 |
| `Input/output error` writing to /mnt NFS | NFS can't host overlay images; large writes may hit a hidden quota; disable Xet/hf_transfer for downloads. §2 |
| `estimated maximum model length ... KV cache` | keep hybrid-KV manager on (official default); else `--max-model-len 131072` / raise `--gpu-memory-utilization`. §4 |
| warmup stalls (returned frozen, GPU busy, errors=0) | giant-context prefills; use the 256k corpus (`WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_061526_256k`) or smaller CONC |
| can't Ctrl-C | script backgrounds `vllm serve`; `pkill -9 -f "vllm serve"` then `pkill -9 -f "^aiperf "`; verify `rocm-smi --showuse` = 0% |

Always run `KV_OFFLOADING=none` first to isolate serving from offload.
