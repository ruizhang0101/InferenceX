# HANDOFF — DeepSeek-V4 KV-offload agentic benchmark (MI355X, agentx-v1.0)

Goal: benchmark DeepSeek-V4-Pro under the agentic trace-replay and compare KV-offload
modes (esp. **none vs LMCache DRAM**) on MI355X.

Base: **`Amd/agentx-v1.0`**. This branch = that base + these docs + the trace/cache
analysis in `scratch/`. Run steps: **`RUN_INSTRUCTIONS.md`**.

## Use the official recipe
`benchmarks/single_node/agentic/dsv4_fp4_mi355x_vllm.sh` (from `Amd/agentx-v1.0`) is
the maintained script — unified offload interface:
- `KV_OFFLOADING=none` → baseline (GPU KV only)
- `KV_OFFLOADING=dram` + `KV_OFFLOAD_BACKEND=native|mooncake|lmcache|hicache` → offload

(An earlier hand-built `dsv4_fp4_mi355x_vllm.sh` + `dsv4_fp8_mi300x_vllm.sh` on the v0.4
base were **superseded by this official version and dropped**. History: branch
`backup/dsv4-v04-work` locally, if ever needed.)

## Key findings that still apply

1. **Keep the hybrid KV-cache manager ON** (`--no-disable-hybrid-kv-cache-manager`, the
   official default). DeepSeek-V4 KV = MLA latent + sparse-attention indexer; the hybrid
   manager sizes it efficiently. **Disabling it inflates KV/token ~13×** (~453 KiB/tok vs
   ~34 KiB/tok expected) → vLLM reserves KV for the 1M default context → `estimated maximum
   model length ... larger than available KV cache`. That's why the earlier offload path
   (which disabled it) needed a `--max-model-len` cap and the official does not.

2. **Trace/cache analysis** (`scratch/weka_traces/`, corpus `cc-traces-weka-061526`):
   prefill-heavy (ISL:OSL ≈ 211:1), **98.4% prefix-reusable** (infinite-cache upper bound),
   per-conversation KV mean ~29 GB / p90 ~67 GB. LMCache **cache-simulator** sweep shows
   GPU-only hit rate cratering under concurrency (e.g. 64 GiB / CONC-32 → ~17%) — the
   quantitative motivation for offload. See `FINDINGS.md` / `FINDINGS.html`.

3. **Storage (bare box):** container images need a **local** FS (overlay can't run on NFS);
   HF weights can sit on NFS. Grow a small root LV with `lvextend +100%FREE` + `resize2fs`.
   Watch NFS directory **quotas** (hidden from `df`, EIO large writes) and disable HF
   **Xet/hf_transfer** for NFS downloads. Details in RUN_INSTRUCTIONS §2.

4. **Metrics:** agg JSON (`$RESULT_FILENAME.json`) has mean/p75/p90/p95 (seconds); **medians**
   are only in `aiperf_artifacts/.../profile_export_aiperf.json` (ms). tok/s/gpu = throughput ÷ TP.
   Table + jq in RUN_INSTRUCTIONS §5.

## Next
Run `KV_OFFLOADING=none` first (serving sanity), then `KV_OFFLOADING=dram
KV_OFFLOAD_BACKEND=lmcache` at the same CONC; compare TTFT (mean/p90/median), prefix-cache
hit rate, tok/s/gpu, and confirm TPOT doesn't regress. Sweep CONC to widen the gap.
