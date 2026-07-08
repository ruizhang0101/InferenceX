#!/usr/bin/env python3
"""Analyze Weka CC agentic-coding traces: structure + KV-cache reuse/footprint.

Reuse model (matches the corpus's local block-hash semantics): within each
trace (hash_id_scope='local'), replay requests in file order maintaining a
seen-set of block hash IDs. For each request, hit_blocks = #hash_ids already
seen; new_blocks = the rest (the prefill the engine MUST compute even with a
perfect cache). Working set = distinct hash_ids in the trace x block_size =
peak unique KV tokens the conversation touches (offload pool sizing).
"""
import json, sys
import numpy as np

PATH = sys.argv[1] if len(sys.argv) > 1 else "weka/traces.jsonl"

def pct(a, ps=(50,75,90,99)):
    if not len(a): return {p:0 for p in ps}
    a = np.asarray(a, dtype=np.float64)
    return {p: float(np.percentile(a, p)) for p in ps}

def fmt(d): return "  ".join(f"p{p}={v:,.1f}" for p,v in d.items())

def iter_reqs(trace):
    """Yield (request, is_subagent_inner) in file order."""
    for r in trace["requests"]:
        t = r.get("type")
        if t in ("n","s"):
            yield r, False
        elif t == "subagent":
            for ir in r.get("requests", []):
                yield ir, True

n_traces = 0
block_sizes = set()
req_hit_rate = []          # per-request hit rate, all requests
req_hit_rate_sub = []      # subagent-inner only
isl, osl = [], []
new_blocks_tokens = []     # per-request new (uncached) tokens = compute floor
total_blocks = 0
total_hit_blocks = 0
workingset_tokens = []     # per-trace distinct-block tokens
turns_per_trace = []
first_turn_hit = []        # cold first request per trace
later_turn_hit = []
reuse_per_trace = []       # token-weighted reuse fraction per trace

with open(PATH) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        tr = json.loads(line)
        n_traces += 1
        bs = tr["block_size"]; block_sizes.add(bs)
        seen = set()
        tb = hb = 0
        nturns = 0
        for i,(r,is_sub) in enumerate(iter_reqs(tr)):
            h = r.get("hash_ids") or []
            nb = len(h)
            isl.append(r.get("in", 0)); osl.append(r.get("out", 0))
            nturns += 1
            if nb == 0:
                continue
            hits = sum(1 for x in h if x in seen)
            hr = hits/nb
            req_hit_rate.append(hr)
            if is_sub: req_hit_rate_sub.append(hr)
            (first_turn_hit if i==0 else later_turn_hit).append(hr)
            new_blocks_tokens.append((nb-hits)*bs)
            tb += nb; hb += hits
            seen.update(h)
        total_blocks += tb; total_hit_blocks += hb
        if tb: reuse_per_trace.append(hb/tb)
        workingset_tokens.append(len(seen)*bs)
        turns_per_trace.append(nturns)

print(f"traces={n_traces}  block_size(s)={sorted(block_sizes)}  total_requests={len(isl):,}")
print(f"subagent-inner requests={len(req_hit_rate_sub):,}")
print()
print("=== per-request CACHE HIT RATE (local block hash) ===")
print(f"  all reqs:   mean={np.mean(req_hit_rate):.4f}   {fmt(pct(req_hit_rate))}")
print(f"  subagent:   mean={np.mean(req_hit_rate_sub):.4f}   {fmt(pct(req_hit_rate_sub))}")
print(f"  first turn (cold): mean={np.mean(first_turn_hit):.4f}   (n={len(first_turn_hit)})")
print(f"  later turns:       mean={np.mean(later_turn_hit):.4f}   {fmt(pct(later_turn_hit))}")
print()
print("=== TOKEN-WEIGHTED REUSE (compute actually saved) ===")
print(f"  overall reuse = cached_blocks/total_blocks = {total_hit_blocks/total_blocks:.4f}")
print(f"  => {100*total_hit_blocks/total_blocks:.1f}% of all prefill blocks were already-seen prefix")
print(f"  per-trace reuse: mean={np.mean(reuse_per_trace):.4f}  {fmt(pct(reuse_per_trace))}")
print()
print("=== PER-TURN NEW (uncached) TOKENS = prefill compute floor ===")
print(f"  mean={np.mean(new_blocks_tokens):,.0f} tok   {fmt(pct(new_blocks_tokens))}")
print(f"  (vs mean ISL={np.mean(isl):,.0f} tok -> with perfect cache you prefill ~{100*np.mean(new_blocks_tokens)/np.mean(isl):.1f}% of input)")
print()
print("=== KV WORKING SET per conversation (distinct blocks x block_size) ===")
print(f"  tokens: mean={np.mean(workingset_tokens):,.0f}   {fmt(pct(workingset_tokens))}   max={np.max(workingset_tokens):,.0f}")
# DSv4 MLA KV: ~ (kv_lora_rank 512 + rope 64)=576 elems * 1 byte(fp8) per token per layer; 61 layers
bytes_per_tok = 576 * 1 * 61
gb = np.array(workingset_tokens)*bytes_per_tok/1e9
print(f"  est DSv4 MLA KV @fp8 (~{bytes_per_tok/1024:.1f}KiB/tok): mean={gb.mean():.2f}GB  p90={np.percentile(gb,90):.2f}GB  max={gb.max():.2f}GB per conversation")
print()
print("=== ISL / OSL ===")
print(f"  ISL: mean={np.mean(isl):,.0f}  {fmt(pct(isl))}  max={np.max(isl):,}")
print(f"  OSL: mean={np.mean(osl):,.0f}  {fmt(pct(osl))}  max={np.max(osl):,}")
print(f"  ISL:OSL ratio (mean) = {np.mean(isl)/np.mean(osl):.0f}:1")
print()
print("=== turns per conversation ===")
print(f"  mean={np.mean(turns_per_trace):,.0f}  {fmt(pct(turns_per_trace))}  max={np.max(turns_per_trace):,}")
