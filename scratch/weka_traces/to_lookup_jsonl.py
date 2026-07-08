#!/usr/bin/env python3
"""Convert Weka CC traces -> LMCache MP_LOOKUP lookup-hash JSONL.

Each model request (main + subagent-inner) becomes one lookup event:
  chunk_hashes : per-trace-namespaced block hashes (local scope, no cross-trace alias)
  seq_len      : input tokens
  chunk_size   : 64
  timestamp    : global emission order (sequential conversations)

--mode sequential : conversations replayed one after another (default)
--mode interleave : round-robin across --conc lanes (models concurrent serving;
                    when a lane's conversation ends, the next trace fills it)
"""
import json, sys, argparse
ap=argparse.ArgumentParser()
ap.add_argument("-i","--input",required=True)
ap.add_argument("-o","--output",required=True)
ap.add_argument("--mode",choices=["sequential","interleave"],default="sequential")
ap.add_argument("--conc",type=int,default=32)
ap.add_argument("--max-traces",type=int,default=0)
a=ap.parse_args()

def flatten(tr):
    """All model requests in trace order: (seq_len, hash_ids)."""
    out=[]
    for r in tr["requests"]:
        if r["type"] in ("n","s"):
            out.append((r.get("in",0), r.get("hash_ids") or []))
        elif r["type"]=="subagent":
            for ir in r["requests"]:
                out.append((ir.get("in",0), ir.get("hash_ids") or []))
    return out

# load traces (as flattened request lists), namespaced hashes
traces=[]
with open(a.input) as f:
    for ti,line in enumerate(f):
        if a.max_traces and ti>=a.max_traces: break
        tr=json.loads(line)
        reqs=[(sl,[f"{ti}_{h}" for h in hs]) for sl,hs in flatten(tr)]
        traces.append(reqs)

def emit(o, reqs, ts0):
    ts=ts0
    for seq_len,hashes in reqs:
        o.write(json.dumps({
            "timestamp": ts, "chunk_hashes": hashes,
            "seq_len": seq_len, "chunk_size": 64, "model_name": "dsv4",
        })+"\n")
        ts+=1
    return ts

n_events=0
with open(a.output,"w") as o:
    if a.mode=="sequential":
        ts=0
        for reqs in traces:
            ts=emit(o,reqs,ts); n_events+=len(reqs)
    else:
        # interleave: --conc lanes; each lane is a cursor into a queue of traces
        import collections
        q=collections.deque(traces)
        lanes=[]  # each: (iterator over reqs)
        for _ in range(min(a.conc,len(q))):
            lanes.append(iter(q.popleft()))
        ts=0
        active=list(range(len(lanes)))
        while active:
            nxt=[]
            for li in active:
                try:
                    seq_len,hashes=next(lanes[li])
                    o.write(json.dumps({"timestamp":ts,"chunk_hashes":hashes,
                        "seq_len":seq_len,"chunk_size":64,"model_name":"dsv4"})+"\n")
                    ts+=1; n_events+=1; nxt.append(li)
                except StopIteration:
                    if q: lanes[li]=iter(q.popleft()); nxt.append(li)  # refill lane
            active=nxt
print(f"wrote {n_events} events ({len(traces)} traces, mode={a.mode}) -> {a.output}")
