# Bench harnesses

All scripts talk to a running server through the OpenAI-compatible `/v1/completions` endpoint. Set `BASE_URL`
(default `http://127.0.0.1:5000`) and, with auth on, `API_KEY` (the `api_key` from `tabbyAPI/api_tokens.yml`).
Prompts come from `common.py` (fixed texts; long prompts are built by repeating seed texts, so they are identical on
every machine).

| script | what it measures |
|---|---|
| `final.py` | the release benchmark: decode at 1/2/4/8/12 users × prose/chat/code/math × greedy/preset, cold prefill 2k/32k/128k, with first-token timing; GPU power, clocks and throttle reasons per section |
| `timing.py` | streamed request with first-token timing (used by `final.py`, `needle.py`, `prompt_cache.py`) |
| `prompt_cache.py` | time to first token of a ~100k prompt after its pages were evicted from VRAM, and after a server restart |
| `needle.py` | a passphrase planted in a ~520k-token prompt, retrieved at the end |
| `identity_kl.py` | greedy identity on 8 prompts and next-token top-20 KL at short and long context |
| `accuracy.py` | 50 exact-answer questions as chat messages (thinking on), graded on the final answer after the thinking |
| `accuracy_raw.py` | V2's raw-completion probe (24 prompts, answer searched in the continuation); for comparison with older results only, not a gate |
| `matrix.py` | aggregate decode throughput at 1..8 concurrent streams (V2 headline harness, end-to-end timing) |
| `decode_natural.py` | single-stream 2048-token decode on prose / code / chat / math (V2 harness, end-to-end timing) |
| `coldttft.py` | cold time to first token at 2k / 32k / 128k, with the server's resident and locked memory |
| `prefill_under_load.py` | a cold long prompt arriving while other streams decode |
| `soak.py` | long mixed-load soak (errors, memory, latency drift) |
| `profile/` | nsys-based prefill profiler |

## Timing

`timing.py` streams the response (`stream: true` with `stream_options.include_usage`) and records

- **TTFT**: request sent → first chunk with generated text (prefill, queueing and the first decode step);
- **decode t/s from the first token**: (generated tokens − 1) / (end of stream − first token);
- **end-to-end t/s**: generated tokens / (end of stream − request sent).

Token counts come from the final usage chunk, not from counting chunks: with speculative decoding one chunk can carry
several tokens. In `final.py`, aggregate decode t/s for N users is (all tokens − N) / (first token of any stream →
last stream's end); per-user decode t/s is the median of the streams' own decode speeds. The V2 tables were measured
end to end with `matrix.py` / `decode_natural.py`; for short prompts the two agree within the first token's time.

```bash
python bench/final.py --label B1 --out B1.jsonl                      # everything, 3 runs per cell
python bench/final.py --label quick --sections decode --conc 1,8 --tags code --sampling greedy --reps 1
python bench/prompt_cache.py evict --key run1      # cold, hot, fill the GPU pool, resend
# restart the server, then
python bench/prompt_cache.py send --key run1
python bench/needle.py
```

## Gate policy for changes

A change passes when, against the reference build measured in the same session:

- **`accuracy.py`** scores the same as the reference (every miss is reported with its full answer), and

- **top-1 agreement** of the next-token distribution matches on every probe prompt (short prompts and the
  8.7k / 35k / 157k long-context prompts), and
- **KL** (reference || candidate, over the reference's top-20) stays inside the noise floor measured by running the
  reference against itself, separately for short and long context.

Greedy text / token identity is reported for every change but is **not a hard gate**. The server is not deterministic
against itself: batched verification, the growable KV pool and kernel autotune choices move near-tied tokens between
sessions (a reference build compared with its own restart has flipped 1-2 of the 8 identity prompts, and concurrent
8-stream identity varies from 4/8 to 8/8 run to run). A greedy difference without a top-1 or KL change is a near-tie
flip, not a regression.

Speed claims use back-to-back A, A, B, B runs (one server restart per configuration) with matched host load.
