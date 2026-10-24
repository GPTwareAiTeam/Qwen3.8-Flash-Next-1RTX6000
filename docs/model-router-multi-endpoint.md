# Model Router: multi-endpoint providers

Small change made in the local Model Router (`~/projects/app-launcher`, Next.js on port 8734) so that one
provider entry can front several OpenAI-compatible endpoints. Motivation: the machine hosts two engines —
a DeepSeek vLLM on `127.0.0.1:8001` and this Qwen/TabbyAPI on `127.0.0.1:5000` — and both belong under the
provider named `L-P7`.

## Before

`Provider` had a single `baseUrl`; `Model` had no endpoint. Discovery fetched `{baseUrl}/v1/models`,
cached per provider id, and overwrote the cache with seed-only models when a fetch failed (so a down
endpoint's models vanished). Multi-endpoint providers were not expressible.

## After

|file|change|
|---|---|
|`src/lib/router/types.ts`|`Provider.baseUrls?: string[]` (extra endpoints; `baseUrl` stays the primary), `Model.baseUrl?: string` (endpoint that served the model, absent = primary), mirrored in `ProviderPublic`|
|`src/lib/router/store.ts`|`normalizeBaseUrls(primary, extra)` — trims, strips trailing `/`, drops blanks/non-strings, case-insensitive dedupe, never duplicates the primary; emitted by `toPublic`, round-tripped by the config PUT route so a lane edit cannot erase it|
|`src/lib/router/discovery.ts`|`providerEndpoints()` (primary first, deduped); fetchers take an endpoint and tag each model; `discoverModels` loops endpoints with independent try/catch and, for a failed endpoint, re-inserts only that endpoint's previously cached models while live endpoints refresh; merge keeps primary-first precedence on id collisions; `modelEndpoint()` resolves the serving endpoint|
|`src/lib/router/engine.ts`|chat URL, Codex `/responses`, the Anthropic request and the `/v1/models` probe all use the model's own endpoint|
|`src/app/api/router/config/route.ts`|PUT preserves `baseUrls` when the editor omits the key; an explicit `[]` clears it|
|`data/router-config.json`|`L-P7` gains `"baseUrls": ["http://127.0.0.1:5000/v1"]` (backup written beside it)|

## Verified

- `GET /api/router/config` -> `baseUrl: http://127.0.0.1:8001/v1`, `baseUrls: ["http://127.0.0.1:5000/v1"]`.
- Discovery with `:8001` down: DeepSeek entries preserved from cache, TabbyAPI entries refreshed live.
- Routing split by endpoint (stub on `:8001`): `deepseek-ai/DeepSeek-V4.1-Flash` hit `:8001`,
  `Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8` hit `:5000`, and the Qwen request never touched `:8001`.
- Single-endpoint providers unchanged (ollama id set identical before/after, all URLs bit-identical).
- `pnpm build` clean; `systemctl --user restart app-launcher-web` leaves 8734 serving.

Note: TabbyAPI's `/v1/models` lists every model directory under its model dir, so the Qwen endpoint also
advertises the DeepSeek checkpoint directories that happen to live there. Only the loaded model is servable.
