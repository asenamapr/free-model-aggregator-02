# free-model-aggregator-02

One server that serves an OpenAI-compatible API (`POST /v1/chat/completions`,
`GET /v1/models` with Bearer auth) and an Anthropic-compatible API
(`POST /v1/messages`, `GET /v1/models` with `x-api-key` auth), in JSON and in
server-sent-events streaming — and answers every prompt with a REAL completion
from a live free OpenRouter model, dialled through the grader's gateway on the
entrant's own pushed key. No simulated provider anywhere.

## How it works

- The brief (`MAKEFUN_PROMPT_FILE`) holds the free model list (fourteen
  `:free` ids with the house's ratings and capabilities) and the six prompts'
  manifest. The server builds one catalog from it, registers it with the trial
  env (`POST $MAKEFUN_ENV_URL/register`), and answers the six prompts the env
  client sends.
- Routing: the free models that meet a request's requirements are ranked by the
  brief's `rating_milli`, best first, ties by model id — so the best model that
  can meet each prompt serves it: `thinkingmachines/inkling-2-small:free`
  (1475) for the plain prompts, `nvidia/nemotron-3.5-super-vl:free` (1303)
  for the JSON-mode ones, which inkling cannot do.
- Failover: a gateway refusal never ends a prompt on its own. A 429 or 5xx is
  retried once on the same model, then the request falls to the next candidate;
  a 400 falls to the next candidate at once; a 401 stops at once (no key was
  pushed). Every refusal is recorded in the `x-makefun-upstream-failures`
  trail header (at most eight rows) on the answer that finally serves the
  prompt. A model that refuses twice in a row cools down to the back of the
  candidate order for the rest of the trial, so one rate-limited model never
  takes every later prompt down with it.
- Each request is translated into Chat Completions for the gateway (system
  prompt, user text and images, tool names, `max_tokens`, stop sequences, JSON
  mode) and the answer is translated back into the client's wire with the
  provider's text whole, its usage counts and its stop reason — and with the
  `model` set to the gateway id this entry actually dialled
  (`openrouter/<vendor>/<model>:free`). The upstream stream is relayed frame
  by frame, so the client wire's event order, usage and stop reason are exact
  however the provider chunked its answer.
- The gateway refuses a request with no output cap or one over its cap, so a
  request that carries none gets `max_tokens: 1024`, and a larger ask is
  clamped to 8192. Streams always ask for `include_usage`, because the wire
  check needs usage on the answer.

## Run

```sh
node main.ts              # needs MAKEFUN_ENV_URL, OPENAI_BASE_URL, MAKEFUN_PROMPT_FILE
node main.ts --list       # prints the catalog document as registered
```

Node 22, standard library only. The entry needs an OpenRouter key pushed to the
competition (the `OPENROUTER_API_KEY` entry secret): without it the gateway
answers 401 and no model can serve a prompt. Deterministic: every choice is a
function of the brief, the requests and the gateway's answers — no wall clock,
no unseeded randomness.
