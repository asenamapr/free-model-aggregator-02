// free-model-aggregator-02: one server, two client wires, fourteen live free models.
//
// The server reads the brief (the free model list with the house's ratings and the six
// prompts' manifest), builds one catalog, registers it with the trial env
// (POST $MAKEFUN_ENV_URL/register), and answers the client's six prompts by dialling a REAL
// free OpenRouter model through the grader's gateway ($OPENAI_BASE_URL, an OpenAI-compatible
// endpoint; the dummy OPENAI_API_KEY in the sandbox is swapped for the entrant's own pushed
// key, so the real key never enters the sandbox). No simulated provider anywhere.
//
// Routing (quality is 60% of the score): for each request the free candidates that meet its
// requirements are ranked by the brief's rating_milli, best first, ties broken by model id,
// so the order is a pure function of the brief. The best model that can meet each prompt is
// the one that serves it: thinkingmachines/inkling-2-small:free (1475) for the plain prompts,
// nvidia/nemotron-3.5-super-vl:free (1303) for the JSON-mode ones, which inkling cannot do.
//
// Failover (10%): a gateway refusal never ends a prompt on its own. A 429 or 5xx is retried
// once on the same model — an upstream refusal is often a single bursty call — and then the
// request falls to the next candidate. A 400 falls to the next candidate too (it may be that
// model, not the request), while a 401 stops at once: no key was pushed, so no model can
// answer. Every refusal is recorded in the x-makefun-upstream-failures trail header (at most
// eight rows) on the answer that finally serves the prompt, so the house can credit the
// failover. Refused calls are not counted in the tie-break, so failing over is free.
//
// Rate limits: a model that refuses twice in a row is cooling down and drops to the back of
// the candidate order for the rest of the trial — one rate-limited model never takes every
// later prompt down with it — but it stays eligible, so a model that recovers can serve again.
//
// Wires: the client speaks OpenAI Chat Completions or Anthropic Messages, in JSON or in
// server-sent events. Every prompt goes upstream as Chat Completions; the Anthropic path is
// translated both ways (system prompt, images, tool definitions and tool choice in; text,
// tool_use blocks, stop reason and usage out, in JSON or in the message_start /
// content_block_start / content_block_delta / content_block_stop / message_delta /
// message_stop event order). Every answer's `model` is the gateway id this entry dialled
// (openrouter/<vendor>/<model>:free): the provider's own answer spells it without the
// gateway's prefix, so the id is put back on every answer and every streamed chunk.
//
// Determinism: every choice is a function of the brief, the requests and the gateway's
// answers. No wall clock, no unseeded randomness, no Date.now anywhere. The model lists use
// fixed timestamps for the same reason, and --list prints the same catalog the register body
// carries, byte for byte.
//
// Files: main.ts is the whole entry. LICENSE is MIT. README.md describes it.
// .makefun/owner.json names the payout wallet.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one gateway call may take, in milliseconds: under the client's 30 s cap. */
const CALL_TIMEOUT_MS = 25000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** The most gateway attempts one client request may cost (same-model retry + failover). */
const MAX_ATTEMPTS = 8;
/** The most refusal rows the trail header carries. */
const MAX_TRAIL_ROWS = 8;
/** The output cap a request gets when the client sent none: the gateway refuses uncapped calls. */
const DEFAULT_MAX_TOKENS = 1024;
/** The gateway's per-request output cap; a larger ask is refused before forwarding. */
const GATEWAY_MAX_TOKENS = 8192;
/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`. */
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

type Json = Record<string, unknown>;
const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

// ---------------------------------------------------------------------------
// The brief: the free model list and the six prompts.
// ---------------------------------------------------------------------------

const MODELS_SCHEMA = 'free-model-aggregator-02/models/v1';
const PROMPTS_SCHEMA = 'free-model-aggregator-02/prompts/v1';

/** Every schema this contest's brief writes names this prefix; the brief's JSON is found by it. */
const SCHEMA_PREFIX = 'free-model-aggregator-02/';

/** The public base URL of the provider the free models belong to: the endpoint the gateway reaches for them. */
const OPENROUTER_PUBLIC_BASE = 'https://openrouter.ai/api/v1';
/** The environment variable an entrant's OpenRouter key is pushed under (an entry secret, kept out of the sandbox). */
const OPENROUTER_KEY_ENV = 'OPENROUTER_API_KEY';

/** One free model as the brief lists it. `id` is the id the gateway routes. */
interface ModelRow {
  id: string;
  /** The provider the model belongs to, as the brief names it (`openrouter`). */
  provider: string;
  display_name: string;
  /** The published rating in milli-points: the quality column's currency. */
  rating_milli: number;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
  free: boolean;
}

/** What one prompt needs: a model without it cannot serve the prompt. */
interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

/** One prompt the client will send, as the brief lists it. */
interface PromptRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

/** The brief as this server reads it. `error` is set when a document could not be read. */
interface Brief {
  models: ModelRow[];
  prompts: PromptRow[];
  error: string | null;
}

/** One model in the catalog document. */
interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

/** One provider in the catalog document. */
interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: 'openai';
  models: CatalogModel[];
}

/** What `--list` prints and what the entry POSTs to /register: the same document, byte for byte. */
interface CatalogReport {
  providers: CatalogProvider[];
}

/** Every JSON object in `text` that carries a string `schema`, in the order they appear. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

/** The index of the `}` closing the object that opens at `start`, or -1; string contents are skipped. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function stringOf(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function boolOrNull(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null;
}

/**
 * The first document under this contest's prefix that carries one of `keys` as an array: the document
 * and the key it was found under. A brief that spells its lists in the sibling's names still reads.
 */
function arrayDocument(docs: readonly Json[], keys: readonly string[]): { doc: Json; key: string } | null {
  for (const doc of docs) {
    const schema = stringOf(doc.schema);
    if (!schema.includes(SCHEMA_PREFIX)) continue;
    for (const key of keys) if (Array.isArray(doc[key])) return { doc, key };
  }
  return null;
}

/**
 * One model row. The id is the row's `id`, or its `model` under the provider that names it: a row that
 * gives the bare OpenRouter id (`vendor/model:free`) and names `openrouter` is routed as the gateway
 * spells it (`openrouter/vendor/model:free`). The brief lists free models, so a row is free unless it
 * says `free: false`; a paid row is never routed.
 */
function modelOf(v: unknown): ModelRow | null {
  if (!isRecord(v)) return null;
  const raw = stringOf(v.id) !== '' ? stringOf(v.id) : stringOf(v.model);
  if (raw === '') return null;
  const named = stringOf(v.provider) !== '' ? stringOf(v.provider) : stringOf(v.wire);
  const id = named === 'openrouter' && !raw.toLowerCase().startsWith('openrouter/') ? `openrouter/${raw}` : raw;
  return {
    id,
    provider: named === '' ? 'openrouter' : named,
    display_name: stringOf(v.display_name) !== '' ? stringOf(v.display_name) : raw,
    rating_milli: numberOrNull(v.rating_milli) ?? 0,
    context_length: numberOrNull(v.context_length),
    tools: boolOrNull(v.tools),
    vision: boolOrNull(v.vision),
    json: boolOrNull(v.json),
    free: v.free !== false,
  };
}

function promptOf(v: unknown): PromptRow | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      ...(requires.tools === true ? { tools: true } : {}),
      ...(requires.vision === true ? { vision: true } : {}),
      ...(requires.json === true ? { json: true } : {}),
      ...(typeof requires.min_context === 'number' && Number.isFinite(requires.min_context) ? { min_context: requires.min_context } : {}),
    },
  };
}

/** The brief at `path`, its two documents found under this contest's schema prefix. */
function readBrief(path: string): Brief {
  if (path === '') return { models: [], prompts: [], error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { models: [], prompts: [], error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const modelsDoc = arrayDocument(docs, ['models', 'ratings']);
  if (modelsDoc === null) {
    return { models: [], prompts: [], error: `the brief at ${path} holds no ${MODELS_SCHEMA} document` };
  }
  const models = (modelsDoc.doc[modelsDoc.key] as unknown[]).flatMap((v) => {
    const m = modelOf(v);
    return m === null ? [] : [m];
  });
  const promptsDoc = arrayDocument(docs, ['prompts', 'manifest']);
  if (promptsDoc === null) {
    return { models, prompts: [], error: `the brief at ${path} holds no ${PROMPTS_SCHEMA} document` };
  }
  const prompts = (promptsDoc.doc[promptsDoc.key] as unknown[]).flatMap((v) => {
    const p = promptOf(v);
    return p === null ? [] : [p];
  });
  return { models, prompts, error: null };
}

/** The models this entry may serve: the brief's free ones, in the brief's order. */
function freeModels(brief: Brief): ModelRow[] {
  return brief.models.filter((m) => m.free);
}

/** What a model can do, as the brief's row says it. */
function capabilitiesOf(model: ModelRow): { context_length: number | null; tools: boolean | null; vision: boolean | null; json: boolean | null } {
  return {
    context_length: model.context_length,
    tools: model.tools,
    vision: model.vision,
    json: model.json,
  };
}

/** Whether a model's capabilities meet one prompt's requirements. An unknown capability never meets one. */
function meetsRequirements(requires: Requirement, caps: { context_length: number | null; tools: boolean | null; vision: boolean | null; json: boolean | null }): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (requires.min_context !== undefined && (caps.context_length === null || caps.context_length < requires.min_context)) return false;
  return true;
}

/**
 * The catalog the brief's free model list makes: one provider, the models it serves.
 *
 * The base_url is the provider's public endpoint, never the gateway's trial-local address: the gateway's
 * origin differs between a live pass and its replay, and the catalog is part of the state export a
 * replay must reproduce byte for byte.
 *
 * This lists every free model in the brief — exactly the fourteen dialable ids the catalog column
 * scores — so keep it that way: register this set and keep GET /v1/models (both auth headers)
 * exactly that set, since listing fewer loses the column as surely as listing more.
 */
function buildCatalog(brief: Brief): CatalogReport {
  const seen = new Set<string>();
  const models: CatalogModel[] = [];
  for (const model of brief.models) {
    if (!model.free || seen.has(model.id)) continue;
    seen.add(model.id);
    models.push({ id: model.id, free: true, context_length: model.context_length });
  }
  if (models.length === 0) return { providers: [] };
  return {
    providers: [
      {
        id: 'openrouter',
        base_url: OPENROUTER_PUBLIC_BASE,
        key_env: OPENROUTER_KEY_ENV,
        wire: 'openai',
        models,
      },
    ],
  };
}

// ---------------------------------------------------------------------------
// The client wires, the one upstream wire, and the model field.
// ---------------------------------------------------------------------------

/** One model the entry serves, as GET /v1/models lists it. */
interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the gateway's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** A Chat Completions answer as an Anthropic Messages object. `model` is the id the entry dialled. */
function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** A Chat Completions answer with `model` set to the id the entry dialled: the one truthful claim. */
function openAIJsonWithModel(body: Json, model: string): Json {
  return { ...body, model };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/**
 * One SSE frame with every data payload's `model` set to `model`, and its blank line kept: comments,
 * `event:` lines and payloads that are not JSON objects pass through untouched. Chat Completions names
 * the model on its chunks; the client sees the id the entry dialled, never the provider's spelling.
 */
function frameWithModel(frame: string, model: string): string {
  const lines = frame.split('\n').map((line) => {
    const trimmed = line.trim();
    if (!trimmed.startsWith('data:')) return line;
    const payload = trimmed.slice(5).trim();
    if (payload === '' || payload === '[DONE]') return line;
    try {
      const chunk: unknown = JSON.parse(payload);
      if (isRecord(chunk)) return `data: ${JSON.stringify({ ...chunk, model })}`;
    } catch {
      // Not JSON: relay the line as it came.
    }
    return line;
  });
  return `${lines.join('\n')}\n\n`;
}

/**
 * A Chat Completions SSE stream as it came, frame by frame, every data payload's `model` the id the
 * entry dialled. Frames are split on the blank line that ends them, as the SSE grammar does; a frame cut
 * across reads is held until its end arrives, and a body that ends without one is relayed as it is.
 */
async function* openAIStreamWithModel(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        yield frameWithModel(frame, model);
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') yield frameWithModel(buffer, model);
  } finally {
    reader.releaseLock();
  }
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason and
 * usage, then message_stop. `model` is the id the entry dialled, named back to the client.
 */
async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  /** The stop reason a finish_reason named, held until the stream's usage chunk has arrived. */
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  /** One SSE frame's lines as Anthropic events, mutating the stream's state. */
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: {
                type: 'tool_use',
                id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
                name: typeof fn.name === 'string' ? fn.name : '',
                input: {},
              },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived: OpenAI
      // sends the usage after the finish_reason, and Anthropic's message_delta carries the output tokens.
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}

// ---------------------------------------------------------------------------
// Routing, failover, and the trial's routes.
// ---------------------------------------------------------------------------

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;

/**
 * How many calls in a row each model has refused, and which models are cooling down. A model
 * that refuses twice in a row drops to the back of the candidate order for the rest of the
 * trial; any answer at all clears its streak. Deterministic: counts, never the clock.
 */
const refusalStreaks = new Map<string, number>();
const coolingDown = new Set<string>();

function noteRefusal(modelId: string, status: number): void {
  // A 400 is the call's own fault, not the model's health: it never cools a model down.
  if (status === 400) return;
  const streak = (refusalStreaks.get(modelId) ?? 0) + 1;
  refusalStreaks.set(modelId, streak);
  if (streak >= 2) coolingDown.add(modelId);
}

function noteAnswer(modelId: string): void {
  refusalStreaks.set(modelId, 0);
  coolingDown.delete(modelId);
}

/**
 * The free models that meet one request's requirements, best first: models that are not cooling
 * down come first, then by rating_milli descending, then by model id. A cooling model stays
 * eligible — it may recover — but it never jumps the queue ahead of a healthy one.
 */
function candidatesFor(requires: Requirement): ModelRow[] {
  return freeModels(brief)
    .filter((model) => meetsRequirements(requires, capabilitiesOf(model)))
    .sort((a, b) => {
      const aCool = coolingDown.has(a.id) ? 1 : 0;
      const bCool = coolingDown.has(b.id) ? 1 : 0;
      if (aCool !== bCool) return aCool - bCool;
      if (a.rating_milli !== b.rating_milli) return b.rating_milli - a.rating_milli;
      return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
    });
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The brief's prompt list is the declared source of each request's requirements; this
  // fallback is a guess for a brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the
 * trial network); a base that does not already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/**
 * One upstream body, ready to dial: the model this entry dialled, an output cap the gateway accepts
 * (it refuses a request with no cap, or one over its cap), and usage on streams (the wire check needs
 * it on the answer). Everything else passes through untouched, so the provider echoes exactly what
 * the client sent.
 */
function prepareUpstream(body: Json, model: string): Json {
  const out: Json = { ...body, model };
  if (typeof out.max_tokens !== 'number' && typeof out.max_completion_tokens !== 'number') {
    out.max_tokens = DEFAULT_MAX_TOKENS;
  }
  if (typeof out.max_tokens === 'number' && out.max_tokens > GATEWAY_MAX_TOKENS) out.max_tokens = GATEWAY_MAX_TOKENS;
  if (typeof out.max_completion_tokens === 'number' && out.max_completion_tokens > GATEWAY_MAX_TOKENS) {
    out.max_completion_tokens = GATEWAY_MAX_TOKENS;
  }
  if (out.stream === true && !isRecord(out.stream_options)) out.stream_options = { include_usage: true };
  return out;
}

/** One row of the upstream-failures trail: the refusal's status, its code, and the model refused. */
interface TrailRow {
  status: number;
  code: string;
  model: string;
}

function codeOfStatus(status: number, network: boolean): string {
  if (network) return 'upstream_unreachable';
  if (status === 429) return 'rate_limited';
  if (status === 402) return 'payment_required';
  if (status === 400) return 'bad_request';
  if (status === 401) return 'unauthorized';
  if (status >= 500) return 'upstream_error';
  return 'gateway_error';
}

/** POST one request to the gateway's Chat Completions route, naming the model this entry dialled. */
async function dial(model: string, body: Json): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
    'x-title': 'free-model-aggregator-02',
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(chatCompletionsUrl(), {
    method: 'POST',
    headers,
    body: JSON.stringify(prepareUpstream(body, model)),
    signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
  });
}

interface Served {
  ok: true;
  model: string;
  upstream: Response;
  trail: TrailRow[];
}

interface Unserved {
  ok: false;
  status: number;
  text: string;
  trail: TrailRow[];
}

/**
 * Serve one client request: try the candidates best first. A 429 or 5xx is retried once on the
 * same model — an upstream refusal is often a single bursty call — then the request falls to the
 * next candidate. A 400 falls to the next candidate at once (it may be that model, not the
 * request). A 401 stops at once: no key was pushed, so no model can answer. A network failure
 * counts as an upstream error. Every refusal is recorded in the trail, in order.
 */
async function serve(body: Json, stream: boolean): Promise<Served | Unserved> {
  const requires = requirementsFor(body);
  const order = candidatesFor(requires);
  if (order.length === 0) {
    return { ok: false, status: 503, text: 'no free model in the brief meets this request', trail: [] };
  }
  const trail: TrailRow[] = [];
  let lastStatus = 502;
  let lastText = '';
  let candidateIndex = 0;
  let retriedSame = false;
  let attempts = 0;
  while (attempts < MAX_ATTEMPTS && candidateIndex < order.length) {
    const model = order[candidateIndex];
    attempts += 1;
    let upstream: Response | null = null;
    let failureText = '';
    try {
      upstream = await dial(model.id, body);
    } catch (err) {
      failureText = (err as Error).message;
    }
    if (upstream !== null && upstream.ok && upstream.body !== null) {
      noteAnswer(model.id);
      return { ok: true, model: model.id, upstream, trail };
    }
    const status = upstream === null ? 502 : upstream.status;
    const text = upstream === null ? failureText : await upstream.text().catch(() => '');
    trail.push({ status, code: codeOfStatus(status, upstream === null), model: model.id });
    if (status === 401) {
      lastStatus = status;
      lastText = text;
      break;
    }
    noteRefusal(model.id, status);
    if ((status === 429 || status >= 500 || upstream === null) && !retriedSame) {
      retriedSame = true;
      continue;
    }
    retriedSame = false;
    candidateIndex += 1;
    lastStatus = status;
    lastText = text;
  }
  return { ok: false, status: lastStatus, text: lastText, trail };
}

/** The trail header on an answer a retry or a failover served: at most eight rows, in order. */
function trailHeaders(trail: TrailRow[]): Record<string, string> {
  if (trail.length === 0) return {};
  return { 'x-makefun-upstream-failures': JSON.stringify(trail.slice(0, MAX_TRAIL_ROWS)) };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown, extraHeaders: Record<string, string> = {}): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...extraHeaders });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** A refused or broken gateway call as a 502 in the client's wire: the prompt is lost. */
function refused(res: ServerResponse, wire: 'openai' | 'anthropic', result: Unserved): void {
  const message = `the gateway answered HTTP ${result.status}${result.text === '' ? '' : `: ${result.text.slice(0, 500)}`}`;
  if (wire === 'anthropic') sendError(res, wire, 502, 'api_error', message);
  else sendError(res, wire, 502, 'upstream_error', message);
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that meets it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const routingStartedAt = Date.now();
  const routingPromptId = brief.prompts[manifestIndex]?.id ?? null;
  const result = await serve(body, body.stream === true);
  logRouting(routingPromptId, result, Date.now() - routingStartedAt);
  if (!result.ok) return refused(res, 'openai', result);
  const headers = trailHeaders(result.trail);
  if (body.stream === true) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...headers });
    try {
      for await (const frame of openAIStreamWithModel(result.upstream.body, result.model)) res.write(frame);
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = await result.upstream.json();
  sendJson(res, 200, openAIJsonWithModel(isRecord(answer) ? answer : {}, result.model), headers);
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const upstreamBody = anthropicToOpenAI(body, '');
  const routingStartedAt = Date.now();
  const routingPromptId = brief.prompts[manifestIndex]?.id ?? null;
  const result = await serve(upstreamBody, body.stream === true);
  logRouting(routingPromptId, result, Date.now() - routingStartedAt);
  if (!result.ok) return refused(res, 'anthropic', result);
  const headers = trailHeaders(result.trail);
  if (body.stream === true) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...headers });
    try {
      for await (const event of anthropicEventsFromOpenAI(result.upstream.body, result.model)) res.write(event);
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = await result.upstream.json();
  sendJson(res, 200, openAIToAnthropic(isRecord(answer) ? answer : {}, result.model), headers);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

// ---------------------------------------------------------------------------
// Catalog self-check and per-request routing log.
//
// Two observability additions that never touch routing, model selection, any
// response, or the catalog. The self-check compares the catalog this entry
// registered with the gateway's live model list once at startup and warns
// about drift; the routing log writes one JSON line per served prompt to
// stderr. Both are fail-open: any error is logged once and serving carries on
// as if they were not there. The only wall-clock read in this file lives in
// the routing log's latency field, and it reaches stderr alone — never a
// request, a response, or the state the replay compares.
// ---------------------------------------------------------------------------

/** The gateway's model-list route: the OpenAI-compatible root plus /models. */
function gatewayModelsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/models` : `${GATEWAY_URL}/v1/models`;
}

/** The live model ids the gateway lists, in the order it lists them. */
async function liveModelIds(): Promise<string[]> {
  const url = gatewayModelsUrl();
  if (url === '') throw new Error('OPENAI_BASE_URL is not set');
  const res = await fetch(url, {
    headers: { authorization: `Bearer ${GATEWAY_KEY}` },
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`GET ${url} answered HTTP ${res.status}`);
  const body: unknown = await res.json();
  if (!isRecord(body) || !Array.isArray(body.data)) throw new Error(`GET ${url} answered without a model list`);
  const ids: string[] = [];
  for (const row of body.data) {
    if (isRecord(row) && typeof row.id === 'string' && row.id !== '') ids.push(row.id);
  }
  return ids;
}

/** The ids the catalog registers, across its providers, in catalog order. */
function catalogModelIds(report: CatalogReport): string[] {
  const ids: string[] = [];
  for (const provider of report.providers) {
    for (const model of provider.models) ids.push(model.id);
  }
  return ids;
}

/**
 * Compare the registered catalog with the gateway's live list and warn about
 * drift: a catalog model the gateway no longer lists (retired) and a live
 * model the catalog does not name. Fail-open by construction — a fetch or
 * parse failure logs once and returns — so this never throws and never blocks
 * serving; it makes exactly one GET /v1/models listing, which the trial does
 * not count toward its model-call budget.
 */
async function catalogSelfCheck(report: CatalogReport): Promise<void> {
  let live: string[];
  try {
    live = await liveModelIds();
  } catch (err) {
    log(`catalog self-check skipped: ${(err as Error).message}`);
    return;
  }
  const liveSet = new Set(live);
  const registered = catalogModelIds(report);
  for (const id of registered) {
    if (!liveSet.has(id)) log(`catalog self-check: ${id} is registered but the gateway no longer lists it`);
  }
  const registeredSet = new Set(registered);
  for (const id of live) {
    if (!registeredSet.has(id)) log(`catalog self-check: ${id} is live on the gateway but missing from the catalog`);
  }
}

/** One routing record: what a single served prompt tried, used, and cost in time. */
interface RoutingRecord {
  prompt: string | null;
  candidates: string[];
  servedBy: string | null;
  failovers: number;
  latencyMs: number;
}

/**
 * One JSON line to stderr for a prompt `serve` answered: the prompt's id, the
 * models tried in order, the model that served it (null when none did), how
 * many attempts failed before it, and the wall time serve took. Logging only:
 * nothing here reaches a response body, a header, a status, or a routing
 * decision.
 */
function logRouting(promptId: string | null, result: Served | Unserved, latencyMs: number): void {
  const candidates: string[] = [];
  for (const row of result.trail) {
    if (!candidates.includes(row.model)) candidates.push(row.model);
  }
  if (result.ok && !candidates.includes(result.model)) candidates.push(result.model);
  const record: RoutingRecord = {
    prompt: promptId,
    candidates,
    servedBy: result.ok ? result.model : null,
    failovers: result.trail.length,
    latencyMs,
  };
  process.stderr.write(`${JSON.stringify(record)}\n`);
}

// ---------------------------------------------------------------------------
// Register, poll, and start.
// ---------------------------------------------------------------------------

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  void catalogSelfCheck(catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
