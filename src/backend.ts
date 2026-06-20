import type {
  ChatBackend,
  ChatRequest,
  ModelInfo,
  StreamEvent,
  TokenUsage,
} from './types.js';

type JsonObject = Record<string, unknown>;

export class BackendError extends Error {
  override name = 'BackendError';
}

export class OpenAIBackend implements ChatBackend {
  readonly #baseUrl: string;
  readonly #apiKey: string | undefined;
  readonly #fallbackContextWindow: number;

  constructor(
    baseUrl: string,
    apiKey?: string,
    fallbackContextWindow = 32_768,
  ) {
    this.#baseUrl = baseUrl.replace(/\/+$/, '');
    this.#apiKey = apiKey;
    this.#fallbackContextWindow = fallbackContextWindow;
  }

  async listModels(signal?: AbortSignal): Promise<ModelInfo[]> {
    const response = await fetch(`${this.#baseUrl}/v1/models`, {
      headers: this.#headers(),
      ...(signal ? {signal} : {}),
    });

    if (!response.ok) {
      throw new BackendError(await errorMessage(response));
    }

    const payload: unknown = await response.json();
    if (!isObject(payload) || !Array.isArray(payload.data)) {
      throw new BackendError('Backend returned an invalid model list.');
    }

    const models = payload.data.flatMap((item): ModelInfo[] => {
      if (!isObject(item) || typeof item.id !== 'string') return [];
      const contextWindow = numericField(
        item,
        'context_window',
        'context_window_tokens',
      );
      return [{
        id: item.id,
        contextWindow:
          contextWindow !== undefined && contextWindow > 0
            ? contextWindow
            : this.#fallbackContextWindow,
      }];
    });

    if (models.length === 0) {
      throw new BackendError(
        'Backend returned no usable models.',
      );
    }

    return models;
  }

  async *streamChat(
    request: ChatRequest,
    signal: AbortSignal,
  ): AsyncIterable<StreamEvent> {
    const response = await fetch(`${this.#baseUrl}/v1/chat/completions`, {
      method: 'POST',
      headers: {
        ...this.#headers(),
        'content-type': 'application/json',
      },
      body: JSON.stringify({
        model: request.model,
        messages: request.messages,
        stream: true,
        stream_options: {include_usage: true},
      }),
      signal,
    });

    if (!response.ok) {
      throw new BackendError(await errorMessage(response));
    }
    if (!response.body) {
      throw new BackendError('Backend returned an empty streaming response.');
    }

    let completed = false;
    for await (const data of parseServerSentEvents(response.body)) {
      if (data === '[DONE]') {
        completed = true;
        yield {type: 'completed'};
        break;
      }

      let payload: unknown;
      try {
        payload = JSON.parse(data);
      } catch {
        yield {type: 'error', message: 'Backend sent malformed streaming JSON.'};
        return;
      }

      if (!isObject(payload)) continue;
      const backendMessage = readError(payload);
      if (backendMessage) {
        yield {type: 'error', message: backendMessage};
        return;
      }

      const delta = readTextDelta(payload);
      if (delta) yield {type: 'text_delta', text: delta};

      const usage = readUsage(payload);
      if (usage) yield {type: 'usage', ...usage};
    }

    if (!completed) yield {type: 'completed'};
  }

  #headers(): Record<string, string> {
    return this.#apiKey
      ? {authorization: `Bearer ${this.#apiKey}`}
      : {};
  }
}

export async function* parseServerSentEvents(
  stream: ReadableStream<Uint8Array>,
): AsyncIterable<string> {
  const decoder = new TextDecoder();
  let buffer = '';
  let dataLines: string[] = [];

  const consumeLine = (line: string): string | undefined => {
    if (line === '') {
      if (dataLines.length === 0) return undefined;
      const event = dataLines.join('\n');
      dataLines = [];
      return event;
    }
    if (line.startsWith('data:')) {
      dataLines.push(line.slice(5).replace(/^ /, ''));
    }
    return undefined;
  };

  for await (const chunk of stream) {
    buffer += decoder.decode(chunk, {stream: true});
    const lines = buffer.split(/\r?\n/);
    buffer = lines.pop() ?? '';
    for (const line of lines) {
      const event = consumeLine(line);
      if (event !== undefined) yield event;
    }
  }

  buffer += decoder.decode();
  if (buffer) {
    const event = consumeLine(buffer.replace(/\r$/, ''));
    if (event !== undefined) yield event;
  }
  if (dataLines.length > 0) yield dataLines.join('\n');
}

function readTextDelta(payload: JsonObject): string | undefined {
  const choices = payload.choices;
  if (!Array.isArray(choices)) return undefined;
  const first = choices[0];
  if (!isObject(first) || !isObject(first.delta)) return undefined;
  return typeof first.delta.content === 'string'
    ? first.delta.content
    : undefined;
}

function readUsage(payload: JsonObject): TokenUsage | undefined {
  if (!isObject(payload.usage)) return undefined;
  const promptTokens = numericField(payload.usage, 'prompt_tokens');
  const completionTokens = numericField(payload.usage, 'completion_tokens');
  const totalTokens = numericField(payload.usage, 'total_tokens');
  if (
    promptTokens === undefined ||
    completionTokens === undefined ||
    totalTokens === undefined
  ) {
    return undefined;
  }
  return {promptTokens, completionTokens, totalTokens};
}

function readError(payload: JsonObject): string | undefined {
  if (!isObject(payload.error)) return undefined;
  return typeof payload.error.message === 'string'
    ? payload.error.message
    : 'Backend returned an unspecified error.';
}

function numericField(
  value: JsonObject,
  ...keys: string[]
): number | undefined {
  for (const key of keys) {
    const item = value[key];
    if (typeof item === 'number' && Number.isFinite(item)) return item;
  }
  return undefined;
}

function isObject(value: unknown): value is JsonObject {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

async function errorMessage(response: Response): Promise<string> {
  try {
    const payload: unknown = await response.json();
    if (isObject(payload)) {
      return readError(payload) ?? `Backend request failed (${response.status}).`;
    }
  } catch {
    // Fall through to the status-only message.
  }
  return `Backend request failed (${response.status}).`;
}
