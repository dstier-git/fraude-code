import assert from 'node:assert/strict';
import test from 'node:test';
import {OpenAIBackend, parseServerSentEvents} from '../src/backend.js';

test('parses fragmented and multiline SSE records', async () => {
  const encoder = new TextEncoder();
  const chunks = [
    'data: {"choices":[{"delta":',
    '{"content":"hello"}}]}\r\n\r\n',
    ': keepalive\n',
    'data: first\ndata: second\n\n',
    'data: [DONE]\n\n',
  ];
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });

  const events: string[] = [];
  for await (const event of parseServerSentEvents(stream)) events.push(event);

  assert.deepEqual(events, [
    '{"choices":[{"delta":{"content":"hello"}}]}',
    'first\nsecond',
    '[DONE]',
  ]);
});

test('flushes an SSE record without a trailing blank line', async () => {
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(new TextEncoder().encode('data: final'));
      controller.close();
    },
  });

  const events: string[] = [];
  for await (const event of parseServerSentEvents(stream)) events.push(event);
  assert.deepEqual(events, ['final']);
});

test('maps the OpenAI-compatible HTTP contract to frontend events', async () => {
  const originalFetch = globalThis.fetch;
  const requests: Array<{url: string; init?: RequestInit}> = [];
  globalThis.fetch = async (input, init) => {
    const url = String(input);
    requests.push(init ? {url, init} : {url});
    if (url.endsWith('/v1/models')) {
      return Response.json({
        data: [{id: 'test-model', context_window: 4096}],
      });
    }
    const body = [
      'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n',
      'data: {"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}\n\n',
      'data: [DONE]\n\n',
    ].join('');
    return new Response(body, {
      headers: {'content-type': 'text/event-stream'},
    });
  };

  try {
    const backend = new OpenAIBackend('http://backend.test/', 'secret');
    assert.deepEqual(await backend.listModels(), [
      {id: 'test-model', contextWindow: 4096},
    ]);

    const events = [];
    for await (const event of backend.streamChat(
      {
        model: 'test-model',
        messages: [{role: 'user', content: 'hi'}],
      },
      new AbortController().signal,
    )) {
      events.push(event);
    }

    assert.deepEqual(events, [
      {type: 'text_delta', text: 'hello'},
      {
        type: 'usage',
        promptTokens: 2,
        completionTokens: 1,
        totalTokens: 3,
      },
      {type: 'completed'},
    ]);
    assert.equal(requests[1]?.init?.method, 'POST');
    assert.deepEqual(JSON.parse(String(requests[1]?.init?.body)), {
      model: 'test-model',
      messages: [{role: 'user', content: 'hi'}],
      stream: true,
      stream_options: {include_usage: true},
    });
    assert.equal(
      new Headers(requests[1]?.init?.headers).get('authorization'),
      'Bearer secret',
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('uses the configured context fallback for stock MLX model metadata', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => Response.json({
    data: [
      {
        id: 'mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit',
        object: 'model',
        created: 1_750_000_000,
      },
    ],
  });

  try {
    const backend = new OpenAIBackend(
      'http://127.0.0.1:8080',
      undefined,
      32_768,
    );
    assert.deepEqual(await backend.listModels(), [
      {
        id: 'mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit',
        contextWindow: 32_768,
      },
    ]);
  } finally {
    globalThis.fetch = originalFetch;
  }
});
