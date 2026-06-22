import assert from 'node:assert/strict';
import test from 'node:test';
import React from 'react';
import {render} from 'ink-testing-library';
import {App} from '../src/app.js';
import type {
  ChatBackend,
  ChatRequest,
  ModelInfo,
  StreamEvent,
} from '../src/types.js';

class FakeBackend implements ChatBackend {
  readonly requests: ChatRequest[] = [];
  readonly models: ModelInfo[] = [{id: 'fraude-test-model', contextWindow: 1000}];
  responses: StreamEvent[][] = [];

  async listModels(): Promise<ModelInfo[]> {
    return this.models;
  }

  async *streamChat(request: ChatRequest): AsyncIterable<StreamEvent> {
    this.requests.push(request);
    for (const event of this.responses.shift() ?? []) {
      await settle();
      yield event;
    }
  }
}

test('renders the frame-one cover at supported widths', async () => {
  for (const columns of [88, 60]) {
    const view = render(
      <App backend={new FakeBackend()} columns={columns} cwd="/tmp/project" />,
    );
    await settle();
    const frame = view.lastFrame() ?? '';
    assert.match(frame, /Welcome to Fraude Code,/);
    assert.match(frame, /fraude-test-model/);
    assert.match(frame, /\/tmp\/project/);
    assert.match(frame, /context left: 100%/);
    view.unmount();
  }
});

test('shows a compact warning below 60 columns', () => {
  const view = render(<App backend={new FakeBackend()} columns={59} />);
  assert.match(view.lastFrame() ?? '', /needs at least 60 columns/);
  view.unmount();
});

test('streams a response and sends ordered conversation history', async () => {
  const backend = new FakeBackend();
  backend.responses = [
    [
      {type: 'text_delta', text: 'Confidently '},
      {type: 'text_delta', text: 'wrong.'},
      {
        type: 'usage',
        promptTokens: 100,
        completionTokens: 50,
        totalTokens: 150,
      },
      {type: 'completed'},
    ],
    [
      {type: 'text_delta', text: 'Still wrong.'},
      {type: 'completed'},
    ],
  ];
  const view = render(<App backend={backend} columns={88} />);
  await settle();

  view.stdin.write('first prompt');
  view.stdin.write('\r');
  await settle(400);
  assert.match(allFrames(view.frames), /Confidently wrong\./);
  assert.match(view.lastFrame() ?? '', /context left: 85%/);

  view.stdin.write('second prompt');
  view.stdin.write('\r');
  await settle(300);

  assert.equal(backend.requests.length, 2);
  assert.deepEqual(backend.requests[1]?.messages, [
    {role: 'user', content: 'first prompt'},
    {role: 'assistant', content: 'Confidently wrong.'},
    {role: 'user', content: 'second prompt'},
  ]);
  assert.equal((allFrames(view.frames).match(/Welcome to Fraude Code,/g) ?? []).length > 0, true);
  view.unmount();
});

test('preserves partial output when Escape cancels generation', async () => {
  const backend: ChatBackend = {
    async listModels() {
      return [{id: 'fraude-test-model', contextWindow: 1000}];
    },
    async *streamChat(_request, signal) {
      yield {type: 'text_delta', text: 'Partial answer'};
      await new Promise<void>((resolve, reject) => {
        signal.addEventListener('abort', () => reject(new Error('aborted')), {once: true});
      });
      yield {type: 'completed'};
    },
  };
  const view = render(<App backend={backend} columns={88} />);
  await settle();
  view.stdin.write('cancel me');
  view.stdin.write('\r');
  await settle(180);
  view.stdin.write('\u001B');
  await settle(180);

  const output = allFrames(view.frames);
  assert.match(output, /Partial answer/);
  assert.match(output, /interrupted/);
  view.unmount();
});

test('does not carry failed prompts into later model context', async () => {
  const backend = new FakeBackend();
  backend.responses = [
    [{type: 'error', message: 'malformed tool call'}],
    [
      {type: 'text_delta', text: 'Hello!'},
      {type: 'completed'},
    ],
  ];
  const view = render(<App backend={backend} columns={88} />);
  await settle();

  view.stdin.write('write a file');
  view.stdin.write('\r');
  await settle(180);
  view.stdin.write('hi');
  view.stdin.write('\r');
  await settle(180);

  assert.equal(backend.requests.length, 2);
  assert.deepEqual(backend.requests[1]?.messages, [
    {role: 'user', content: 'hi'},
  ]);
  view.unmount();
});

function allFrames(frames: string[]): string {
  return frames.join('\n');
}

function settle(milliseconds = 20): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}
