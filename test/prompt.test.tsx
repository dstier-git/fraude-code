import assert from 'node:assert/strict';
import test from 'node:test';
import React from 'react';
import {render} from 'ink-testing-library';
import {Prompt} from '../src/prompt.js';

test('edits and submits a single-line prompt', async () => {
  const submissions: string[] = [];
  const view = render(
    <Prompt
      width={60}
      disabled={false}
      onSubmit={(value) => submissions.push(value)}
      onCancel={() => undefined}
      onExit={() => undefined}
    />,
  );

  view.stdin.write('helo');
  view.stdin.write('\u001B[D');
  view.stdin.write('l');
  view.stdin.write('\r');
  await settle();

  assert.deepEqual(submissions, ['hello']);
  view.unmount();
});

test('normalizes pasted newlines and handles Escape while disabled', async () => {
  let cancelled = 0;
  const submissions: string[] = [];
  const view = render(
    <Prompt
      width={60}
      disabled={false}
      onSubmit={(value) => submissions.push(value)}
      onCancel={() => cancelled++}
      onExit={() => undefined}
    />,
  );

  view.stdin.write('\u001B[200~first\nsecond\u001B[201~');
  view.stdin.write('\r');
  await settle();
  assert.deepEqual(submissions, ['first second']);

  view.rerender(
    <Prompt
      width={60}
      disabled
      onSubmit={(value) => submissions.push(value)}
      onCancel={() => cancelled++}
      onExit={() => undefined}
    />,
  );
  view.stdin.write('\u001B');
  await settle();
  assert.equal(cancelled, 1);
  view.unmount();
});

function settle(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 20));
}
