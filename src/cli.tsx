#!/usr/bin/env node
import React from 'react';
import {render} from 'ink';
import {App} from './app.js';
import {OpenAIBackend} from './backend.js';

if (!process.stdin.isTTY || !process.stdout.isTTY) {
  process.stderr.write('Fraude Code requires an interactive TTY.\n');
  process.exitCode = 1;
} else {
  const baseUrl = process.env.FRAUDE_API_BASE_URL ?? 'http://127.0.0.1:8000';
  const apiKey = process.env.FRAUDE_API_KEY;
  const contextWindow = parseContextWindow(process.env.FRAUDE_CONTEXT_WINDOW);
  if (contextWindow === undefined) {
    process.stderr.write(
      'FRAUDE_CONTEXT_WINDOW must be a positive integer when set.\n',
    );
    process.exitCode = 1;
  } else {
    const backend = new OpenAIBackend(baseUrl, apiKey, contextWindow);
    render(
      <App backend={backend} preferredModel={process.env.FRAUDE_MODEL} />,
      {exitOnCtrlC: false},
    );
  }
}

function parseContextWindow(value: string | undefined): number | undefined {
  if (value === undefined) return 32_768;
  if (!/^\d+$/.test(value)) return undefined;
  const parsed = Number(value);
  return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : undefined;
}
