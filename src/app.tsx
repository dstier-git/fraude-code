import React, {useEffect, useRef, useState} from 'react';
import {
  Box,
  Static,
  Text,
  useApp,
  useWindowSize,
} from 'ink';
import {Prompt} from './prompt.js';
import type {
  ChatBackend,
  ChatMessage,
  ModelInfo,
  TokenUsage,
} from './types.js';

const BANNER = `███████╗██████╗  █████╗ ██╗   ██╗██████╗ ███████╗
██╔════╝██╔══██╗██╔══██╗██║   ██║██╔══██╗██╔════╝
█████╗  ██████╔╝███████║▓▓║   ██║██║  ██║█████╗
██╔══╝  ██╔══██╗██╔══██║██║   ██║██║  ██║██╔══╝
██║     ██║  ██║██║  ██║╚██████╔╝██████╔╝███████╗
╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝`;

const CODE_BANNER = ` ██████╗ ██████╗ ██████╗ ███████╗
██╔════╝██╔═══██╗██╔══██╗██╔════╝
██║     ██║   ██║██║  ██║█████╗
██║     ██║   ██║██║  ██║██╔══╝
╚██████╗╚██████╔╝██████╔╝███████╗
 ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝`;

type FeedItem =
  | {id: string; kind: 'cover'; model: ModelInfo}
  | {id: string; kind: 'user'; text: string}
  | {id: string; kind: 'assistant'; text: string; interrupted?: boolean}
  | {id: string; kind: 'error'; text: string};

type ActiveTurn = {
  text: string;
  startedAt: number;
};

export type AppProps = {
  backend: ChatBackend;
  preferredModel?: string | undefined;
  cwd?: string;
  columns?: number;
};

export function App({
  backend,
  preferredModel,
  cwd = process.cwd(),
  columns: columnsOverride,
}: AppProps) {
  const {exit} = useApp();
  const windowSize = useWindowSize();
  const columns = columnsOverride ?? windowSize.columns;
  const width = Math.min(88, Math.max(1, columns));
  const [model, setModel] = useState<ModelInfo>();
  const [startupError, setStartupError] = useState<string>();
  const [feed, setFeed] = useState<FeedItem[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [active, setActive] = useState<ActiveTurn>();
  const [contextLeft, setContextLeft] = useState(100);
  const abortRef = useRef<AbortController | undefined>(undefined);
  const nextId = useRef(0);

  const id = (prefix: string) => `${prefix}-${nextId.current++}`;

  useEffect(() => {
    const controller = new AbortController();
    void backend.listModels(controller.signal).then((models) => {
      const selected = preferredModel
        ? models.find((candidate) => candidate.id === preferredModel)
        : models[0];
      if (!selected) {
        throw new Error(
          preferredModel
            ? `Configured model “${preferredModel}” is unavailable.`
            : 'Backend returned no usable models.',
        );
      }
      setModel(selected);
      setFeed([{id: 'cover', kind: 'cover', model: selected}]);
    }).catch((error: unknown) => {
      if (!controller.signal.aborted) setStartupError(messageOf(error));
    });
    return () => controller.abort();
  }, [backend, preferredModel]);

  useEffect(() => () => abortRef.current?.abort(), []);

  const submit = (prompt: string) => {
    if (!model || active) return;
    const requestMessages: ChatMessage[] = [
      ...messages,
      {role: 'user', content: prompt},
    ];
    setMessages(requestMessages);
    setFeed((items) => [...items, {id: id('user'), kind: 'user', text: prompt}]);
    const controller = new AbortController();
    abortRef.current = controller;
    const startedAt = Date.now();
    setActive({text: '', startedAt});

    void consumeStream({
      backend,
      model,
      messages: requestMessages,
      signal: controller.signal,
      onText(text) {
        setActive((turn) => turn ? {...turn, text} : turn);
      },
      onUsage(usage) {
        setContextLeft(contextPercentage(model.contextWindow, usage));
      },
    }).then(({text, usage}) => {
      if (usage) setContextLeft(contextPercentage(model.contextWindow, usage));
      if (text) {
        setMessages((items) => [...items, {role: 'assistant', content: text}]);
        setFeed((items) => [
          ...items,
          {id: id('assistant'), kind: 'assistant', text},
        ]);
      }
    }).catch((error: unknown) => {
      const partial = activeText(error);
      if (controller.signal.aborted) {
        if (partial) {
          setMessages((items) => [...items, {role: 'assistant', content: partial}]);
          setFeed((items) => [
            ...items,
            {id: id('assistant'), kind: 'assistant', text: partial, interrupted: true},
          ]);
        } else {
          setFeed((items) => [
            ...items,
            {id: id('error'), kind: 'error', text: 'Generation interrupted.'},
          ]);
        }
      } else {
        setFeed((items) => [
          ...items,
          {id: id('error'), kind: 'error', text: messageOf(error)},
        ]);
      }
    }).finally(() => {
      if (abortRef.current === controller) abortRef.current = undefined;
      setActive(undefined);
    });
  };

  const cancel = () => abortRef.current?.abort();
  const quit = () => {
    abortRef.current?.abort();
    exit();
  };

  if (columns < 60) {
    return (
      <Box flexDirection="column">
        <Text color="red">Fraude Code needs at least 60 columns.</Text>
        <Text dimColor>Current terminal width: {columns}</Text>
      </Box>
    );
  }

  return (
    <Box flexDirection="column" width={width}>
      <Static items={feed}>
        {(item) => <FeedItemView key={item.id} item={item} cwd={cwd} width={width} />}
      </Static>

      {startupError ? (
        <Box marginTop={1} flexDirection="column">
          <Text color="red">✗ {startupError}</Text>
          <Text dimColor>Check FRAUDE_API_BASE_URL, then restart.</Text>
        </Box>
      ) : null}

      {active ? <ActiveResponse turn={active} width={width} /> : null}

      <Box marginTop={1}>
        <Prompt
          width={width}
          disabled={!model || Boolean(active) || Boolean(startupError)}
          disabledMessage={
            active
              ? 'esc to interrupt'
              : startupError
                ? 'backend unavailable'
                : 'connecting to backend…'
          }
          onSubmit={submit}
          onCancel={cancel}
          onExit={quit}
        />
      </Box>
      <Box justifyContent="space-between" paddingX={2} width={width}>
        <Text dimColor>? for shortcuts</Text>
        <Text dimColor>context left: {contextLeft}%</Text>
      </Box>
    </Box>
  );
}

function FeedItemView({
  item,
  cwd,
  width,
}: {
  item: FeedItem;
  cwd: string;
  width: number;
}) {
  if (item.kind === 'cover') {
    return <Cover model={item.model} cwd={cwd} width={width} />;
  }
  if (item.kind === 'user') {
    return (
      <Box marginTop={1}>
        <Text color="blue">{'>'} </Text><Text wrap="wrap">{item.text}</Text>
      </Box>
    );
  }
  if (item.kind === 'error') {
    return <Box marginTop={1}><Text color="red">✗ {item.text}</Text></Box>;
  }
  return (
    <Box marginTop={1} flexDirection="column">
      <Text wrap="wrap">● {item.text}</Text>
      {item.interrupted ? <Text dimColor>  interrupted</Text> : null}
    </Box>
  );
}

function Cover({model, cwd, width}: {model: ModelInfo; cwd: string; width: number}) {
  return (
    <Box flexDirection="column" width={width}>
      <Text color="white">{BANNER}</Text>
      <Text dimColor>{CODE_BANNER}</Text>
      <Box marginTop={1} flexDirection="column">
        <Text><Text color="red">✻</Text> Welcome to Fraude Code!</Text>
        <Text> </Text>
        <Text dimColor>  v0.0.0-fraudulent</Text>
        <Text dimColor>  the first AI coding harness designed to</Text>
        <Text dimColor>  maximize hallucination</Text>
        <Text> </Text>
        <Text wrap="truncate"><Text dimColor>  cwd:   </Text><Text color="blue">{cwd}</Text></Text>
        <Text wrap="truncate"><Text dimColor>  model: </Text>{model.id}</Text>
      </Box>
    </Box>
  );
}

function ActiveResponse({turn, width}: {turn: ActiveTurn; width: number}) {
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const timer = setInterval(() => setTick((value) => value + 1), 100);
    return () => clearInterval(timer);
  }, []);
  const frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
  const elapsed = ((Date.now() - turn.startedAt) / 1000).toFixed(1);
  return (
    <Box marginTop={1} flexDirection="column" width={width}>
      {turn.text ? <Text wrap="wrap">● {turn.text}</Text> : null}
      <Text dimColor>{frames[tick % frames.length]} generating · — tokens · {elapsed}s</Text>
    </Box>
  );
}

async function consumeStream({
  backend,
  model,
  messages,
  signal,
  onText,
  onUsage,
}: {
  backend: ChatBackend;
  model: ModelInfo;
  messages: ChatMessage[];
  signal: AbortSignal;
  onText: (text: string) => void;
  onUsage: (usage: TokenUsage) => void;
}): Promise<{text: string; usage?: TokenUsage}> {
  let text = '';
  let usage: TokenUsage | undefined;
  try {
    for await (const event of backend.streamChat(
      {model: model.id, messages},
      signal,
    )) {
      if (event.type === 'text_delta') {
        text += event.text;
        onText(text);
      } else if (event.type === 'usage') {
        usage = event;
        onUsage(event);
      } else if (event.type === 'error') {
        throw new Error(event.message);
      }
    }
    return usage ? {text, usage} : {text};
  } catch (error) {
    throw new PartialResponseError(messageOf(error), text);
  }
}

class PartialResponseError extends Error {
  constructor(message: string, readonly partialText: string) {
    super(message);
    this.name = 'PartialResponseError';
  }
}

function activeText(error: unknown): string {
  return error instanceof PartialResponseError ? error.partialText : '';
}

function contextPercentage(contextWindow: number, usage: TokenUsage): number {
  const percentage = ((contextWindow - usage.totalTokens) / contextWindow) * 100;
  return Math.round(Math.min(100, Math.max(0, percentage)));
}

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : 'Unknown backend error.';
}
