export type ChatRole = 'user' | 'assistant';

export type ChatMessage = {
  role: ChatRole;
  content: string;
};

export type ModelInfo = {
  id: string;
  contextWindow: number;
};

export type TokenUsage = {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
};

export type StreamEvent =
  | {type: 'text_delta'; text: string}
  | ({type: 'usage'} & TokenUsage)
  | {type: 'completed'}
  | {type: 'error'; message: string};

export type ChatRequest = {
  model: string;
  messages: ChatMessage[];
};

export interface ChatBackend {
  listModels(signal?: AbortSignal): Promise<ModelInfo[]>;
  streamChat(request: ChatRequest, signal: AbortSignal): AsyncIterable<StreamEvent>;
}
