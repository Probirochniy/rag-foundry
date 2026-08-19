export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  isStreaming?: boolean;
  wasReset?: boolean;
}

export interface IngestDocument {
  id: string;
  content: string;
  source_id: string;
}
