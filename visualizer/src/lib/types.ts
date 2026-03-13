// Types matching the RLM log format

// Serialized PIL image stored in REPL locals
export interface SerializedPILImage {
  __type__: 'pil_image';
  data: string;       // base64-encoded JPEG
  format: string;     // 'jpeg'
  mode: string;       // e.g. 'RGB'
  size: [number, number]; // [original_width, original_height]
}

export function isSerializedPILImage(value: unknown): value is SerializedPILImage {
  return (
    typeof value === 'object' &&
    value !== null &&
    (value as Record<string, unknown>).__type__ === 'pil_image' &&
    typeof (value as Record<string, unknown>).data === 'string'
  );
}

// Context preview (image or video) stored at the top of the JSONL
export interface ImageContextPreview {
  context_type: 'image';
  image_data: string;   // base64-encoded JPEG
  source_path: string | null;
}

export interface VideoFramePreview {
  timestamp: number;
  data: string; // base64-encoded JPEG
}

export interface VideoContextPreview {
  context_type: 'video';
  frames: VideoFramePreview[];
  source_path: string | null;
  duration_sec: number;
  fps: number;
  resolution: [number, number];
}

export type ContextPreview = ImageContextPreview | VideoContextPreview;

export interface RLMChatCompletion {
  prompt: string | Record<string, unknown>;
  response: string;
  prompt_tokens: number;
  completion_tokens: number;
  execution_time: number;
}

export interface REPLResult {
  stdout: string;
  stderr: string;
  locals: Record<string, unknown>;
  execution_time: number;
  rlm_calls: RLMChatCompletion[];
}

export interface CodeBlock {
  code: string;
  result: REPLResult;
}

export interface RLMIteration {
  type?: string;
  iteration: number;
  timestamp: string;
  prompt: Array<{ role: string; content: string }>;
  response: string;
  code_blocks: CodeBlock[];
  final_answer: string | [string, string] | null;
  iteration_time: number | null;
}

// Metadata saved at the start of a log file about RLM configuration
export interface RLMConfigMetadata {
  root_model: string | null;
  max_depth: number | null;
  max_iterations: number | null;
  backend: string | null;
  backend_kwargs: Record<string, unknown> | null;
  environment_type: string | null;
  environment_kwargs: Record<string, unknown> | null;
  other_backends: string[] | null;
}

export interface RLMLogFile {
  fileName: string;
  filePath: string;
  iterations: RLMIteration[];
  metadata: LogMetadata;
  config: RLMConfigMetadata;
  contextPreview: ContextPreview | null;
}

export interface LogMetadata {
  totalIterations: number;
  totalCodeBlocks: number;
  totalSubLMCalls: number;
  contextQuestion: string;
  finalAnswer: string | null;
  totalExecutionTime: number;
  hasErrors: boolean;
}

export function extractFinalAnswer(answer: string | [string, string] | null): string | null {
  if (!answer) return null;
  if (Array.isArray(answer)) {
    return answer[1];
  }
  return answer;
}

