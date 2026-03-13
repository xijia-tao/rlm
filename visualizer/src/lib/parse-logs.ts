import { RLMIteration, RLMLogFile, LogMetadata, RLMConfigMetadata, ContextPreview, extractFinalAnswer } from './types';

// Extract the context variable from code block locals
export function extractContextVariable(iterations: RLMIteration[]): string | null {
  for (const iter of iterations) {
    for (const block of iter.code_blocks) {
      if (block.result?.locals?.context) {
        const ctx = block.result.locals.context;
        if (typeof ctx === 'string') {
          return ctx;
        }
      }
    }
  }
  return null;
}

// Default config when metadata is not present (backwards compatibility)
function getDefaultConfig(): RLMConfigMetadata {
  return {
    root_model: null,
    max_depth: null,
    max_iterations: null,
    backend: null,
    backend_kwargs: null,
    environment_type: null,
    environment_kwargs: null,
    other_backends: null,
  };
}

export interface ParsedJSONL {
  iterations: RLMIteration[];
  config: RLMConfigMetadata;
  contextPreview: ContextPreview | null;
}

export function parseJSONL(content: string): ParsedJSONL {
  const lines = content.trim().split('\n').filter(line => line.trim());
  const iterations: RLMIteration[] = [];
  let config: RLMConfigMetadata = getDefaultConfig();
  let contextPreview: ContextPreview | null = null;
  
  for (const line of lines) {
    try {
      const parsed = JSON.parse(line);
      
      if (parsed.type === 'metadata') {
        config = {
          root_model: parsed.root_model ?? null,
          max_depth: parsed.max_depth ?? null,
          max_iterations: parsed.max_iterations ?? null,
          backend: parsed.backend ?? null,
          backend_kwargs: parsed.backend_kwargs ?? null,
          environment_type: parsed.environment_type ?? null,
          environment_kwargs: parsed.environment_kwargs ?? null,
          other_backends: parsed.other_backends ?? null,
        };
      } else if (parsed.type === 'context') {
        if (parsed.context_type === 'image') {
          contextPreview = {
            context_type: 'image',
            image_data: parsed.image_data ?? '',
            source_path: parsed.source_path ?? null,
          };
        } else if (parsed.context_type === 'video') {
          contextPreview = {
            context_type: 'video',
            frames: parsed.frames ?? [],
            source_path: parsed.source_path ?? null,
            duration_sec: parsed.duration_sec ?? 0,
            fps: parsed.fps ?? 0,
            resolution: parsed.resolution ?? [0, 0],
          };
        }
      } else {
        // This is an iteration entry
        iterations.push(parsed as RLMIteration);
      }
    } catch (e) {
      console.error('Failed to parse line:', line, e);
    }
  }
  
  return { iterations, config, contextPreview };
}

export function extractContextQuestion(iterations: RLMIteration[]): string {
  if (iterations.length === 0) return 'No context found';
  
  const firstIteration = iterations[0];
  const prompt = firstIteration.prompt;
  
  // Look for user message that contains the actual question
  for (const msg of prompt) {
    if (msg.role === 'user' && msg.content) {
      // Try to extract quoted query
      const queryMatch = msg.content.match(/original query: "([^"]+)"/);
      if (queryMatch) {
        return queryMatch[1];
      }
      
      // Check if it contains the actual query pattern
      if (msg.content.includes('answer the prompt')) {
        continue;
      }
      
      // Take first substantial user message
      if (msg.content.length > 50 && msg.content.length < 500) {
        return msg.content.slice(0, 200) + (msg.content.length > 200 ? '...' : '');
      }
    }
  }
  
  // Fallback: look in system prompt for context info
  const systemMsg = prompt.find(m => m.role === 'system');
  if (systemMsg?.content) {
    const contextMatch = systemMsg.content.match(/context variable.*?:(.*?)(?:\n|$)/i);
    if (contextMatch) {
      return contextMatch[1].trim().slice(0, 200);
    }
  }
  
  // Check code block output for actual context
  for (const iter of iterations) {
    for (const block of iter.code_blocks) {
      if (block.result?.locals?.context) {
        const ctx = block.result.locals.context;
        if (typeof ctx === 'string' && ctx.length < 500) {
          return ctx;
        }
      }
    }
  }
  
  return 'Context available in REPL environment';
}

export function computeMetadata(iterations: RLMIteration[]): LogMetadata {
  let totalCodeBlocks = 0;
  let totalSubLMCalls = 0;
  let totalExecutionTime = 0;
  let hasErrors = false;
  let finalAnswer: string | null = null;
  
  for (const iter of iterations) {
    totalCodeBlocks += iter.code_blocks.length;
    
    // Use iteration_time if available, otherwise sum code block times
    if (iter.iteration_time != null) {
      totalExecutionTime += iter.iteration_time;
    } else {
      for (const block of iter.code_blocks) {
        if (block.result) {
          totalExecutionTime += block.result.execution_time || 0;
        }
      }
    }
    
    for (const block of iter.code_blocks) {
      if (block.result) {
        if (block.result.stderr) {
          hasErrors = true;
        }
        if (block.result.rlm_calls) {
          totalSubLMCalls += block.result.rlm_calls.length;
        }
      }
    }
    
    if (iter.final_answer) {
      finalAnswer = extractFinalAnswer(iter.final_answer);
    }
  }
  
  return {
    totalIterations: iterations.length,
    totalCodeBlocks,
    totalSubLMCalls,
    contextQuestion: extractContextQuestion(iterations),
    finalAnswer,
    totalExecutionTime,
    hasErrors,
  };
}

export function parseLogFile(fileName: string, content: string): RLMLogFile {
  const { iterations, config, contextPreview } = parseJSONL(content);
  const metadata = computeMetadata(iterations);
  
  return {
    fileName,
    filePath: fileName,
    iterations,
    metadata,
    config,
    contextPreview,
  };
}

