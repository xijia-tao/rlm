'use client';

import { useState } from 'react';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { cn } from '@/lib/utils';
import { RLMIteration, SerializedPILImage, extractFinalAnswer, isSerializedPILImage } from '@/lib/types';

interface TrajectoryPanelProps {
  iterations: RLMIteration[];
  selectedIteration: number;
  onSelectIteration: (index: number) => void;
}

// Helper to format message content for display
function formatMessageContent(content: string): string {
  if (content.length > 8000) {
    return content.slice(0, 8000) + '\n\n... [content truncated for display]';
  }
  return content;
}

// Role icon component
function RoleIcon({ role }: { role: string }) {
  if (role === 'system') {
    return (
      <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center shadow-lg shadow-violet-500/20">
        <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
        </svg>
      </div>
    );
  }
  if (role === 'user') {
    return (
      <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-emerald-500 to-green-600 flex items-center justify-center shadow-lg shadow-emerald-500/20">
        <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
        </svg>
      </div>
    );
  }
  // assistant
  return (
    <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-sky-500 to-blue-600 flex items-center justify-center shadow-lg shadow-sky-500/20">
      <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
      </svg>
    </div>
  );
}

// Role label
function RoleLabel({ role }: { role: string }) {
  const labels: Record<string, { name: string; color: string }> = {
    system: { name: 'System Prompt', color: 'text-violet-600 dark:text-violet-400' },
    user: { name: 'User', color: 'text-emerald-600 dark:text-emerald-400' },
    assistant: { name: 'Assistant', color: 'text-sky-600 dark:text-sky-400' },
  };
  const config = labels[role] || { name: role, color: 'text-muted-foreground' };
  
  return (
    <span className={cn('font-semibold text-sm', config.color)}>
      {config.name}
    </span>
  );
}

// ── PIL image gallery ────────────────────────────────────────────────────────

interface LabeledImage {
  varName: string;
  image: SerializedPILImage;
  blockIndex: number;
}

function collectPILImages(iteration: RLMIteration): LabeledImage[] {
  const images: LabeledImage[] = [];
  for (let bi = 0; bi < iteration.code_blocks.length; bi++) {
    const block = iteration.code_blocks[bi];
    if (!block.result?.locals) continue;
    for (const [key, value] of Object.entries(block.result.locals)) {
      if (isSerializedPILImage(value)) {
        images.push({ varName: key, image: value, blockIndex: bi });
      }
    }
  }
  return images;
}

function PILImageGallery({ images }: { images: LabeledImage[] }) {
  const [selected, setSelected] = useState(0);
  if (images.length === 0) return null;
  const current = images[Math.min(selected, images.length - 1)];

  return (
    <div className="rounded-xl border border-amber-500/30 bg-amber-500/5 dark:bg-amber-400/5 p-4">
      {/* Header */}
      <div className="flex items-center gap-2 mb-3">
        <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-amber-500 to-orange-500 flex items-center justify-center">
          <svg className="w-3.5 h-3.5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
          </svg>
        </div>
        <span className="font-semibold text-sm text-amber-700 dark:text-amber-400">
          Visual Variables
        </span>
        <Badge variant="outline" className="ml-auto text-[10px] border-amber-500/30 text-amber-700 dark:text-amber-400">
          {images.length} image{images.length !== 1 ? 's' : ''}
        </Badge>
      </div>

      {/* Main image */}
      <div className="mb-3 rounded-lg overflow-hidden border border-border/60 bg-black/10 flex items-center justify-center min-h-[120px]">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={`data:image/${current.image.format};base64,${current.image.data}`}
          alt={current.varName}
          className="max-h-64 max-w-full object-contain"
        />
      </div>

      {/* Var name + size */}
      <div className="flex items-center justify-between mb-2">
        <span className="font-mono text-xs text-sky-600 dark:text-sky-400">
          {current.varName}
        </span>
        <span className="text-[10px] text-muted-foreground font-mono">
          {current.image.size[0]}×{current.image.size[1]} · block #{current.blockIndex + 1}
        </span>
      </div>

      {/* Thumbnail strip (when multiple images) */}
      {images.length > 1 && (
        <div className="flex gap-1.5 overflow-x-auto mt-2">
          {images.map((img, i) => (
            <button
              key={i}
              onClick={() => setSelected(i)}
              className={`flex-shrink-0 rounded overflow-hidden border transition-all ${
                i === selected
                  ? 'border-amber-500 ring-1 ring-amber-500'
                  : 'border-border/40 opacity-60 hover:opacity-90'
              }`}
              title={img.varName}
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={`data:image/${img.image.format};base64,${img.image.data}`}
                alt={img.varName}
                className="w-14 h-10 object-cover"
              />
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export function TrajectoryPanel({ 
  iterations, 
  selectedIteration, 
}: TrajectoryPanelProps) {
  const currentIteration = iterations[selectedIteration];
  const pilImages = currentIteration ? collectPILImages(currentIteration) : [];

  return (
    <div className="h-full flex flex-col bg-background overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-border flex items-center justify-between bg-muted/30 flex-shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-sky-500 to-indigo-600 flex items-center justify-center">
            <span className="text-white text-sm font-bold">◈</span>
          </div>
          <div>
            <h2 className="font-semibold text-sm">Conversation</h2>
            <p className="text-[11px] text-muted-foreground">
              Iteration {selectedIteration + 1} of {iterations.length}
            </p>
          </div>
        </div>
        <div className="flex gap-2">
          {currentIteration?.code_blocks.length > 0 && (
            <Badge variant="secondary" className="text-[10px]">
              {currentIteration.code_blocks.length} code
            </Badge>
          )}
          {currentIteration?.final_answer && (
            <Badge className="bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border-emerald-500/30 text-[10px]">
              ✓ Answer
            </Badge>
          )}
        </div>
      </div>

      {/* Content - with explicit height constraint for scrolling */}
      <div className="flex-1 min-h-0 overflow-hidden">
        <ScrollArea className="h-full">
          <div className="p-4 space-y-4">
            {/* Prompt messages */}
            {currentIteration?.prompt.map((msg, idx) => (
              <div 
                key={idx}
                className={cn(
                  'rounded-xl border p-4 transition-all',
                  msg.role === 'system' && 'bg-violet-500/5 border-violet-500/20 dark:bg-violet-500/10',
                  msg.role === 'user' && 'bg-emerald-500/5 border-emerald-500/20 dark:bg-emerald-500/10',
                  msg.role === 'assistant' && 'bg-sky-500/5 border-sky-500/20 dark:bg-sky-500/10'
                )}
              >
                {/* Message header */}
                <div className="flex items-center gap-3 mb-3 pb-3 border-b border-border/50">
                  <RoleIcon role={msg.role} />
                  <div className="flex-1">
                    <RoleLabel role={msg.role} />
                    {msg.role === 'system' && (
                      <p className="text-[10px] text-muted-foreground mt-0.5">
                        Instructions & context setup
                      </p>
                    )}
                    {msg.role === 'user' && idx > 0 && (
                      <p className="text-[10px] text-muted-foreground mt-0.5">
                        Continuation prompt
                      </p>
                    )}
                  </div>
                </div>
                
                {/* Message content */}
                <div className="bg-background/60 rounded-lg p-3 border border-border/50">
                  <pre className="whitespace-pre-wrap font-mono text-foreground/90 text-[12px] leading-relaxed overflow-x-auto">
                    {formatMessageContent(msg.content)}
                  </pre>
                </div>
              </div>
            ))}
            
            {/* Current response - highlighted */}
            {currentIteration?.response && (
              <div className="rounded-xl border-2 border-sky-500/40 bg-gradient-to-br from-sky-500/10 to-indigo-500/10 p-4 shadow-lg shadow-sky-500/5">
                {/* Response header */}
                <div className="flex items-center gap-3 mb-3 pb-3 border-b border-sky-500/20">
                  <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-sky-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-sky-500/20">
                    <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                    </svg>
                  </div>
                  <div className="flex-1">
                    <span className="font-semibold text-sm text-sky-600 dark:text-sky-400">
                      Model Response
                    </span>
                    <p className="text-[10px] text-muted-foreground mt-0.5">
                      Iteration {currentIteration.iteration}
                    </p>
                  </div>
                  <Badge variant="outline" className="text-[10px] border-sky-500/30 text-sky-600 dark:text-sky-400">
                    {currentIteration.response.length.toLocaleString()} chars
                  </Badge>
                </div>
                
                {/* Response content */}
                <div className="bg-background/80 rounded-lg p-3 border border-sky-500/20">
                  <pre className="whitespace-pre-wrap font-mono text-foreground text-[12px] leading-relaxed overflow-x-auto">
                    {formatMessageContent(currentIteration.response)}
                  </pre>
                </div>
              </div>
            )}

            {/* PIL image gallery */}
            {pilImages.length > 0 && (
              <PILImageGallery images={pilImages} />
            )}

            {/* Final answer highlight */}
            {currentIteration?.final_answer && (
              <div className="rounded-xl border-2 border-emerald-500/50 bg-gradient-to-br from-emerald-500/15 to-green-500/15 p-4 shadow-lg shadow-emerald-500/10">
                <div className="flex items-center gap-3 mb-3">
                  <div className="w-10 h-10 rounded-full bg-gradient-to-br from-emerald-500 to-green-600 flex items-center justify-center shadow-lg shadow-emerald-500/30">
                    <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
                    </svg>
                  </div>
                  <div>
                    <span className="font-bold text-emerald-600 dark:text-emerald-400 text-base">
                      Final Answer
                    </span>
                    <p className="text-[10px] text-muted-foreground">
                      Task completed successfully
                    </p>
                  </div>
                </div>
                <div className="bg-background/80 rounded-lg p-4 border border-emerald-500/30">
                  <p className="text-[15px] font-medium text-foreground leading-relaxed">
                    {extractFinalAnswer(currentIteration.final_answer)}
                  </p>
                </div>
              </div>
            )}
            
            {/* Bottom padding for scroll */}
            <div className="h-4" />
          </div>
        </ScrollArea>
      </div>
    </div>
  );
}
