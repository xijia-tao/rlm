'use client';

import { useState, useEffect, useCallback } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from '@/components/ui/resizable';
import { StatsCard } from './StatsCard';
import { TrajectoryPanel } from './TrajectoryPanel';
import { ExecutionPanel } from './ExecutionPanel';
import { IterationTimeline } from './IterationTimeline';
import { ThemeToggle } from './ThemeToggle';
import { ContextPreview, RLMLogFile } from '@/lib/types';

// ── Context preview panel ────────────────────────────────────────────────────

function ContextPreviewPanel({ preview }: { preview: ContextPreview }) {
  const [selectedFrame, setSelectedFrame] = useState(0);

  if (preview.context_type === 'image') {
    return (
      <div className="flex flex-col gap-1.5">
        <p className="text-[10px] uppercase tracking-wider text-muted-foreground font-medium">
          Input Image
        </p>
        <div className="rounded-lg overflow-hidden border border-border/60 bg-muted/30 flex items-center justify-center" style={{ maxHeight: 120 }}>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={`data:image/jpeg;base64,${preview.image_data}`}
            alt="Input image"
            className="object-contain max-h-[120px] max-w-full"
          />
        </div>
        {preview.source_path && (
          <p className="text-[9px] text-muted-foreground truncate font-mono">
            {preview.source_path.split('/').pop()}
          </p>
        )}
      </div>
    );
  }

  // Video
  const frame = preview.frames[selectedFrame];
  const minutes = Math.floor(preview.duration_sec / 60);
  const secs = (preview.duration_sec % 60).toFixed(0).padStart(2, '0');
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center justify-between">
        <p className="text-[10px] uppercase tracking-wider text-muted-foreground font-medium">
          Input Video
        </p>
        <span className="text-[9px] text-muted-foreground font-mono">
          {minutes}:{secs} · {preview.resolution[0]}×{preview.resolution[1]}
        </span>
      </div>
      {/* Main frame */}
      <div className="rounded-lg overflow-hidden border border-border/60 bg-muted/30 flex items-center justify-center" style={{ maxHeight: 100 }}>
        {frame && (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={`data:image/jpeg;base64,${frame.data}`}
            alt={`Frame at ${frame.timestamp}s`}
            className="object-contain max-h-[100px] max-w-full"
          />
        )}
      </div>
      {/* Frame strip */}
      <div className="flex gap-1 overflow-x-auto">
        {preview.frames.map((f, i) => (
          <button
            key={i}
            onClick={() => setSelectedFrame(i)}
            className={`flex-shrink-0 rounded overflow-hidden border transition-all ${
              i === selectedFrame ? 'border-primary ring-1 ring-primary' : 'border-border/40 opacity-60 hover:opacity-90'
            }`}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={`data:image/jpeg;base64,${f.data}`}
              alt={`Frame ${i}`}
              className="w-12 h-8 object-cover"
            />
          </button>
        ))}
      </div>
      {frame && (
        <p className="text-[9px] text-muted-foreground font-mono">t={frame.timestamp.toFixed(1)}s</p>
      )}
    </div>
  );
}

interface LogViewerProps {
  logFile: RLMLogFile;
  onBack: () => void;
}

export function LogViewer({ logFile, onBack }: LogViewerProps) {
  const [selectedIteration, setSelectedIteration] = useState(0);
  const { iterations, metadata, config } = logFile;

  const goToPrevious = useCallback(() => {
    setSelectedIteration(prev => Math.max(0, prev - 1));
  }, []);

  const goToNext = useCallback(() => {
    setSelectedIteration(prev => Math.min(iterations.length - 1, prev + 1));
  }, [iterations.length]);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft' || e.key === 'j') {
        goToPrevious();
      } else if (e.key === 'ArrowRight' || e.key === 'k') {
        goToNext();
      } else if (e.key === 'Escape') {
        onBack();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [goToPrevious, goToNext, onBack]);

  return (
    <div className="h-screen flex flex-col overflow-hidden bg-background">
      {/* Top Bar - Compact header */}
      <header className="border-b border-border bg-card/80 backdrop-blur-sm">
        <div className="px-6 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Button 
                variant="ghost" 
                size="sm" 
                onClick={onBack}
                className="text-muted-foreground hover:text-foreground"
              >
                ← Back
              </Button>
              <div className="h-5 w-px bg-border" />
              <div>
                <h1 className="font-semibold flex items-center gap-2 text-sm">
                  <span className="text-primary">◈</span>
                  {logFile.fileName}
                </h1>
                <p className="text-[10px] text-muted-foreground font-mono mt-0.5">
                  {config.root_model ?? 'Unknown model'} • {config.backend ?? 'Unknown backend'} • {config.environment_type ?? 'Unknown env'}
                </p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              {metadata.hasErrors && (
                <Badge variant="destructive" className="text-xs">Has Errors</Badge>
              )}
              {metadata.finalAnswer && (
                <Badge className="bg-emerald-500 hover:bg-emerald-600 text-white text-xs">
                  Completed
                </Badge>
              )}
              <ThemeToggle />
            </div>
          </div>
        </div>
      </header>

      {/* Question & Answer + Stats Row */}
      <div className="border-b border-border bg-muted/30 px-6 py-4">
        <div className="flex gap-6">
          {/* Question & Answer Summary */}
          <Card className="flex-1 bg-gradient-to-r from-primary/5 to-accent/5 border-primary/20">
            <CardContent className="p-4">
              <div className={`grid gap-4 ${logFile.contextPreview ? 'md:grid-cols-3' : 'md:grid-cols-2'}`}>
                {/* Original image/video input */}
                {logFile.contextPreview && (
                  <div>
                    <ContextPreviewPanel preview={logFile.contextPreview} />
                  </div>
                )}
                <div>
                  <p className="text-[10px] uppercase tracking-wider text-muted-foreground font-medium mb-1">
                    Context / Question
                  </p>
                  <p className="text-sm font-medium line-clamp-2">
                    {metadata.contextQuestion}
                  </p>
                </div>
                <div>
                  <p className="text-[10px] uppercase tracking-wider text-muted-foreground font-medium mb-1">
                    Final Answer
                  </p>
                  <p className="text-sm font-medium text-emerald-600 dark:text-emerald-400 line-clamp-2">
                    {metadata.finalAnswer || 'Not yet completed'}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Quick Stats */}
          <div className="flex gap-2">
            <StatsCard
              label="Iterations"
              value={metadata.totalIterations}
              icon="◎"
              variant="cyan"
            />
            <StatsCard
              label="Code"
              value={metadata.totalCodeBlocks}
              icon="⟨⟩"
              variant="green"
            />
            <StatsCard
              label="Sub-LM"
              value={metadata.totalSubLMCalls}
              icon="◇"
              variant="magenta"
            />
            <StatsCard
              label="Exec"
              value={`${metadata.totalExecutionTime.toFixed(2)}s`}
              icon="⏱"
              variant="yellow"
            />
          </div>
        </div>
      </div>

      {/* Iteration Timeline - Full width scrollable row */}
      <IterationTimeline
        iterations={iterations}
        selectedIteration={selectedIteration}
        onSelectIteration={setSelectedIteration}
      />

      {/* Main Content - Resizable Split View */}
      <div className="flex-1 min-h-0">
        <ResizablePanelGroup orientation="horizontal">
          {/* Left Panel - Prompt & Response */}
          <ResizablePanel defaultSize={50} minSize={20} maxSize={80}>
            <div className="h-full border-r border-border">
              <TrajectoryPanel
                iterations={iterations}
                selectedIteration={selectedIteration}
                onSelectIteration={setSelectedIteration}
              />
            </div>
          </ResizablePanel>

          <ResizableHandle withHandle className="bg-border hover:bg-primary/30 transition-colors" />

          {/* Right Panel - Code Execution & Sub-LM Calls */}
          <ResizablePanel defaultSize={50} minSize={20} maxSize={80}>
            <div className="h-full bg-background">
              <ExecutionPanel
                iteration={iterations[selectedIteration] || null}
              />
            </div>
          </ResizablePanel>
        </ResizablePanelGroup>
      </div>

      {/* Keyboard hint footer */}
      <div className="border-t border-border bg-muted/30 px-6 py-1.5">
        <div className="flex items-center justify-center gap-6 text-[10px] text-muted-foreground">
          <span className="flex items-center gap-1">
            <kbd className="px-1 py-0.5 bg-muted rounded text-[9px]">←</kbd>
            <kbd className="px-1 py-0.5 bg-muted rounded text-[9px]">→</kbd>
            Navigate
          </span>
          <span className="flex items-center gap-1">
            <kbd className="px-1 py-0.5 bg-muted rounded text-[9px]">Esc</kbd>
            Back
          </span>
        </div>
      </div>
    </div>
  );
}
