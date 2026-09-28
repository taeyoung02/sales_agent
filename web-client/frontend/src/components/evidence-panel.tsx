"use client";

import { Evidence } from "@/lib/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ExternalLink, Calendar } from 'lucide-react';
import { ScrollArea } from "@/components/ui/scroll-area";

interface EvidencePanelProps {
  evidence: Evidence[];
  title?: string;
}

export function EvidencePanel({
  evidence,
  title = "Sources & Evidence",
}: EvidencePanelProps) {
  if (evidence.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-center">
        <div className="text-muted-foreground">
          <p className="text-sm">No sources available yet.</p>
          <p className="mt-1 text-xs">
            Sources will appear here as you interact with the AI.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full">
      <div className="border-b border-border px-4 py-3">
        <h3 className="font-semibold">{title}</h3>
      </div>
      <ScrollArea className="h-[calc(100%-57px)]">
        <div className="space-y-3 p-4">
          {evidence.map((item, index) => (
            <Card key={index} className="shadow-sm">
              <CardHeader className="pb-3">
                <CardTitle className="flex items-start justify-between gap-2 text-sm">
                  <span className="line-clamp-2">{item.title}</span>
                  {item.href && (
                    <a
                      href={item.href}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="shrink-0 text-primary hover:text-primary/80"
                    >
                      <ExternalLink className="h-4 w-4" />
                      <span className="sr-only">Open link</span>
                    </a>
                  )}
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <p className="text-sm text-muted-foreground">{item.snippet}</p>
                <div className="flex items-center gap-1 text-xs text-muted-foreground">
                  <Calendar className="h-3 w-3" />
                  <span>as of {item.asOf}</span>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </ScrollArea>
    </div>
  );
}
