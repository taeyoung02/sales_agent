"use client";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { sendVanillaChatMessageStream } from "@/lib/api";
import { ChatMessage } from "@/lib/types";
import { Send, Sparkles } from 'lucide-react';
import { useEffect, useRef, useState } from "react";

interface VanillaChatPanelProps {
  locale?: "en" | "ko";
}

const PARTICIPANT_ID_KEY = "vanilla_chat_participant_id";

/**
 * URL 쿼리 파라미터에서 participant_id 읽어서 sessionStorage에 저장
 * Google Form에서 링크를 통해 들어올 때 사용
 */
function initializeParticipantMetadata(): void {
  if (typeof window === "undefined") return;

  const urlParams = new URLSearchParams(window.location.search);
  
  // participant_id 처리
  const participantId = urlParams.get("participant_id");
  if (participantId) {
    // URL에 있으면 sessionStorage에 저장 (기존 값이 없을 때만)
    if (!sessionStorage.getItem(PARTICIPANT_ID_KEY)) {
      sessionStorage.setItem(PARTICIPANT_ID_KEY, participantId);
      console.log(`[VanillaChatPanel] Participant ID from URL: ${participantId}`);
    }
  }
}

/**
 * sessionStorage에서 participant_id 가져오기
 */
function getParticipantMetadata(): { participant_id?: string } {
  if (typeof window === "undefined") {
    return {};
  }

  return {
    participant_id: sessionStorage.getItem(PARTICIPANT_ID_KEY) || undefined,
  };
}

export function VanillaChatPanel({ 
  locale = "ko"
}: VanillaChatPanelProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "1",
      role: "assistant",
      content:
        "안녕하세요! 중고차 정보를 도와드리는 챗봇입니다.\n무엇을 도와드릴까요?",
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [loadingStage, setLoadingStage] = useState<{ stage: string; status: string; message?: string } | null>(null);

  const conversationHistoryRef = useRef<Array<{ role: string; content: string }>>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  // 페이지 로드 시 URL 쿼리에서 participant_id 읽기
  useEffect(() => {
    initializeParticipantMetadata();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const scrollToBottom = () => {
    setTimeout(() => {
      if (messagesEndRef.current) {
        messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
      } else if (scrollAreaRef.current) {
        scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight;
      }
    }, 100);
  };

  const handleSend = async (text?: string) => {
    const messageText = text || input;
    if (!messageText.trim()) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: "user",
      content: messageText,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);
    setLoadingStage(null);

    // Add to conversation history
    conversationHistoryRef.current.push({
      role: "user",
      content: messageText,
    });

    try {
      // Streaming을 사용하여 dialogue를 실시간으로 표시
      let accumulatedDialogue = "";
      let evidence: any[] = [];
      let assistantMessageId = (Date.now() + 1).toString();
      
      // Assistant 메시지를 먼저 생성 (빈 내용으로 시작)
      const assistantMessage: ChatMessage = {
        id: assistantMessageId,
        role: "assistant",
        content: "",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, assistantMessage]);

      // participant_id 가져오기
      const metadata = getParticipantMetadata();

      // 서버 중심 방식: session_id는 SessionMiddleware가 쿠키(HttpOnly)에서 자동으로 읽음
      await sendVanillaChatMessageStream(
        {
          message: messageText,
          conversation_history: conversationHistoryRef.current,
          participant_id: metadata.participant_id,
        },
        {
          onDialogueChunk: (text: string, accumulated: string) => {
            // 실시간으로 dialogue 업데이트
            accumulatedDialogue = accumulated;
            // Dialogue가 시작되면 로딩 메시지 숨기기
            if (accumulated.trim().length > 0) {
              setLoadingStage(null);
            }
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantMessageId
                  ? { ...msg, content: accumulated }
                  : msg
              )
            );
          },
          onStageUpdate: (stage: string, status: string, data?: any) => {
            // 진행 상황 업데이트 및 UI 표시
            console.log(`[VanillaChat] Stage: ${stage}, Status: ${status}`, data);
            
            // 단계별 메시지 생성 (한글만)
            let message = "";
            if (stage === "rag_search") {
              if (status === "started") {
                message = "관련 정보 검색 중";
              }
            } else if (stage === "response_generation") {
              if (status === "started") {
                message = "답변 생성 중";
              }
            }
            
            setLoadingStage({ stage, status, message });
          },
          onEvidence: (receivedEvidence: any[]) => {
            evidence = receivedEvidence;
          },
          onComplete: () => {
            // Evidence 업데이트
            const formattedEvidence = evidence.map((e: any) => ({
              title: e.title || "RAG Context",
              snippet: e.snippet || "",
              asOf: e.asOf || "Current",
              href: e.href,
            }));

            // Assistant 메시지에 evidence 추가
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantMessageId
                  ? { ...msg, evidence: formattedEvidence }
                  : msg
              )
            );

            // Conversation history 업데이트
            conversationHistoryRef.current.push({
              role: "assistant",
              content: accumulatedDialogue,
            });

            setIsLoading(false);
            setLoadingStage(null);
          },
          onError: (error: Error) => {
            console.error("Vanilla chat stream error:", error);
            setIsLoading(false);
            setLoadingStage(null);
            
            // 에러 메시지 표시
            const errorMessage: ChatMessage = {
              id: (Date.now() + 2).toString(),
              role: "assistant",
              content: locale === "en"
                ? `An error occurred: ${error.message}`
                : `오류가 발생했습니다: ${error.message}`,
              timestamp: new Date(),
            };
            setMessages((prev) => [...prev, errorMessage]);
          },
        }
      );
    } catch (error) {
      console.error("Vanilla chat API error:", error);
      setIsLoading(false);
      setLoadingStage(null);
      
      // Show error message
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: locale === "en" 
          ? "Sorry, I encountered an error. Please try again."
          : "죄송합니다. 오류가 발생했습니다. 다시 시도해주세요.",
        timestamp: new Date(),
      };
      
      setMessages((prev) => [...prev, errorMessage]);
    }
  };

  return (
    <div className="flex h-full flex-col min-h-0">
      <div 
        ref={scrollAreaRef}
        className="flex-1 overflow-y-auto p-4 scroll-smooth min-h-0"
      >
        <div className="space-y-4">
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex ${
                msg.role === "user" ? "justify-end" : "justify-start"
              }`}
            >
              <div
                className={`max-w-[80%] rounded-lg px-4 py-2 ${
                  msg.role === "user"
                    ? "bg-primary text-primary-foreground"
                    : "bg-muted"
                }`}
              >
                {!isLoading && msg.role === "assistant" && (
                  <Sparkles className="mb-1 inline h-4 w-4" />
                )}
                <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
              </div>
            </div>
          ))}
          {isLoading && loadingStage && (
            <div className="flex justify-start">
              <div className="rounded-lg bg-muted px-4 py-2">
                <div className="flex items-center gap-2">
                  {loadingStage.message && (
                    <span className="text-sm text-foreground/70">
                      {loadingStage.message}
                    </span>
                  )}
                  <div className="flex gap-1">
                    <div className="h-2 w-2 animate-bounce rounded-full bg-foreground/50" />
                    <div className="h-2 w-2 animate-bounce rounded-full bg-foreground/50 [animation-delay:0.2s]" />
                    <div className="h-2 w-2 animate-bounce rounded-full bg-foreground/50 [animation-delay:0.4s]" />
                  </div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      <div className="space-y-3 border-t border-border p-4">
        <div className="flex gap-2">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
            placeholder={
              locale === "en"
                ? "Ask about used cars..."
                : "중고차에 대해 물어보세요..."
            }
            disabled={isLoading}
          />
          <Button onClick={() => handleSend()} disabled={isLoading} size="icon">
            <Send className="h-4 w-4" />
            <span className="sr-only">Send message</span>
          </Button>
        </div>
      </div>
    </div>
  );
}

