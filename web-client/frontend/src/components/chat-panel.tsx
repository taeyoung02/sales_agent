"use client";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { generateHeatmap, sendChatMessage, sendChatMessageStream, saveSessionLog, getCameraPoseData } from "@/lib/api";
import { calculateInitialCameraView, transformToPLY } from "@/lib/camera-utils";
import { useViewerStore } from "@/lib/state";
import { ChatMessage, PresentationScript, PresentationSegment } from "@/lib/types";
import { Send, Sparkles } from 'lucide-react';
import { useEffect, useRef, useState } from "react";
import { buildPointCloudPath } from "@/lib/static-path";

interface ChatPanelProps {
  onEvidenceUpdate?: (evidence: any[]) => void;
  onHeatmapGenerated?: (path: string) => void;
  onVehicleRequested?: (vehicleId: string, sourcePath: string, compareWithVehicleId?: string, compareWithSourcePath?: string) => void;
  locale?: "en" | "ko";
  vehicleId?: string;
}

const suggestedPrompts = {
  en: [
    "Recommend a used car for me",
    "Show me the list of used cars",
    "Show me the wheels of the car",
  ],
  ko: [
    "중고차를 추천해줘",
    "중고차 목록을 보여줘",
    "자동차 트렁크를 보여줘",
  ],
};

// 세션 ID는 서버 중심 방식으로 관리됨
// - SessionMiddleware가 쿠키(HttpOnly)로 세션 ID 생성/관리
// - 프론트엔드에서는 sessionStorage 사용 안 함

const PARTICIPANT_ID_KEY = "chat_participant_id";

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
      console.log(`[ChatPanel] Participant ID from URL: ${participantId}`);
    }
  }
}

/**
 * sessionStorage에서 participant_id와 user_goal_segment 가져오기
 */
function getParticipantMetadata(): { participant_id?: string; user_goal_segment?: string } {
  if (typeof window === "undefined") {
    return {};
  }

  return {
    participant_id: sessionStorage.getItem(PARTICIPANT_ID_KEY) || undefined,
  };
}

export function ChatPanel({ 
  onEvidenceUpdate, 
  onHeatmapGenerated,
  onVehicleRequested,
  locale = "ko",
  vehicleId 
}: ChatPanelProps) {
  const { startPresentation, stopPresentation, show3DViewer } = useViewerStore();
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "1",
      role: "assistant",
      content: "안녕하세요!\n무엇을 도와드릴까요?",
      timestamp: new Date(),
    }
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [loadingStage, setLoadingStage] = useState<{ stage: string; status: string; message?: string } | null>(null);
  const { 
    setCamera, 
    setCameraPosition, 
    setCameraTarget,
    setCameraUp,
    setCameraZoom,
    zoomCamera
  } = useViewerStore();

  const conversationHistoryRef = useRef<Array<{ role: string; content: string }>>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  // 페이지 로드 시 URL 쿼리에서 participant_id 읽기
  useEffect(() => {
    initializeParticipantMetadata();
  }, []);

  // Note: 서버 중심 방식에서는 브라우저 종료 시 명시적 저장 불필요
  // - Redis TTL 만료 시 자동 저장 (30분 비활성)
  // - BaseAgent 주기적 정리 (5분마다)
  // - HttpOnly 쿠키로 인해 프론트엔드에서 session_id 접근 불가

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const scrollToBottom = () => {
    // Use setTimeout to ensure DOM is updated
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

    // 기존 실행 중인 프레젠테이션 중단 (비동기로 완전히 정리될 때까지 대기)
    await stopPresentation();

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
      let toolCalls: any[] = [];
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

      // participant_id와 user_goal_segment 가져오기
      const metadata = getParticipantMetadata();

      // 서버 중심 방식: session_id는 SessionMiddleware가 쿠키(HttpOnly)에서 자동으로 읽음
      // 프론트엔드에서는 session_id를 전송하지 않음 (또는 optional로 처리)
      await sendChatMessageStream(
        {
          message: messageText,
          // session_id 제거 - 서버가 쿠키에서 자동으로 읽음
          conversation_history: conversationHistoryRef.current,
          vehicle_id: vehicleId,
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
            console.log(`[Chat] Stage: ${stage}, Status: ${status}`, data);
            
            // 단계별 메시지 생성 (한글만, "..." 제거)
            let message = "";
            if (stage === "planner") {
              if (status === "started") {
                message = "답변 생각 중";
              }
            } else if (stage === "evaluator") {
              if (status === "started") {
                message = "답변 검토 중";
              }
            } else if (stage === "executor") {
              if (status === "started") {
                message = "최종 답변 생성 중";
              }
            }
            
            setLoadingStage({ stage, status, message });
          },
          onToolCalls: (receivedToolCalls: any[]) => {
            toolCalls = receivedToolCalls;
          },
          onEvidence: (receivedEvidence: any[]) => {
            evidence = receivedEvidence;
            if (onEvidenceUpdate) {
              onEvidenceUpdate(receivedEvidence);
            }
          },
          onComplete: async () => {
            // Tool calls 처리 (기존 로직 재사용)
            const heatmapToolCall = toolCalls.find(tc => tc.name === "generateHeatmap");
            const otherToolCalls = toolCalls.filter(tc => tc.name !== "generateHeatmap") || [];
      
      // generateHeatmap이 호출되었는지 추적 (setCamera와의 충돌 방지)
      let heatmapWasCalled = false;

      // 먼저 generateHeatmap 처리
      if (heatmapToolCall) {
        heatmapWasCalled = true;
        const heatmapQuery = heatmapToolCall.arguments.query;
        
        if (!heatmapQuery) {
          console.error("❌ generateHeatmap: query is missing");
        } else {
          console.log(`🚀 Generating heatmap for query: "${heatmapQuery}"`);
          
          try {
            const heatmapResult = await generateHeatmap({
              query: heatmapQuery,
              vehicle_id: vehicleId,
            });

            console.log("✅ Heatmap result:", heatmapResult);

            // Check environment variable for heatmap display control
            const shouldDisplayHeatmap = process.env.NEXT_PUBLIC_HEATMAP_DISPLAY_PLY === 'true';
            
            // Note: In production, we don't use the PLY file - only target_position for camera movement
            // PLY file generation is only for development/debugging (HEATMAP_GENERATE_PLY=true)
            // Only call onHeatmapGenerated if display is enabled and output_path is provided (dev mode only)
            if (heatmapResult.success && onHeatmapGenerated && shouldDisplayHeatmap && 
                heatmapResult.output_path && heatmapResult.output_path.trim() !== '') {
              // Check if output_path is a valid PLY file path
              if (heatmapResult.output_path.endsWith('.ply')) {
                console.log(`📦 Loading heatmap PLY file for overlay: ${heatmapResult.output_path}`);
                onHeatmapGenerated(heatmapResult.output_path);
              } else {
                console.log(`ℹ️  Heatmap output_path provided but not a PLY file: ${heatmapResult.output_path}`);
              }
            } else if (heatmapResult.success && !shouldDisplayHeatmap) {
              console.log("ℹ️  Heatmap display disabled (NEXT_PUBLIC_HEATMAP_DISPLAY_PLY=false) - camera only mode");
            } else if (heatmapResult.success && !heatmapResult.output_path) {
              console.log("ℹ️  Heatmap metadata loaded (production mode - camera only, no PLY file)");
            }

            // Move camera using calculated position from backend (preset system)
            if (heatmapResult.camera_position && heatmapResult.surface_center) {
              // Transform from JSON coordinates to PLY coordinates
              const plyCameraPos = transformToPLY(heatmapResult.camera_position as [number, number, number]);
              const plySurfaceCenter = transformToPLY(heatmapResult.surface_center as [number, number, number]);
              
              setCameraPosition(plyCameraPos);
              setCameraTarget(plySurfaceCenter);
              
              // ✅ up vector도 설정 (있으면)
              if (heatmapResult.camera_up) {
                const plyUpVector = transformToPLY(heatmapResult.camera_up as [number, number, number]);
                setCameraUp(plyUpVector);
                console.log(`📐 Camera up vector (PLY coords): (${plyUpVector[0].toFixed(3)}, ${plyUpVector[1].toFixed(3)}, ${plyUpVector[2].toFixed(3)})`);
              }
              
              console.log(`📷 Camera moved to preset position (PLY coords): (${plyCameraPos[0].toFixed(2)}, ${plyCameraPos[1].toFixed(2)}, ${plyCameraPos[2].toFixed(2)})`);
              console.log(`🎯 Surface center (PLY coords): (${plySurfaceCenter[0].toFixed(2)}, ${plySurfaceCenter[1].toFixed(2)}, ${plySurfaceCenter[2].toFixed(2)})`);
              if (heatmapResult.surface_normal) {
                const plyNormal = transformToPLY(heatmapResult.surface_normal as [number, number, number]);
                console.log(`📐 Surface normal (PLY coords): (${plyNormal[0].toFixed(3)}, ${plyNormal[1].toFixed(3)}, ${plyNormal[2].toFixed(3)})`);
              }
            } else if (heatmapResult.target_position) {
              // Fallback to legacy calculation if preset data not available
              const plyTargetPos = transformToPLY(heatmapResult.target_position as [number, number, number]);
              const [x, y, z] = plyTargetPos;
              
              // Calculate camera position: offset from target to view the object
              const cameraDistance = 2.0;
              const cameraX = x + cameraDistance;
              const cameraY = y + cameraDistance * 0.5;
              const cameraZ = z + cameraDistance;
              
              setCameraPosition([cameraX, cameraY, cameraZ]);
              setCameraTarget(plyTargetPos);
              
              console.log(`📷 Camera moved to target position (legacy, PLY coords): (${cameraX.toFixed(2)}, ${cameraY.toFixed(2)}, ${cameraZ.toFixed(2)}) for object at (${x.toFixed(2)}, ${y.toFixed(2)}, ${z.toFixed(2)})`);
            }
          } catch (error) {
            console.error("Heatmap generation error:", error);
          }
        }
      }

      // 다른 tool calls 처리 (카메라 제어 등)
      for (const toolCall of otherToolCalls) {
        console.log(`Processing tool call: ${toolCall.name}`, toolCall.arguments);
        if (toolCall.name === "loadVehicle") {
          const vehicleId = toolCall.arguments.vehicle_id as string | undefined;
          const compareWithVehicleId = toolCall.arguments.compare_with_vehicle_id as string | undefined;
          const sourcePathFromBackend = toolCall.arguments.source_path as string | undefined;

          if (vehicleId && onVehicleRequested) {
            const canonicalSourcePath = buildPointCloudPath(vehicleId);

            if (
              sourcePathFromBackend &&
              sourcePathFromBackend !== canonicalSourcePath
            ) {
              console.warn(
                "[ChatPanel] Ignoring mismatched source_path from backend:",
                { vehicleId, sourcePathFromBackend, canonicalSourcePath }
              );
            }

            // 다중 차량 비교 모드
            if (compareWithVehicleId) {
              const compareWithSourcePath = buildPointCloudPath(compareWithVehicleId);
              onVehicleRequested(vehicleId, canonicalSourcePath, compareWithVehicleId, compareWithSourcePath);
              console.log(`[ChatPanel] Loading vehicles for comparison: ${vehicleId} and ${compareWithVehicleId}`);
            } else {
              onVehicleRequested(vehicleId, canonicalSourcePath);
            }
          }

          console.log("Vehicle ID:", vehicleId);
          if (compareWithVehicleId) {
            console.log("Compare with Vehicle ID:", compareWithVehicleId);
          }
        } else if (toolCall.name === "setCamera") {
          const preset = toolCall.arguments.preset;
          const position = toolCall.arguments.position;
          
          if (preset) {
            setCamera(preset as any);
          }
          if (position && Array.isArray(position) && position.length === 3) {
            setCameraPosition(position as [number, number, number]);
          }
        } else if (toolCall.name === "zoomCamera") {
          const direction = toolCall.arguments.direction;
          const level = toolCall.arguments.level;
          
          if (level !== undefined) {
            setCameraZoom(level);
          } else if (direction) {
            zoomCamera(direction as "in" | "out");
          }
        } else if (toolCall.name === "generatePresentationScript") {
          // Handle presentation script
          const rawScript = toolCall.arguments.script as any;
          const vehicle_id = toolCall.arguments.vehicle_id as string;
          
          // Validate script structure before using
          if (!rawScript) {
            console.error("[ChatPanel] generatePresentationScript: script is missing");
            return;
          }
          
          if (!vehicle_id) {
            console.error("[ChatPanel] generatePresentationScript: vehicle_id is missing");
            return;
          }
          
          // Validate segments array
          if (!rawScript.segments || !Array.isArray(rawScript.segments)) {
            console.error("[ChatPanel] generatePresentationScript: script.segments is missing or not an array", {
              script: rawScript,
              hasSegments: !!rawScript.segments,
              segmentsType: typeof rawScript.segments,
            });
            return;
          }
          
          if (rawScript.segments.length === 0) {
            console.error("[ChatPanel] generatePresentationScript: script.segments is empty");
            return;
          }
          
          // Validate segments format (must match backend tool definition)
          // Backend must send: {text: string, actions: Array, timestamp?: number}
          // Actions must be: {type: string, preset?: string, duration?: number, ...}
          const invalidSegments = rawScript.segments.filter(
            (seg: any, idx: number) => {
              // Check required fields
              if (!seg || !seg.text || !seg.actions || !Array.isArray(seg.actions)) {
                return true;
              }
              
              // Validate each action has required 'type' field
              const invalidActions = seg.actions.filter((action: any) => !action || !action.type);
              if (invalidActions.length > 0) {
                console.warn(`[ChatPanel] Segment ${idx} has invalid actions:`, invalidActions);
                return true;
              }
              
              return false;
            }
          );
          
          if (invalidSegments.length > 0) {
            console.error("[ChatPanel] generatePresentationScript: some segments are invalid", {
              invalidCount: invalidSegments.length,
              expectedFormat: {
                text: "string (required)",
                actions: "Array<{type: string, preset?: string, duration?: number, ...}> (required)",
                timestamp: "number (optional)",
              },
            });
            return;
          }
          
          try {
            // Use script directly as-is (backend must send correct format)
            // Format: {text: string, actions: Array, timestamp?: number}
            const script: PresentationScript = {
              vehicle_id: vehicle_id,
              total_duration: rawScript.total_duration || 0,
              segments: rawScript.segments as PresentationSegment[],
            };
            
            // Ensure total_duration is set (calculate if missing)
            if (!script.total_duration || script.total_duration <= 0) {
              // Estimate total duration from segments (rough estimate: 0.05 seconds per character)
              const totalTextLength = script.segments.reduce((sum, seg) => sum + (seg.text?.length || 0), 0);
              script.total_duration = Math.max(30, totalTextLength * 0.05); // At least 30 seconds
              console.warn("[ChatPanel] generatePresentationScript: total_duration was missing, estimated:", script.total_duration);
            }
            
            console.log("[ChatPanel] Starting presentation:", {
              vehicle_id,
              segmentsCount: script.segments.length,
              totalDuration: script.total_duration,
              firstSegment: script.segments[0],
              transformedActions: script.segments[0]?.actions,
            });

            // Stop any existing presentation first to prevent overlap
            await stopPresentation();

            // Use store's startPresentation to ensure PresentationControls is displayed
            startPresentation(script);
          } catch (error) {
            console.error("[ChatPanel] Error starting presentation:", error);
          }
        }
      }

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
            console.error("Chat stream error:", error);
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
      console.error("Chat API error:", error);
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
    } finally {
      setIsLoading(false);
      setLoadingStage(null);
    }
  };

  return (
    <div 
      className={`flex h-full flex-col min-h-0 bg-gradient-to-b from-purple-50/30 to-white ${
        !show3DViewer ? 'shadow-2xl border-1' : ''
      }`}
      style={{
        ...(!show3DViewer && {
          boxShadow: '0 20px 60px rgba(99, 39, 242, 0.15), 0 0 0 1px rgba(99, 39, 242, 0.1)',
        })
      }}
    >
      <div 
        ref={scrollAreaRef}
        className="flex-1 overflow-y-auto px-4 py-6 scroll-smooth min-h-0"
        style={{
          backgroundImage: 'radial-gradient(circle at 20% 50%, rgba(99, 39, 242, 0.04) 0%, transparent 50%), radial-gradient(circle at 80% 80%, rgba(99, 39, 242, 0.04) 0%, transparent 50%)'
        }}
      >
        <div className="space-y-3 max-w-4xl mx-auto">
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex ${
                msg.role === "user" ? "justify-end" : "justify-start"
              } animate-in fade-in slide-in-from-bottom-2 duration-300`}
            >
              {
                msg.content !== "" && (
                  <div
                    className={`max-w-[85%] rounded-2xl px-4 py-3 shadow-sm ${
                      msg.role === "user"
                      ? "text-white rounded-br-sm"
                      : "bg-white text-gray-800 rounded-bl-sm border border-purple-100"
                      }`}
                      style={{
                        backgroundColor: msg.role === "user" ? "#6327F2" : "#F5EFFF",
                        boxShadow: msg.role === "user" 
                        ? "0 2px 8px rgba(99, 39, 242, 0.35)" 
                        : "0 2px 8px rgba(99, 39, 242, 0.1)"
                      }}
                      >
                  {!isLoading && msg.role === "assistant" && (
                    <Sparkles className="mb-1.5 inline h-3.5 w-3.5" style={{ color: "#6327F2" }} />
                  )}
                  <p className="text-sm leading-relaxed whitespace-pre-wrap break-words">{msg.content}</p>
                </div>
              )}
            </div>
          ))}
          {isLoading && loadingStage && (
            <div className="flex justify-start animate-in fade-in slide-in-from-bottom-2 duration-300">
              <div className="rounded-2xl rounded-bl-sm bg-white px-4 py-3 border border-purple-100 shadow-sm">
                <div className="flex items-center gap-2">
                  {loadingStage.message && (
                    <span className="text-sm text-gray-600">
                      {loadingStage.message}
                    </span>
                  )}
                  <div className="flex gap-1">
                    <div className="h-2 w-2 animate-bounce rounded-full" style={{ backgroundColor: "#6327F2" }} />
                    <div className="h-2 w-2 animate-bounce rounded-full [animation-delay:0.2s]" style={{ backgroundColor: "#6327F2" }} />
                    <div className="h-2 w-2 animate-bounce rounded-full [animation-delay:0.4s]" style={{ backgroundColor: "#6327F2" }} />
                  </div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      <div className="border-t border-purple-100 bg-white/90 backdrop-blur-sm p-4">
        {/* <div className="flex flex-wrap gap-2 mb-3">
          {suggestedPrompts[locale].map((prompt, i) => (
            <Button
              key={i}
              variant="outline"
              size="sm"
              onClick={() => handleSend(prompt)}
              disabled={isLoading}
              className="rounded-full text-xs"
            >
              {prompt}
            </Button>
          ))}
        </div> */}

        <div className="flex gap-2 max-w-4xl mx-auto">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
            placeholder={
              locale === "en"
                ? "Ask about heatmaps, camera controls, or 3D model features..."
                : "메시지를 입력하세요..."
            }
            disabled={isLoading}
            className="rounded-full border-purple-200 bg-purple-50/50 focus:bg-white focus:border-[#6327F2] focus:ring-2 focus:ring-[#6327F2]/20 transition-all"
          />
          <Button 
            onClick={() => handleSend()} 
            disabled={isLoading || !input.trim()} 
            size="icon"
            className="rounded-full text-white shadow-md hover:shadow-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed h-10 w-10"
            style={{
              backgroundColor: "#6327F2",
            }}
            onMouseEnter={(e) => {
              if (!isLoading && input.trim()) {
                e.currentTarget.style.backgroundColor = "#5420D9";
              }
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = "#6327F2";
            }}
          >
            <Send className="h-4 w-4" />
            <span className="sr-only">Send message</span>
          </Button>
        </div>
      </div>
    </div>
  );
}
