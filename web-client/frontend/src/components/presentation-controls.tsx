"use client";

import { Button } from "@/components/ui/button";
import { useViewerStore } from "@/lib/state";
import { Pause, Play, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";

export function PresentationControls() {
  const { presentationState, startPresentation, stopPresentation, pausePresentation, resumePresentation, getViewerByVehicleId } = useViewerStore();
  const [progress, setProgress] = useState(0);
  const [displayTime, setDisplayTime] = useState(0);
  const lastTimestampRef = useRef(0);
  const animationFrameRef = useRef<number | null>(null);
  
  // Check if model is loaded for the current script's vehicle
  const isModelLoaded = (() => {
    const vehicleId = presentationState.currentScript?.vehicle_id;
    if (!vehicleId) return true; // If no vehicle_id, assume loaded (backward compatibility)
    const viewer = getViewerByVehicleId(vehicleId);
    const loaded = viewer?.isModelLoaded ?? false;
    return loaded;
  })();

  // Update progress and timer based on current timestamp and pause state
  useEffect(() => {
    if (!presentationState.currentScript) {
      setProgress(0);
      setDisplayTime(0);
      lastTimestampRef.current = 0;
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
      return;
    }

    const totalDuration = presentationState.currentScript.total_duration || 0;
    const currentTime = presentationState.currentTimestamp || 0;
    const isPlaying = presentationState.isPlaying;
    const isPaused = presentationState.isPaused;

    // total_duration이 0이거나 없으면 현재 시간 + 여유 시간(5초)로 추정
    const effectiveTotalDuration = totalDuration > 0 ? totalDuration : (currentTime + 5);
    
    // Progress 계산
    const progressPercent = effectiveTotalDuration > 0 ? (currentTime / effectiveTotalDuration) * 100 : 0;
    setProgress(Math.min(100, Math.max(0, progressPercent))); // Clamp between 0-100

    // Display time 업데이트
    setDisplayTime(currentTime);
    lastTimestampRef.current = currentTime;

    // pause 중이 아닐 때만 애니메이션 프레임으로 부드럽게 업데이트
    if (isPlaying && !isPaused && animationFrameRef.current === null) {
      const updateDisplay = () => {
        // timestamp가 업데이트되면 displayTime도 업데이트
        if (presentationState.currentTimestamp !== lastTimestampRef.current) {
          setDisplayTime(presentationState.currentTimestamp);
          lastTimestampRef.current = presentationState.currentTimestamp;
        }
        animationFrameRef.current = requestAnimationFrame(updateDisplay);
      };
      animationFrameRef.current = requestAnimationFrame(updateDisplay);
    } else if ((!isPlaying || isPaused) && animationFrameRef.current !== null) {
      // pause 또는 stop 시 애니메이션 프레임 정리
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
    };
  }, [presentationState.currentTimestamp, presentationState.isPlaying, presentationState.isPaused, presentationState.currentScript]);

  // Don't render if no presentation is active
  if (!presentationState.currentScript) {
    return null;
  }

  // Validate script structure
  const script = presentationState.currentScript;
  if (!script.segments || !Array.isArray(script.segments)) {
    console.error("[PresentationControls] Invalid script structure: segments missing or not an array");
    return null;
  }

  // Validate segment index
  const segmentIndex = presentationState.currentSegmentIndex || 0;
  if (segmentIndex < 0 || segmentIndex >= script.segments.length) {
    console.warn("[PresentationControls] Invalid segment index:", segmentIndex, "max:", script.segments.length - 1);
    return null;
  }

  const currentSegment = script.segments[segmentIndex];
  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  // Use total_duration from script, or estimate from current time if not available
  const scriptTotalDuration = presentationState.currentScript?.total_duration || 0;
  const currentTime = displayTime;
  const totalTime = scriptTotalDuration > 0 ? scriptTotalDuration : (currentTime + 5); // Fallback: current time + 5 seconds

  return (
    <div className="fixed bottom-4 left-1/2 -translate-x-1/2 z-50 w-full max-w-2xl px-4">
      <div className="bg-card border border-border rounded-lg shadow-lg p-4 space-y-3 relative">
        {/* Close Button */}
        <Button
          variant="ghost"
          size="icon"
          className="absolute top-2 right-2 h-6 w-6 text-muted-foreground hover:text-foreground"
          onClick={stopPresentation}
          title="닫기"
        >
          <X className="h-4 w-4" />
        </Button>
        
        {/* Progress Bar */}
        <div className="space-y-1">
          <div className="flex justify-between items-center text-xs text-muted-foreground px-4">
            <span className={presentationState.isPaused ? "text-yellow-500 font-medium" : ""}>
              {presentationState.isPaused ? "일시정지" : ""}
            </span>
            <span>{formatTime(currentTime)}</span>
          </div>
          <div className="h-2 bg-muted rounded-full overflow-hidden">
            <div
              className={`h-full bg-primary transition-all duration-300 ease-out ${
                presentationState.isPaused ? "opacity-60" : ""
              }`}
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>

        {/* Current Reading Text */}
        <div className="text-sm text-foreground/80 bg-muted/50 rounded-md p-2 min-h-[2.5rem] flex items-center">
          {presentationState.currentReadingText ? (
            <p className="line-clamp-2 animate-pulse">
              <span className="font-semibold text-primary">읽는 중: </span>
              {presentationState.currentReadingText}
            </p>
          ) : !isModelLoaded && presentationState.isPlaying ? (
            <p className="line-clamp-2 text-yellow-600 dark:text-yellow-400">
              <span className="font-semibold">로딩 중: </span>
              3D 모델을 불러오는 중입니다...
            </p>
          ) : currentSegment ? (
            <p className="line-clamp-2 text-muted-foreground">{currentSegment.text}</p>
          ) : (
            <p className="line-clamp-2 text-muted-foreground">대기 중...</p>
          )}
        </div>

        {/* Controls */}
        <div className="flex items-center justify-center gap-2">
          {presentationState.isPlaying && !presentationState.isPaused ? (
            <Button
              variant="default"
              size="icon"
              onClick={pausePresentation}
              title="일시정지"
            >
              <Pause className="h-4 w-4" />
            </Button>
          ) : (
            <Button
              variant="default"
              size="icon"
              onClick={async () => {
                if (presentationState.isPaused) {
                  await resumePresentation();
                } else if (presentationState.currentScript) {
                  startPresentation(presentationState.currentScript);
                }
              }}
              title={!isModelLoaded ? "모델 로딩 중..." : "재생"}
              disabled={!isModelLoaded && !presentationState.isPaused}
            >
              <Play className="h-4 w-4" />
            </Button>
          )}

        </div>
      </div>
    </div>
  );
}

