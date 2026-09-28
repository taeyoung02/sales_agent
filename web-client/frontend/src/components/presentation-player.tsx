/**
 * Presentation Player for TTS-based vehicle presentations
 * Handles synchronization between TTS narration and 3D viewer controls
 */

import { PresentationScript, PresentationSegment, PresentationAction, VehicleViewer } from "@/lib/types";
import { useViewerStore } from "@/lib/state";
import { OpenAITTSStream } from "@/lib/openai-tts";
import { generateHeatmap, getCameraPoseData, getCameraPresets } from "@/lib/api";
import { calculateRotateCamera, getRotateDuration, getZoomLevelForShot, getZoomDuration, transformToPLY, getZoomLevelForRotateShot } from "@/lib/camera-utils";

export class PresentationPlayer {
  private script: PresentationScript; // 전체 script 저장
  private segments: PresentationSegment[];
  private currentSegmentIndex: number = 0;
  private executedActions: Set<string> = new Set(); // 실행된 액션 추적 (세그먼트 인덱스 + 액션 타입)
  private isPlaying: boolean = false;
  private isPaused: boolean = false;
  private ttsStream: OpenAITTSStream | null = null;
  private onSegmentChange?: (index: number) => void;
  private onComplete?: () => void;
  private onError?: (error: Error) => void;
  private startTime: number = 0; // 시작 시간 추적
  private pauseTime: number = 0; // pause 시점의 경과 시간 (초)
  private timestampUpdateInterval: NodeJS.Timeout | null = null; // 시간 업데이트 인터벌
  private onTimestampUpdate?: (timestamp: number) => void; // 시간 업데이트 콜백
  private onTextUpdate?: (text: string, index: number) => void; // 현재 읽고 있는 텍스트 콜백
  private segmentStartTime: number = 0; // 현재 세그먼트 시작 시간
  private streamingPromise: Promise<void> | null = null; // 스트리밍 Promise 추적

  constructor(
    script: PresentationScript,
    callbacks?: {
      onSegmentChange?: (index: number) => void;
      onComplete?: () => void;
      onError?: (error: Error) => void;
      onTimestampUpdate?: (timestamp: number) => void;
      onTextUpdate?: (text: string, index: number) => void; // 현재 읽고 있는 텍스트
    }
  ) {
    // Validate script structure
    if (!script || !script.segments || !Array.isArray(script.segments)) {
      const error = new Error("Invalid script: segments missing or not an array");
      console.error("[PresentationPlayer] Constructor error:", error, { script });
      throw error;
    }
    
    if (script.segments.length === 0) {
      const error = new Error("Invalid script: segments array is empty");
      console.error("[PresentationPlayer] Constructor error:", error);
      throw error;
    }
    
    // Validate each segment has required fields
    const invalidSegments = script.segments.filter(
      (seg, idx) => !seg || !seg.text || !seg.actions || !Array.isArray(seg.actions)
    );
    
    if (invalidSegments.length > 0) {
      console.warn("[PresentationPlayer] Some segments are invalid:", invalidSegments.length);
      // Filter out invalid segments instead of throwing
      this.segments = script.segments.filter(
        (seg) => seg && seg.text && seg.actions && Array.isArray(seg.actions)
      );
      
      if (this.segments.length === 0) {
        const error = new Error("Invalid script: all segments are invalid");
        console.error("[PresentationPlayer] Constructor error:", error);
        throw error;
      }
    } else {
      this.segments = script.segments;
    }
    
    this.script = script; // 전체 script 저장
    this.onSegmentChange = callbacks?.onSegmentChange;
    this.onComplete = callbacks?.onComplete;
    this.onError = callbacks?.onError;
    this.onTimestampUpdate = callbacks?.onTimestampUpdate;
    this.onTextUpdate = callbacks?.onTextUpdate;
    this.ttsStream = new OpenAITTSStream();
  }

  /**
   * Start playing the presentation
   */
  async play(): Promise<void> {
    if (this.isPlaying) {
      console.warn("Presentation is already playing");
      return;
    }

    // Wait for model to be loaded before starting TTS
    const store = useViewerStore.getState();
    const scriptVehicleId = this.script?.vehicle_id;
    
    if (scriptVehicleId) {
      console.log(`[PresentationPlayer] Waiting for model to load for vehicle: ${scriptVehicleId}`);
      const maxWaitTime = 30000; // 30 seconds max wait
      const checkInterval = 100; // Check every 100ms
      const startWaitTime = Date.now();
      
      while (Date.now() - startWaitTime < maxWaitTime) {
        const viewer = store.getViewerByVehicleId(scriptVehicleId);
        if (viewer?.isModelLoaded) {
          console.log(`[PresentationPlayer] Model loaded, starting presentation`);
          break;
        }
        await new Promise(resolve => setTimeout(resolve, checkInterval));
      }
      
      // Final check
      const finalViewer = store.getViewerByVehicleId(scriptVehicleId);
      if (!finalViewer?.isModelLoaded) {
        console.warn(`[PresentationPlayer] Model not loaded after ${maxWaitTime}ms, starting anyway`);
      }
    }

    this.isPlaying = true;
    this.isPaused = false;
    this.currentSegmentIndex = 0;
    this.startTime = Date.now(); // 시작 시간 기록

    try {
      // Start timestamp update interval
      this.startTimestampUpdates();

      // Start playing segments sequentially (actions will be executed on segment start)
      await this.playSegmentsSequentially();
    } catch (error) {
      this.isPlaying = false;
      this.stopTimestampUpdates();
      const err = error instanceof Error ? error : new Error(String(error));
      this.onError?.(err);
      throw err;
    }
  }

  /**
   * Stop the presentation
   */
  async stop(): Promise<void> {
    this.isPlaying = false;
    this.isPaused = false;
    this.pauseTime = 0;
    this.startTime = 0; // startTime 초기화

    // Clear executed actions tracking
    this.executedActions.clear();

    // Stop timestamp updates
    this.stopTimestampUpdates();

    // Stop TTS streaming (await to ensure complete cleanup)
    if (this.ttsStream) {
      try {
        await this.ttsStream.stop();
      } catch (error) {
        console.warn("[Presentation] Error stopping TTS stream:", error);
      }
    }
    
    // Wait for any pending streaming promise to complete
    if (this.streamingPromise) {
      try {
        await Promise.race([
          this.streamingPromise,
          new Promise((resolve) => setTimeout(resolve, 200)), // 최대 200ms 대기
        ]);
      } catch (error) {
        // Ignore errors from stopped stream
      }
      this.streamingPromise = null;
    }
  }

  /**
   * Pause the presentation
   */
  async pause(): Promise<void> {
    if (!this.isPlaying || this.isPaused) return;

    this.isPaused = true;
    
    // 실제 오디오 재생 시간 저장
    if (this.ttsStream) {
      this.pauseTime = this.ttsStream.getTotalPlaybackTime();
      // pause 시점의 최종 timestamp 업데이트
      this.onTimestampUpdate?.(this.pauseTime);
      // TTS 스트림 pause (재생 시간 보존)
      this.ttsStream.pause();
    } else {
      // TTS 스트림이 없으면 기존 방식 사용
      this.pauseTime = (Date.now() - this.startTime) / 1000;
      const finalElapsed = (Date.now() - this.startTime) / 1000;
      this.onTimestampUpdate?.(finalElapsed);
    }
    
    // 이전 스트리밍 Promise가 완료될 때까지 대기 (겹침 방지)
    if (this.streamingPromise) {
      try {
        // Promise가 완료되거나 거부될 때까지 대기
        await Promise.race([
          this.streamingPromise,
          new Promise((resolve) => setTimeout(resolve, 500)) // 최대 0.5초 대기
        ]);
      } catch (error) {
        // 에러는 무시 (이미 중지된 스트림)
        console.log("[Presentation] Previous stream cleanup:", error);
      }
      // streamingPromise는 null로 설정하지 않음 (resume 시 재사용)
    }
  }

  /**
   * Resume the presentation from where it was paused
   */
  async resume(): Promise<void> {
    if (!this.isPlaying || !this.isPaused) return;

    this.isPaused = false;
    
    // TTS 스트림 resume (재생 시간 보존)
    if (this.ttsStream) {
      this.ttsStream.resume();
      // resume 시 즉시 타임스탬프 업데이트 (UI 동기화)
      const currentElapsed = this.ttsStream.getTotalPlaybackTime();
      this.onTimestampUpdate?.(currentElapsed);
    } else {
      // TTS 스트림이 없으면 기존 방식 사용
      if (this.startTime === 0) {
        this.startTime = Date.now();
      } else {
        this.startTime = Date.now() - this.pauseTime * 1000;
      }
      const currentElapsed = (Date.now() - this.startTime) / 1000;
      this.onTimestampUpdate?.(currentElapsed);
    }
    
    // 타임스탬프 업데이트 재개
    this.startTimestampUpdates();
    
    // 현재 세그먼트부터 남은 세그먼트들을 다시 스트리밍
    await this.resumeFromCurrentSegment();
  }

  /**
   * Handle segment start callback (common logic for both streaming and cached playback)
   */
  private handleSegmentStart(index: number, text: string): void {
    // pause 상태에서는 콜백 처리 안 함
    if (this.isPaused) return;
    
    // 첫 번째 세그먼트가 실제로 재생 시작될 때 타이머 시작
    if (index === 0 && this.startTime === 0) {
      this.startTime = Date.now();
      // 첫 번째 세그먼트 재생 시작 시점에 타이머 시작
      this.startTimestampUpdates();
      console.log("[Presentation] 첫 번째 세그먼트 재생 시작, 타이머 시작");
    }
    
    // 세그먼트 재생 시작 시점에 텍스트 업데이트 및 actions 실행
    this.segmentStartTime = Date.now();
    this.currentSegmentIndex = index;
    this.onSegmentChange?.(index);
    
    // 실제 재생 시작 시점에 텍스트 업데이트 (싱크 맞춤)
    this.onTextUpdate?.(text, index);
    
    const segment = this.segments[index];
    if (segment && segment.actions) {
      // setCamera와 zoomCamera를 함께 처리하기 위해 먼저 액션들을 분석
      const setCameraAction = segment.actions.find(a => a.type === "setCamera" && !a.delay);
      const zoomCameraAction = segment.actions.find(a => a.type === "zoomCamera" && !a.delay);
      
      // setCamera와 zoomCamera가 모두 있고 delay가 없으면 함께 처리
      if (setCameraAction && zoomCameraAction && 
          setCameraAction.duration && zoomCameraAction.duration &&
          zoomCameraAction.level !== undefined) {
        // setCamera에 zoom을 포함시켜서 한 번에 실행
        const combinedAction: PresentationAction = {
          ...setCameraAction,
          level: zoomCameraAction.level,
          vehicle_id: setCameraAction.vehicle_id || zoomCameraAction.vehicle_id || this.script.vehicle_id,
        };
        
        const setCameraKey = `${index}-${segment.actions.indexOf(setCameraAction)}-${setCameraAction.type}`;
        const zoomCameraKey = `${index}-${segment.actions.indexOf(zoomCameraAction)}-${zoomCameraAction.type}`;
        
        // 두 액션 모두 실행된 것으로 표시
        if (!this.executedActions.has(setCameraKey) && !this.executedActions.has(zoomCameraKey)) {
          if (this.isPlaying && !this.isPaused) {
            console.log(`[Presentation] setCamera와 zoomCamera를 함께 처리: preset=${setCameraAction.preset}, zoom=${zoomCameraAction.level}`);
            this.executeAction(combinedAction);
            this.executedActions.add(setCameraKey);
            this.executedActions.add(zoomCameraKey);
          }
        }
      }
      
      // 나머지 actions는 기존 방식대로 처리
      segment.actions.forEach((action, actionIndex) => {
        const actionKey = `${index}-${actionIndex}-${action.type}`;
        
        // 이미 실행된 액션은 스킵
        if (this.executedActions.has(actionKey)) {
          return;
        }
        
        // setCamera와 zoomCamera가 함께 처리된 경우 스킵
        if ((action.type === "setCamera" || action.type === "zoomCamera") && 
            setCameraAction && zoomCameraAction && 
            !action.delay && 
            (action === setCameraAction || action === zoomCameraAction)) {
          return;
        }
        
        // 액션에 vehicle_id가 없으면 스크립트의 vehicle_id를 추가
        const actionWithVehicleId = {
          ...action,
          vehicle_id: action.vehicle_id || this.script.vehicle_id,
        };
        
        // delay가 있으면 지연 실행, 없으면 즉시 실행
        if (action.delay && action.delay > 0) {
          setTimeout(() => {
            if (this.isPlaying && !this.isPaused) {
              this.executeAction(actionWithVehicleId);
              this.executedActions.add(actionKey);
            }
          }, action.delay * 1000);
        } else {
          // 즉시 실행 (문장 시작과 동시에)
          if (this.isPlaying && !this.isPaused) {
            this.executeAction(actionWithVehicleId);
            this.executedActions.add(actionKey);
          }
        }
      });
    }
  }

  /**
   * Handle complete callback (common logic for both streaming and cached playback)
   */
  private handleComplete(): void {
    // pause 상태에서는 complete 신호 무시
    if (this.isPaused) {
      console.log("[Presentation] Complete signal ignored (paused)");
      return;
    }
    
    // Wait for remaining actions to complete
    const lastSegment = this.segments[this.segments.length - 1];
    const remainingTime = this.getRemainingTime(lastSegment);
    
    const finish = () => {
      this.stopTimestampUpdates();
      if (this.isPlaying && !this.isPaused) {
        this.isPlaying = false;
        this.streamingPromise = null;
        this.onComplete?.();
      }
    };
    
    if (remainingTime > 0) {
      setTimeout(finish, remainingTime);
    } else {
      finish();
    }
  }

  /**
   * Start timestamp updates based on actual audio playback time
   */
  private startTimestampUpdates(): void {
    this.stopTimestampUpdates(); // Clear any existing interval
    
    this.timestampUpdateInterval = setInterval(() => {
      if (this.isPlaying && !this.isPaused && this.ttsStream) {
        // 실제 오디오 재생 시간 사용
        const elapsed = this.ttsStream.getTotalPlaybackTime();
        this.onTimestampUpdate?.(elapsed);
      }
    }, 100); // Update every 100ms
  }

  /**
   * Stop timestamp updates
   */
  private stopTimestampUpdates(): void {
    if (this.timestampUpdateInterval) {
      clearInterval(this.timestampUpdateInterval);
      this.timestampUpdateInterval = null;
    }
  }

  /**
   * Play segments sequentially using OpenAI TTS streaming
   */
  private async playSegmentsSequentially(): Promise<void> {
    if (!this.ttsStream) {
      console.error("[TTS] TTS stream not initialized");
      this.isPlaying = false;
      return;
    }

    // Prepare segments for streaming
    // Note: timestamp is optional and not used in actual playback timing
    // Segments are played sequentially, and actual time is calculated at runtime
    const segments = this.segments.map((seg, idx) => ({
      text: seg.text,
      timestamp: seg.timestamp ?? 0, // Default to 0 if not provided (for backward compatibility)
      actions: seg.actions,
      index: idx,
    }));

    // Start TTS streaming
    try {
      this.streamingPromise = this.ttsStream.streamChunks(segments, {
        voice: "nova", // 일관된 음성 사용 (여성 목소리)
        model: "gpt-4o-mini-tts",
        speed: 1.0,
        onChunk: (chunk) => {
          // onChunk는 현재 사용되지 않음 (onSegmentStart에서 처리)
          // 타임스탬프는 timestampUpdateInterval에서 처리되므로 중복 제거
        },
        onSegmentStart: (index: number, text: string) => {
          this.handleSegmentStart(index, text);
        },
        onSegmentEnd: (index: number) => {
          // 세그먼트 종료 시 추가 처리 (필요한 경우)
          console.log(`[Presentation] Segment ${index} ended`);
        },
        onComplete: () => {
          this.handleComplete();
        },
        onError: (error) => {
          console.error("[TTS] Streaming error:", error);
          this.stopTimestampUpdates();
          this.isPlaying = false;
          this.streamingPromise = null;
          this.onError?.(error);
        },
      });
      await this.streamingPromise;
      this.streamingPromise = null;
    } catch (error) {
      console.error("[TTS] Failed to start streaming:", error);
      this.stopTimestampUpdates();
      this.isPlaying = false;
      this.streamingPromise = null;
      this.onError?.(error instanceof Error ? error : new Error(String(error)));
    }
  }

  /**
   * Resume streaming from current segment
   */
  private async resumeFromCurrentSegment(): Promise<void> {
    // TTS 스트림이 없으면 새로 생성
    if (!this.ttsStream) {
      this.ttsStream = new OpenAITTSStream();
    }

    // 마지막 세그먼트 인덱스
    const lastSegmentIndex = this.segments.length - 1;
    
    // 캐시된 오디오가 있는지 확인
    const hasCachedAudio = this.ttsStream.hasCachedAudio(this.currentSegmentIndex, lastSegmentIndex);
    
    if (hasCachedAudio) {
      console.log(`[Presentation] 캐시된 오디오 사용: ${this.currentSegmentIndex}~${lastSegmentIndex}`);
      // 이전 스트림이 있으면 완료 대기 (최소 시간만)
      if (this.streamingPromise) {
        try {
          await Promise.race([
            this.streamingPromise,
            new Promise((resolve) => setTimeout(resolve, 100)) // 최대 0.1초 대기
          ]);
        } catch (error) {
          // 에러는 무시
        }
      }
      
      // 캐시된 오디오로 즉시 재생
      try {
        this.streamingPromise = this.ttsStream.playCachedAudio(
          this.currentSegmentIndex,
          lastSegmentIndex,
          (index: number, text: string) => {
            this.handleSegmentStart(index, text);
          },
          (index: number) => {
            console.log(`[Presentation] Segment ${index} ended`);
          },
          () => {
            this.handleComplete();
          }
        );
        // await하지 않고 즉시 반환 (비동기로 실행)
        this.streamingPromise.catch((error) => {
          console.error("[TTS] Failed to play cached audio:", error);
          this.streamingPromise = null;
          this.onError?.(error instanceof Error ? error : new Error(String(error)));
        });
      } catch (error) {
        console.error("[TTS] Failed to start cached audio playback:", error);
        this.streamingPromise = null;
        this.onError?.(error instanceof Error ? error : new Error(String(error)));
      }
      return;
    }

    // 캐시가 없으면 기존처럼 스트리밍
    console.log(`[Presentation] 캐시 없음, 스트리밍 시작: ${this.currentSegmentIndex}~${lastSegmentIndex}`);
    
    // 이전 스트림이 있으면 완료 대기 (최소 시간만)
    if (this.streamingPromise) {
      try {
        await Promise.race([
          this.streamingPromise,
          new Promise((resolve) => setTimeout(resolve, 200)) // 최대 0.2초 대기
        ]);
      } catch (error) {
        // 에러는 무시
        console.log("[Presentation] Previous stream cleanup:", error);
      }
      this.streamingPromise = null;
    }
    
    // 현재 세그먼트부터 남은 세그먼트들만 준비
    const remainingSegments = this.segments.slice(this.currentSegmentIndex).map((seg, idx) => ({
      text: seg.text,
      timestamp: seg.timestamp ?? 0,
      actions: seg.actions,
      index: this.currentSegmentIndex + idx, // 원래 인덱스 유지
    }));

    if (remainingSegments.length === 0) {
      console.log("[Presentation] No remaining segments to resume");
      return;
    }

    // 남은 세그먼트들 스트리밍
    try {
      this.streamingPromise = this.ttsStream.streamChunks(remainingSegments, {
        voice: "nova", // 일관된 음성 사용 (여성 목소리)
        model: "gpt-4o-mini-tts",
        speed: 1.0,
        onChunk: (chunk) => {
          // onChunk는 현재 사용되지 않음 (onSegmentStart에서 처리)
          // 타임스탬프는 timestampUpdateInterval에서 처리되므로 중복 제거
        },
        onSegmentStart: (index: number, text: string) => {
          this.handleSegmentStart(index, text);
        },
        onSegmentEnd: (index: number) => {
          console.log(`[Presentation] Segment ${index} ended`);
        },
        onComplete: () => {
          this.handleComplete();
        },
        onError: (error) => {
          console.error("[TTS] Streaming error:", error);
          this.stopTimestampUpdates();
          this.isPlaying = false;
          this.streamingPromise = null;
          this.onError?.(error);
        },
      });
      // await하지 않고 즉시 반환 (비동기로 실행)
      this.streamingPromise.catch((error) => {
        console.error("[TTS] Failed to resume streaming:", error);
        this.streamingPromise = null;
        this.onError?.(error instanceof Error ? error : new Error(String(error)));
      });
    } catch (error) {
      console.error("[TTS] Failed to start resume streaming:", error);
      this.streamingPromise = null;
      this.onError?.(error instanceof Error ? error : new Error(String(error)));
    }
  }


  /**
   * Execute a single action
   */
  private async executeAction(action: PresentationAction): Promise<void> {
    console.log(`[Presentation] executeAction 호출:`, action);
    const store = useViewerStore.getState();

    // viewer_id 또는 vehicle_id로 대상 뷰어 찾기
    let targetViewer: VehicleViewer | undefined;
    
    if (action.viewer_id) {
      targetViewer = store.getViewer(action.viewer_id);
      console.log(`[Presentation] viewer_id로 뷰어 찾기: ${action.viewer_id}, found=${!!targetViewer}`);
    } else if (action.vehicle_id) {
      targetViewer = store.getViewerByVehicleId(action.vehicle_id);
      console.log(`[Presentation] vehicle_id로 뷰어 찾기: ${action.vehicle_id}, found=${!!targetViewer}`);
    }
    
    // viewer_id/vehicle_id가 없으면 첫 번째 뷰어에 적용 (하위 호환성)
    if (!targetViewer && store.viewers.length > 0) {
      targetViewer = store.viewers[0];
      console.log(`[Presentation] 첫 번째 뷰어 사용: ${targetViewer.id}`);
    }
    
    if (!targetViewer) {
      console.warn("[Presentation] No viewer found for action:", action, {
        viewersCount: store.viewers.length,
        availableViewers: store.viewers.map(v => ({ id: v.id, vehicleId: v.vehicleId })),
      });
      return;
    }
    
    console.log(`[Presentation] Target viewer: ${targetViewer.id}, vehicleId: ${targetViewer.vehicleId}`);

    try {
      switch (action.type) {
        case "setCamera":
          console.log(`[Presentation] setCamera 실행: preset=${action.preset}, duration=${action.duration}, position=${action.position}, level=${action.level}`);
          
          // Use CameraAnimator if duration is specified, otherwise use legacy method
          if (action.duration !== undefined && action.duration > 0) {
            if (action.preset) {
              // 뷰어별 제어 또는 전체 제어
              // level이 있으면 zoom과 함께 처리 (setCamera + zoomCamera 통합)
              if (targetViewer && (action.viewer_id || action.vehicle_id)) {
                console.log(`[Presentation] animateCameraToPresetForViewer 호출: viewerId=${targetViewer.id}, preset=${action.preset}, duration=${action.duration}, zoom=${action.level}`);
                store.animateCameraToPresetForViewer(targetViewer.id, action.preset, action.duration, action.level);
              } else {
                console.log(`[Presentation] animateCameraToPreset 호출: preset=${action.preset}, duration=${action.duration}, zoom=${action.level}`);
                store.animateCameraToPreset(action.preset, action.duration, action.level);
              }
            } else if (action.position) {
              if (targetViewer && (action.viewer_id || action.vehicle_id)) {
                console.log(`[Presentation] animateCameraToPositionForViewer 호출: viewerId=${targetViewer.id}, position=${action.position}, duration=${action.duration}, zoom=${action.level}`);
                store.animateCameraToPositionForViewer(targetViewer.id, action.position, action.duration, undefined, action.level);
              } else {
                console.log(`[Presentation] animateCameraToPosition 호출: position=${action.position}, duration=${action.duration}, zoom=${action.level}`);
                store.animateCameraToPosition(action.position, action.duration, undefined, action.level);
              }
            }
          } else {
            // Legacy method for immediate camera changes
            if (action.preset) {
              if (targetViewer && (action.viewer_id || action.vehicle_id)) {
                console.log(`[Presentation] setCameraForViewer 호출 (legacy): viewerId=${targetViewer.id}, preset=${action.preset}`);
                store.setCameraForViewer(targetViewer.id, action.preset);
              } else {
                console.log(`[Presentation] setCamera 호출 (legacy): preset=${action.preset}`);
                store.setCamera(action.preset);
              }
            }
            if (action.position) {
              if (targetViewer && (action.viewer_id || action.vehicle_id)) {
                console.log(`[Presentation] setCameraPositionForViewer 호출 (legacy): viewerId=${targetViewer.id}, position=${action.position}`);
                store.setCameraPositionForViewer(targetViewer.id, action.position);
              } else {
                console.log(`[Presentation] setCameraPosition 호출 (legacy): position=${action.position}`);
                store.setCameraPosition(action.position);
              }
            }
          }
          break;

        case "zoomCamera":
          console.log(`[Presentation] zoomCamera 실행: level=${action.level}, direction=${action.direction}, duration=${action.duration}, zoomParams=${JSON.stringify(action.zoom)}`);
          
          // 신규 파라미터: zoom: { shot, speed }
          if (action.zoom) {
            const { shot, speed } = action.zoom;
            const level = getZoomLevelForShot(shot);
            const duration = getZoomDuration(speed);
            if (targetViewer && (action.viewer_id || action.vehicle_id)) {
              console.log(`[Presentation] animateCameraZoomForViewer (shot/speed): viewerId=${targetViewer.id}, level=${level}, duration=${duration}`);
              store.animateCameraZoomForViewer(targetViewer.id, level, duration);
            } else {
              console.log(`[Presentation] animateCameraZoom (shot/speed): level=${level}, duration=${duration}`);
              store.animateCameraZoom(level, duration);
            }
          }
          // 기존 방식: level/direction/duration 그대로 유지 (호환)
          else if (action.duration !== undefined && action.duration > 0) {
            if (action.level !== undefined) {
              if (targetViewer && (action.viewer_id || action.vehicle_id)) {
                console.log(`[Presentation] animateCameraZoomForViewer 호출: viewerId=${targetViewer.id}, level=${action.level}, duration=${action.duration}`);
                store.animateCameraZoomForViewer(targetViewer.id, action.level, action.duration);
              } else {
                console.log(`[Presentation] animateCameraZoom 호출: level=${action.level}, duration=${action.duration}`);
                store.animateCameraZoom(action.level, action.duration);
              }
            } else if (action.direction) {
              // Convert direction to level
              const currentZoom = targetViewer ? targetViewer.cameraZoom : store.cameraZoom;
              const delta = action.direction === "in" ? 20 : -20;
              const newZoom = Math.max(0, Math.min(100, currentZoom + delta));
              if (targetViewer && (action.viewer_id || action.vehicle_id)) {
                console.log(`[Presentation] animateCameraZoomForViewer 호출 (direction): viewerId=${targetViewer.id}, newZoom=${newZoom}, duration=${action.duration}`);
                store.animateCameraZoomForViewer(targetViewer.id, newZoom, action.duration);
              } else {
                console.log(`[Presentation] animateCameraZoom 호출 (direction): newZoom=${newZoom}, duration=${action.duration}`);
                store.animateCameraZoom(newZoom, action.duration);
              }
            }
          } else {
            // Legacy method
            if (action.level !== undefined) {
              if (targetViewer && (action.viewer_id || action.vehicle_id)) {
                console.log(`[Presentation] setCameraZoomForViewer 호출 (legacy): viewerId=${targetViewer.id}, level=${action.level}`);
                store.setCameraZoomForViewer(targetViewer.id, action.level);
              } else {
                console.log(`[Presentation] setCameraZoom 호출 (legacy): level=${action.level}`);
                store.setCameraZoom(action.level);
              }
            } else if (action.direction) {
              console.log(`[Presentation] zoomCamera 호출 (legacy): direction=${action.direction}`);
              store.zoomCamera(action.direction);
            }
          }
          break;

        case "rotateCamera":
          // 새로운 rotateCamera 파라미터 방식
          if (action.rotate) {
            const { target, shot = "full", angle, speed } = action.rotate;
            const duration = getRotateDuration(speed);
            
            // 차량 데이터 로드
            const vehicleId = targetViewer.vehicleId;
            if (!vehicleId) {
              console.error("[Presentation] rotateCamera: vehicleId not found");
              break;
            }
            
            try {
              const cameraPoseData = await getCameraPoseData(vehicleId);
              if (!cameraPoseData || !cameraPoseData.center || !cameraPoseData.forward || !cameraPoseData.side || !cameraPoseData.up) {
                console.error("[Presentation] rotateCamera: Failed to load camera pose data");
                break;
              }
              
              // JSON 프리셋의 camera_position을 높이 범위로 사용
              const cameraPresets = await getCameraPresets(vehicleId);
              const presetLowPos = cameraPresets?.[target]?.camera_position as [number, number, number] | undefined;
              const presetTopPos = cameraPresets?.top?.camera_position as [number, number, number] | undefined;
              
              // 카메라 위치 계산 (Camera Presets와 동일한 방식)
              // bbox를 전달하여 optimal distance 계산
              const { position, targetPoint } = calculateRotateCamera(
                cameraPoseData.center,
                cameraPoseData.forward,
                cameraPoseData.side,
                cameraPoseData.up,
                target,
                shot,
                angle,
                cameraPoseData.bbox_local,
                presetLowPos,
                presetTopPos
              );
              
              // PLY 좌표로 변환
              const plyPosition = transformToPLY(position);
              const plyTarget = transformToPLY(targetPoint);
              const plyCenter = transformToPLY(cameraPoseData.center);
              
              console.log(`[Presentation] rotateCamera: target=${target}, shot=${shot}, angle=${angle}, speed=${speed}, duration=${duration}s, viewerId=${targetViewer.id}`);
              
              // 카메라 애니메이션 실행 (반경/줌을 함께 보간)
              const rotateShotZoom = getZoomLevelForRotateShot(shot);
              
              // 뷰어별 제어 또는 전체 제어
              if (targetViewer && (action.viewer_id || action.vehicle_id)) {
                console.log(`[Presentation] animateCameraToPositionForViewer 호출: viewerId=${targetViewer.id}, position=${plyPosition}, duration=${duration}, zoom=${rotateShotZoom}`);
                store.animateCameraToPositionForViewer(targetViewer.id, plyPosition, duration, plyCenter, rotateShotZoom);
                // 뷰어별 target 설정
                useViewerStore.setState((state) => ({
                  viewers: state.viewers.map((v) =>
                    v.id === targetViewer.id ? { ...v, cameraTarget: plyTarget } : v
                  ),
                  cameraTarget: plyTarget,
                }));
              } else {
                console.log(`[Presentation] animateCameraToPosition 호출: position=${plyPosition}, duration=${duration}, zoom=${rotateShotZoom}`);
                store.animateCameraToPosition(plyPosition, duration, plyCenter, rotateShotZoom);
                store.setCameraTarget(plyTarget);
              }
            } catch (error) {
              console.error("[Presentation] rotateCamera error:", error);
            }
          }
          // 레거시 호환성: 기존 target 방식 유지
          else if (action.target) {
            const duration = action.duration || 2.0;
            // rotateCamera는 현재 뷰어별 제어 미지원 (향후 확장 가능)
            store.animateCameraRotation(action.target, duration);
          }
          break;

        case "generateHeatmap":
          if (action.query) {
            try {
              // Get vehicle_id from action, targetViewer, or script
              const vehicleId = action.vehicle_id || targetViewer?.vehicleId || this.script?.vehicle_id;
              
              const heatmapResult = await generateHeatmap({
                query: action.query,
                vehicle_id: vehicleId,
              });

              // Move camera using calculated position from backend (preset system)
              if (heatmapResult.camera_position && heatmapResult.surface_center) {
                // Transform from JSON coordinates to PLY coordinates
                const plyCameraPos = transformToPLY(heatmapResult.camera_position as [number, number, number]);
                const plySurfaceCenter = transformToPLY(heatmapResult.surface_center as [number, number, number]);
                
                const duration = action.duration || 2.0; // Default duration if not specified
                
                // Use animation if duration is specified, otherwise immediate
                if (duration > 0) {
                  // Use orbit center for smooth arc movement
                  const plyCenter = targetViewer?.vehicleId 
                    ? (await getCameraPoseData(targetViewer.vehicleId))?.center 
                    : null;
                  const orbitCenter = plyCenter ? transformToPLY(plyCenter) : undefined;
                  
                  if (targetViewer && (action.viewer_id || action.vehicle_id)) {
                    store.animateCameraToPositionForViewer(targetViewer.id, plyCameraPos, duration, orbitCenter);
                    // Update viewer's camera target directly
                    useViewerStore.setState((state) => ({
                      viewers: state.viewers.map((v) =>
                        v.id === targetViewer.id ? { ...v, cameraTarget: plySurfaceCenter } : v
                      ),
                      cameraTarget: plySurfaceCenter,
                    }));
                  } else {
                    store.animateCameraToPosition(plyCameraPos, duration, orbitCenter);
                    store.setCameraTarget(plySurfaceCenter);
                  }
                } else {
                  // Immediate camera movement
                  if (targetViewer && (action.viewer_id || action.vehicle_id)) {
                    store.setCameraPositionForViewer(targetViewer.id, plyCameraPos);
                    // Update viewer's camera target directly
                    useViewerStore.setState((state) => ({
                      viewers: state.viewers.map((v) =>
                        v.id === targetViewer.id ? { ...v, cameraTarget: plySurfaceCenter } : v
                      ),
                      cameraTarget: plySurfaceCenter,
                    }));
                  } else {
                    store.setCameraPosition(plyCameraPos);
                    store.setCameraTarget(plySurfaceCenter);
                  }
                }
                
                // ✅ up vector도 설정 (있으면)
                if (heatmapResult.camera_up) {
                  const plyUpVector = transformToPLY(heatmapResult.camera_up as [number, number, number]);
                  if (targetViewer && (action.viewer_id || action.vehicle_id)) {
                    // Update viewer's camera up directly
                    useViewerStore.setState((state) => ({
                      viewers: state.viewers.map((v) =>
                        v.id === targetViewer.id ? { ...v, cameraUp: plyUpVector } : v
                      ),
                      cameraUp: plyUpVector,
                    }));
                  } else {
                    store.setCameraUp(plyUpVector);
                  }
                }
                
                if (heatmapResult.surface_normal) {
                  const plyNormal = transformToPLY(heatmapResult.surface_normal as [number, number, number]);
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
                const cameraPos: [number, number, number] = [cameraX, cameraY, cameraZ];
                
                const duration = action.duration || 2.0;
                
                if (duration > 0) {
                  if (targetViewer && (action.viewer_id || action.vehicle_id)) {
                    store.animateCameraToPositionForViewer(targetViewer.id, cameraPos, duration);
                    // Update viewer's camera target directly
                    useViewerStore.setState((state) => ({
                      viewers: state.viewers.map((v) =>
                        v.id === targetViewer.id ? { ...v, cameraTarget: plyTargetPos } : v
                      ),
                      cameraTarget: plyTargetPos,
                    }));
                  } else {
                    store.animateCameraToPosition(cameraPos, duration);
                    store.setCameraTarget(plyTargetPos);
                  }
                } else {
                  if (targetViewer && (action.viewer_id || action.vehicle_id)) {
                    store.setCameraPositionForViewer(targetViewer.id, cameraPos);
                    // Update viewer's camera target directly
                    useViewerStore.setState((state) => ({
                      viewers: state.viewers.map((v) =>
                        v.id === targetViewer.id ? { ...v, cameraTarget: plyTargetPos } : v
                      ),
                      cameraTarget: plyTargetPos,
                    }));
                  } else {
                    store.setCameraPosition(cameraPos);
                    store.setCameraTarget(plyTargetPos);
                  }
                }
              }
            } catch (error) {
              console.error("[Presentation] Heatmap generation error:", error);
            }
          }
          break;

        default:
          console.warn(`Unknown action type: ${action.type}`);
      }
    } catch (error) {
      console.error(`Error executing action ${action.type}:`, error);
    }
  }

  /**
   * Wait for specified milliseconds
   */
  private wait(ms: number): Promise<void> {
    return new Promise((resolve) => {
      if (!this.isPlaying) {
        resolve();
        return;
      }

      const startTime = Date.now();
      const checkInterval = setInterval(() => {
        if (!this.isPlaying) {
          clearInterval(checkInterval);
          resolve();
          return;
        }

        if (Date.now() - startTime >= ms) {
          clearInterval(checkInterval);
          resolve();
        }
      }, 100);
    });
  }

  /**
   * Get remaining time for last segment
   */
  private getRemainingTime(segment: PresentationSegment): number {
    let maxDuration = 0;
    segment.actions.forEach((action) => {
      const actionDuration = (action.duration || 0) * 1000;
      const actionDelay = (action.delay || 0) * 1000;
      const total = actionDuration + actionDelay;
      if (total > maxDuration) {
        maxDuration = total;
      }
    });
    return maxDuration;
  }

  /**
   * Get current segment index
   */
  getCurrentSegmentIndex(): number {
    return this.currentSegmentIndex;
  }

  /**
   * Get playing state
   */
  getIsPlaying(): boolean {
    return this.isPlaying;
  }

  /**
   * Get paused state
   */
  getIsPaused(): boolean {
    return this.isPaused;
  }
}

