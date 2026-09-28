/**
 * OpenAI TTS Streaming Client
 * Real-time streaming TTS for presentation scripts
 */

// API base URL (same as api.ts)
// Note: If NEXT_PUBLIC_API_URL is not set, it defaults to '/api' which assumes Next.js rewrites
// For local development, set NEXT_PUBLIC_API_URL=http://localhost:8000 in .env.local
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || '';

export interface TTSChunk {
  type: "chunk" | "complete" | "error";
  index?: number;
  text?: string;
  timestamp?: number;
  audio_base64?: string;
  actions?: any[];
  message?: string;
}

export interface TTSStreamOptions {
  voice?: "alloy" | "echo" | "fable" | "onyx" | "nova" | "shimmer" | "ash" | "ballad" | "coral" | "sage" | "verse" | "marin" | "cedar";
  model?: "gpt-4o-mini-tts" | "tts-1" | "tts-1-hd";
  speed?: number;
  instructions?: string; // Instructions for voice tone, emotion, intonation, etc.
  onChunk?: (chunk: TTSChunk) => void;
  onSegmentStart?: (index: number, text: string) => void; // 세그먼트 재생 시작
  onSegmentEnd?: (index: number) => void; // 세그먼트 재생 종료
  onComplete?: () => void;
  onError?: (error: Error) => void;
}

export class OpenAITTSStream {
  private audioContext: AudioContext | null = null;
  private audioQueue: Array<{ audioData: ArrayBuffer; chunk: TTSChunk }> = [];
  private isPlaying = false;
  private isProcessingQueue = false; // 큐 처리 중 플래그
  private isStopped = false; // stream 중지 플래그
  private currentSource: AudioBufferSourceNode | null = null;
  private segmentStartTimes: Map<number, number> = new Map(); // 세그먼트별 재생 시작 시간
  private segmentDurations: Map<number, number> = new Map(); // 세그먼트별 오디오 길이
  private buffer: string = ""; // 불완전한 SSE 메시지를 저장하는 버퍼
  private reader: ReadableStreamDefaultReader<Uint8Array> | null = null; // reader 참조 저장
  private currentPlayingSegmentIndex: number = -1; // 현재 재생 중인 세그먼트 인덱스
  private audioCache: Map<number, ArrayBuffer> = new Map(); // 세그먼트별 오디오 캐시
  private segmentTextCache: Map<number, string> = new Map(); // 세그먼트별 텍스트 캐시
  private totalPlaybackTime: number = 0; // 실제 재생된 총 오디오 시간 (초)
  private playbackStartTime: number = 0; // 재생 시작 시점 (Date.now())
  private pausedPlaybackTime: number = 0; // pause 시점까지 재생된 시간 (초)
  private currentChunkStartTime: number = 0; // 현재 재생 중인 청크의 시작 시간 (Date.now())
  private currentChunkDuration: number = 0; // 현재 재생 중인 청크의 길이 (초)

  constructor() {
    if (typeof window !== "undefined") {
      this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    }
  }

  /**
   * Stream TTS audio chunks from backend
   */
  async streamChunks(
    segments: Array<{ text: string; timestamp: number; actions?: any[] }>,
    options: TTSStreamOptions = {}
  ): Promise<void> {
    const {
      voice = "alloy",
      model = "gpt-4o-mini-tts",
      speed = 1.0,
      instructions,
      onChunk,
      onSegmentStart,
      onSegmentEnd,
      onComplete,
      onError,
    } = options;

    try {
      const response = await fetch(`${API_BASE_URL}/api/chat/tts/stream-chunks`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          segments,
          voice,
          model,
          speed,
          instructions, // Voice tone, emotion, intonation instructions
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      this.reader = response.body?.getReader() || null; // reader 저장
      const decoder = new TextDecoder();

      if (!this.reader) {
        throw new Error("Response body is not readable");
      }

      // Reset buffer and stopped flag for new stream
      this.buffer = "";
      this.isStopped = false;
      this.currentPlayingSegmentIndex = -1;
      // audioCache와 segmentTextCache는 유지 (resume 시 재사용)
      // 재생 시간 추적 초기화 (새 스트림 시작 시)
      this.totalPlaybackTime = 0;
      this.playbackStartTime = Date.now();
      this.pausedPlaybackTime = 0;

      try {
        while (true) {
          if (this.isStopped) {
            console.log("[TTS] Stream stopped by user");
            break;
          }

          const { done, value } = await this.reader.read();
          if (done) {
            // 마지막 버퍼 처리 (중지되지 않은 경우에만)
            if (!this.isStopped && this.buffer.trim()) {
              this.processBuffer(this.buffer, onChunk, onSegmentStart, onSegmentEnd, onComplete, onError);
            }
            break;
          }

          // 중지 플래그 다시 확인
          if (this.isStopped) {
            break;
          }

          // 청크를 버퍼에 추가
          const chunk = decoder.decode(value, { stream: true });
          this.buffer += chunk;

          // 완전한 SSE 메시지 처리 (data: 로 시작하고 \n\n으로 끝나는 메시지)
          // SSE 형식: "data: {...}\n\n"
          const messages = this.buffer.split("\n\n");
          
          // 마지막 메시지는 아직 완전하지 않을 수 있으므로 버퍼에 유지
          this.buffer = messages.pop() || "";

          // 완전한 메시지들 처리
          for (const message of messages) {
            if (this.isStopped) break; // 중지 플래그 확인
            this.processBuffer(message, onChunk, onSegmentStart, onSegmentEnd, onComplete, onError);
          }
        }
      } catch (error) {
        // 중지로 인한 에러는 무시
        if (!this.isStopped) {
          console.error("TTS streaming error:", error);
          onError?.(error instanceof Error ? error : new Error(String(error)));
        }
      } finally {
        // reader 정리
        if (this.reader) {
          try {
            await this.reader.cancel();
          } catch (e) {
            // 이미 취소되었을 수 있음
          }
          this.reader = null;
        }
      }
    } catch (error) {
      // 초기 fetch 에러 처리
      if (!this.isStopped) {
        console.error("TTS streaming error:", error);
        onError?.(error instanceof Error ? error : new Error(String(error)));
      }
    }
  }

  /**
   * Process a complete SSE message buffer
   */
  private async processBuffer(
    buffer: string,
    onChunk?: (chunk: TTSChunk) => void,
    onSegmentStart?: (index: number, text: string) => void,
    onSegmentEnd?: (index: number) => void,
    onComplete?: () => void,
    onError?: (error: Error) => void
  ): Promise<void> {
    // 중지 플래그 확인 
    if(this.isStopped) {
      return;
    }
    
    const lines = buffer.split("\n");
    
    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          const jsonStr = line.slice(6).trim();
          if (!jsonStr) continue; // 빈 데이터 스킵
          
          const data: TTSChunk = JSON.parse(jsonStr);
          
          if (data.type === "chunk" && data.audio_base64) {
            // 중지 플래그 확인
            if (this.isStopped) break;
            
            // Convert base64 to ArrayBuffer
            const audioData = this.base64ToArrayBuffer(data.audio_base64);
            
            // 오디오 캐시에 저장
            if (data.index !== undefined) {
              // 오디오 캐시에 저장 (복사본 저장)
              this.audioCache.set(data.index, audioData.slice(0));
              
              // 텍스트 캐시에 저장
              if (data.text) {
                this.segmentTextCache.set(data.index, data.text);
              }
            }
            
            // onSegmentStart는 processAudioQueue에서 실제 재생 시작 시점에 호출됨
            
            if(!this.isStopped) {
              // 오디오를 큐에 추가하고 순차적으로 재생
              this.audioQueue.push({ audioData, chunk: data });
              console.log(`[TTS] Audio chunk 큐에 추가: index=${data.index}, queueLength=${this.audioQueue.length}`);
              
              // 큐 처리 시작 (이미 처리 중이면 스킵)
              if (!this.isProcessingQueue) {
                console.log(`[TTS] Audio queue 처리 시작`);
                this.processAudioQueue(onChunk, onSegmentStart, onSegmentEnd);
              } else {
                console.log(`[TTS] Audio queue 이미 처리 중, 새 청크는 큐에서 대기`);
              }
            }
          } else if (data.type === "complete") {
            // 중지 플래그 확인 - stop()으로 인한 complete는 무시
            if(!this.isStopped) {
              // Wait for remaining audio to finish
              await this.waitForQueueEmpty();
              
              // 모든 세그먼트 종료 이벤트
              this.segmentStartTimes.forEach((_, index) => {
                onSegmentEnd?.(index);
              });
              
              // onComplete는 PresentationPlayer에서 pause 상태를 확인하여 처리
              onComplete?.();
            } else {
              // stop()으로 인한 complete 신호는 무시
              console.log("[TTS] Complete signal ignored (stream was stopped)");
            }
          } else if (data.type === "error") {
            throw new Error(data.message || "TTS error");
          }
        } catch (e) {
          // JSON 파싱 에러는 로그만 남기고 계속 진행
          // (불완전한 메시지일 수 있으므로)
          if (e instanceof SyntaxError) {
            console.warn("Incomplete SSE message, will retry with next chunk:", e.message);
          } else {
            console.error("Error parsing SSE data:", e);
            onError?.(e instanceof Error ? e : new Error(String(e)));
          }
        }
      }
    }
  }

  /**
   * Process audio queue sequentially to prevent overlapping playback
   */
  private async processAudioQueue(
    onChunk?: (chunk: TTSChunk) => void,
    onSegmentStart?: (index: number, text: string) => void,
    onSegmentEnd?: (index: number) => void
  ): Promise<void> {
    if (this.isProcessingQueue) {
      console.log(`[TTS] processAudioQueue 이미 처리 중, 스킵`);
      return; // 이미 처리 중이면 리턴
    }

    this.isProcessingQueue = true;
    console.log(`[TTS] processAudioQueue 시작: queueLength=${this.audioQueue.length}`);

    while (this.audioQueue.length > 0) {
      // 중지 플래그 확인
      if (this.isStopped) {
        console.log(`[TTS] processAudioQueue 중지됨`);
        this.audioQueue = []; // 큐 비우기
        break;
      }

      const { audioData, chunk } = this.audioQueue.shift()!;
      const chunkSegmentIndex = chunk.index !== undefined ? chunk.index : -1;
      
      // 새로운 세그먼트 체크 (onSegmentStart는 실제 재생 시작 시 호출)
      let isNewSegment = false;
      let segmentText = "";
      if (chunkSegmentIndex >= 0 && chunkSegmentIndex !== this.currentPlayingSegmentIndex) {
        // 이전 세그먼트가 있으면 완료 처리
        if (this.currentPlayingSegmentIndex >= 0) {
          console.log(`[TTS] 세그먼트 ${this.currentPlayingSegmentIndex} 재생 완료`);
          onSegmentEnd?.(this.currentPlayingSegmentIndex);
        }
        isNewSegment = true;
        segmentText = chunk.text || "";
      }
      
      console.log(`[TTS] Audio chunk 재생 준비: index=${chunkSegmentIndex}, queueRemaining=${this.audioQueue.length}, isPlaying=${this.isPlaying}, hasCurrentSource=${!!this.currentSource}, currentPlayingSegment=${this.currentPlayingSegmentIndex}`);

      // 중지 플래그 확인
      if (this.isStopped) {
        console.log(`[TTS] Audio chunk 재생 중지 (shift 후)`);
        break;
      }
      
      // Play audio chunks sequentially (이전 오디오가 끝날 때까지 자동 대기)
      // 실제 재생 시작 시점에 onSegmentStart 호출
      const duration = await this.playAudioChunk(audioData, () => {
        if (isNewSegment) {
          this.currentPlayingSegmentIndex = chunkSegmentIndex;
          this.segmentStartTimes.set(chunkSegmentIndex, Date.now());
          console.log(`[TTS] 새 세그먼트 ${chunkSegmentIndex} 실제 재생 시작: text=${segmentText.substring(0, 50)}...`);
          onSegmentStart?.(chunkSegmentIndex, segmentText);
        }
      });
      console.log(`[TTS] Audio chunk 재생 완료: index=${chunkSegmentIndex}, duration=${duration}s`);

      // 세그먼트 오디오 길이 저장 (백엔드가 세그먼트당 1개 청크만 보내므로 누적 불필요)
      if (chunkSegmentIndex >= 0 && duration) {
        this.segmentDurations.set(chunkSegmentIndex, duration);
      }
      
      // 세그먼트 재생 완료 처리 (백엔드가 세그먼트당 1개 청크만 보내므로 청크 재생 완료 = 세그먼트 완료)
      if (chunkSegmentIndex >= 0 && chunkSegmentIndex === this.currentPlayingSegmentIndex) {
        console.log(`[TTS] 세그먼트 ${chunkSegmentIndex} 재생 완료`);
        onSegmentEnd?.(chunkSegmentIndex);
      }
      
      // 중지 플래그 확인 (재생 후)
      if (this.isStopped) {
        console.log(`[TTS] Audio chunk 재생 후 중지됨`);
        break;
      }
      
      // Callback with chunk info (for UI updates) - 중지되지 않은 경우에만
      if (!this.isStopped) {
        onChunk?.(chunk);
      }
    }

    this.isProcessingQueue = false;
    console.log(`[TTS] processAudioQueue 완료: queueLength=${this.audioQueue.length}`);
  }

  /**
   * Play audio chunk sequentially
   * Returns a Promise that resolves when the audio playback completes
   * @param onPlaybackStarted Optional callback called when audio playback actually starts
   * @returns Promise that resolves with the duration of the audio chunk in seconds
   */
  private async playAudioChunk(
    audioData: ArrayBuffer,
    onPlaybackStarted?: () => void
  ): Promise<number> {
    // 중지 플래그 확인
    if (this.isStopped) {
      return 0;
    }
    
    if (!this.audioContext) {
      console.error("AudioContext not available");
      return 0;
    }

    // Wait for current audio to finish if playing (연속 재생을 위해 대기)
    // 이전 오디오가 완전히 끝날 때까지 Promise로 대기
    if (this.isPlaying && this.currentSource) {
      // 중지 플래그 확인 (대기 중에도)
      if (this.isStopped) {
        return 0;
      }
      
      // 현재 소스를 로컬 변수에 저장하여 race condition 방지
      const previousSource = this.currentSource;
      const originalOnEnded = previousSource.onended;
      
      await new Promise<void>((resolve) => {
        const timeout = setTimeout(() => {
          console.warn("[TTS] Audio playback timeout, forcing continue");
          resolve();
        }, 30000); // 최대 30초 대기 (충분히 긴 시간)
        
        // 이전 소스의 onended 이벤트에 추가 핸들러 연결
        // 원래 핸들러는 유지하면서 추가로 Promise를 resolve
        previousSource.onended = (event?: Event) => {
          // 원래 핸들러 먼저 실행 (this.isPlaying과 this.currentSource 업데이트)
          if (originalOnEnded) {
            try {
              if (event) {
                originalOnEnded.call(previousSource, event);
              } else {
                (originalOnEnded as (this: AudioBufferSourceNode, event: Event) => void).call(previousSource, {} as Event);
              }
            } catch (e) {
              console.warn("[TTS] Error in original onended handler:", e);
            }
          }
          
          // Promise resolve
          clearTimeout(timeout);
          resolve();
        };
      });
      
      // 이전 오디오가 끝난 후 상태 확인
      if (this.isStopped) {
        return 0;
      }
    }

    try {
      // 중지 플래그 확인
      if (this.isStopped) {
        return 0;
      }
      
      // AudioContext가 suspended 상태면 resume
      if (this.audioContext.state === "suspended") {
        await this.audioContext.resume();
      }

      // 중지 플래그 다시 확인
      if (this.isStopped) {
        return 0;
      }

      const audioBuffer = await this.audioContext.decodeAudioData(audioData.slice(0));
      
      // 중지 플래그 확인 (디코딩 후)
      if (this.isStopped) {
        return 0;
      }
      
      const source = this.audioContext.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(this.audioContext.destination);

      const duration = audioBuffer.duration; // 오디오 길이 (초)

      // 중지 플래그 최종 확인 (재생 시작 전)
      if (this.isStopped) {
        try {
          source.stop();
        } catch (e) {
          // 이미 중지되었을 수 있음
        }
        return 0;
      }

      // 오디오 재생이 완전히 끝날 때까지 Promise로 대기
      return new Promise<number>((resolve, reject) => {
        try {
          this.isPlaying = true;
          this.currentSource = source;
          this.currentChunkStartTime = Date.now();
          this.currentChunkDuration = duration;

          source.onended = () => {
            this.isPlaying = false;
            this.currentSource = null;
            // 실제 재생된 시간 누적
            const actualDuration = (Date.now() - this.currentChunkStartTime) / 1000;
            this.totalPlaybackTime += actualDuration;
            this.currentChunkStartTime = 0;
            this.currentChunkDuration = 0;
            resolve(duration);
          };

          source.start(0);
          
          // 실제 오디오 재생 시작 시점에 콜백 호출 (싱크 맞춤)
          onPlaybackStarted?.();
        } catch (error) {
          this.isPlaying = false;
          this.currentSource = null;
          this.currentChunkStartTime = 0;
          this.currentChunkDuration = 0;
          console.error("[TTS] Error starting audio playback:", error);
          reject(error);
        }
      });
    } catch (error) {
      console.error("Error playing audio chunk:", error);
      this.isPlaying = false;
      this.currentSource = null;
      return 0;
    }
  }

  /**
   * Wait for audio queue to be empty
   */
  private async waitForQueueEmpty(): Promise<void> {
    while (this.isPlaying) {
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
  }

  /**
   * Convert base64 string to ArrayBuffer
   */
  private base64ToArrayBuffer(base64: string): ArrayBuffer {
    const binaryString = atob(base64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes.buffer;
  }

  /**
   * Stop all audio playback
   */
  async stop(): Promise<void> {
    this.isStopped = true; // 먼저 플래그 설정
    this.audioQueue = [];
    this.isPlaying = false;
    this.isProcessingQueue = false;
    this.buffer = ""; // 버퍼 초기화
    this.segmentStartTimes.clear();
    this.segmentDurations.clear();
    this.currentPlayingSegmentIndex = -1;
    // audioCache와 segmentTextCache는 유지 (resume 시 재사용)
    // 재생 시간 추적 초기화
    this.totalPlaybackTime = 0;
    this.playbackStartTime = 0;
    this.pausedPlaybackTime = 0;
    this.currentChunkStartTime = 0;
    this.currentChunkDuration = 0;
    
    // 현재 재생 중인 오디오 소스 중지
    if (this.currentSource) {
      try {
        this.currentSource.stop();
      } catch (e) {
        // Source may already be stopped
      }
      this.currentSource = null;
    }
    
    // reader 취소 (백엔드 연결 끊기)
    if (this.reader) {
      try {
        await this.reader.cancel();
      } catch (e) {
        console.warn("[TTS] Error canceling reader:", e);
      }
      this.reader = null;
    }
    
    // AudioContext 일시 중지
    if (this.audioContext && this.audioContext.state !== "closed") {
      try {
        await this.audioContext.suspend();
      } catch (e) {
        console.warn("[TTS] Error suspending audio context:", e);
      }
    }
  }

  /**
   * Get segment duration in seconds
   */
  getSegmentDuration(index: number): number {
    return this.segmentDurations.get(index) || 0;
  }

  /**
   * Get current playing state
   */
  getIsPlaying(): boolean {
    return this.isPlaying;
  }

  /**
   * Play cached audio chunks sequentially
   * Used for resuming from pause without re-requesting TTS
   */
  async playCachedAudio(
    startIndex: number,
    endIndex: number,
    onSegmentStart?: (index: number, text: string) => void,
    onSegmentEnd?: (index: number) => void,
    onComplete?: () => void
  ): Promise<void> {
    console.log(`[TTS] playCachedAudio 시작: startIndex=${startIndex}, endIndex=${endIndex}`);
    
    // 중지 플래그 확인
    if (this.isStopped) {
      console.log(`[TTS] playCachedAudio 중지됨`);
      return;
    }
    
    if (!this.audioContext) {
      console.error("AudioContext not available");
      return;
    }
    
    this.isProcessingQueue = true;
    this.currentPlayingSegmentIndex = -1;
    
    try {
      for (let index = startIndex; index <= endIndex; index++) {
        // 중지 플래그 확인
        if (this.isStopped) {
          console.log(`[TTS] playCachedAudio 중지됨 (index=${index})`);
          break;
        }
        
        // 캐시에서 오디오 가져오기
        const audioData = this.audioCache.get(index);
        if (!audioData) {
          console.warn(`[TTS] 캐시된 오디오 없음: index=${index}, 스킵`);
          continue;
        }
        
        const text = this.segmentTextCache.get(index) || "";
        
        // 새로운 세그먼트 체크 (onSegmentStart는 실제 재생 시작 시 호출)
        let isNewSegment = false;
        if (index !== this.currentPlayingSegmentIndex) {
          // 이전 세그먼트가 있으면 완료 처리
          if (this.currentPlayingSegmentIndex >= 0) {
            console.log(`[TTS] 세그먼트 ${this.currentPlayingSegmentIndex} 재생 완료`);
            onSegmentEnd?.(this.currentPlayingSegmentIndex);
          }
          isNewSegment = true;
        }
        
        // 오디오 재생 (실제 재생 시작 시점에 onSegmentStart 호출)
        const duration = await this.playAudioChunk(audioData, () => {
          if (isNewSegment) {
            this.currentPlayingSegmentIndex = index;
            this.segmentStartTimes.set(index, Date.now());
            console.log(`[TTS] 캐시된 세그먼트 ${index} 실제 재생 시작: text=${text.substring(0, 50)}...`);
            onSegmentStart?.(index, text);
          }
        });
        console.log(`[TTS] 캐시된 오디오 재생 완료: index=${index}, duration=${duration}s`);
        
        // 세그먼트 오디오 길이 누적
        if (duration) {
          const currentDuration = this.segmentDurations.get(index) || 0;
          this.segmentDurations.set(index, currentDuration + duration);
        }
      }
      
      // 마지막 세그먼트 종료 처리
      if (this.currentPlayingSegmentIndex >= 0 && !this.isStopped) {
        console.log(`[TTS] 마지막 캐시된 세그먼트 ${this.currentPlayingSegmentIndex} 재생 완료`);
        onSegmentEnd?.(this.currentPlayingSegmentIndex);
      }
      
      // 완료 콜백 호출
      if (!this.isStopped) {
        onComplete?.();
      }
    } finally {
      this.isProcessingQueue = false;
      console.log(`[TTS] playCachedAudio 완료`);
    }
  }

  /**
   * Check if audio is cached for given segment indices
   */
  hasCachedAudio(startIndex: number, endIndex: number): boolean {
    for (let index = startIndex; index <= endIndex; index++) {
      if (!this.audioCache.has(index)) {
        return false;
      }
    }
    return true;
  }

  /**
   * Get cached audio for a segment
   */
  getCachedAudio(segmentIndex: number): ArrayBuffer | null {
    return this.audioCache.get(segmentIndex) || null;
  }

  /**
   * Clear audio cache
   */
  clearCache(): void {
    this.audioCache.clear();
    this.segmentTextCache.clear();
    console.log("[TTS] 오디오 캐시 정리됨");
  }

  /**
   * Get total playback time in seconds (actual audio playback time)
   */
  getTotalPlaybackTime(): number {
    if (this.isPlaying && this.currentSource && this.currentChunkStartTime > 0) {
      // 현재 재생 중인 청크의 경과 시간 계산
      const currentChunkElapsed = (Date.now() - this.currentChunkStartTime) / 1000;
      // 현재 청크의 경과 시간이 청크 길이를 초과하지 않도록 제한
      const clampedElapsed = Math.min(currentChunkElapsed, this.currentChunkDuration);
      return this.totalPlaybackTime + clampedElapsed;
    }
    return this.totalPlaybackTime;
  }

  /**
   * Pause audio playback (preserve playback time)
   */
  pause(): void {
    if (this.isPlaying) {
      // 현재까지 재생된 시간 저장 (현재 청크의 경과 시간 포함)
      this.pausedPlaybackTime = this.getTotalPlaybackTime();
      this.totalPlaybackTime = this.pausedPlaybackTime;
    }
    this.isStopped = true;
    // 현재 재생 중인 오디오 중지
    if (this.currentSource) {
      try {
        this.currentSource.stop();
      } catch (e) {
        // Source may already be stopped
      }
      this.currentSource = null;
    }
    this.isPlaying = false;
    this.currentChunkStartTime = 0;
    this.currentChunkDuration = 0;
  }

  /**
   * Resume audio playback (reset stopped flag)
   */
  resume(): void {
    this.isStopped = false;
    // 재생 시작 시간 업데이트 (pause 시점의 시간 유지)
    this.playbackStartTime = Date.now();
    // totalPlaybackTime은 pausedPlaybackTime 유지
    this.totalPlaybackTime = this.pausedPlaybackTime;
  }
}


