/**
 * API client for backend communication
 */

import type { CameraPoseData } from "./camera-utils";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || '';

export interface ChatRequest {
  message: string;
  session_id?: string;
  conversation_history?: Array<{ role: string; content: string }>;
  vehicle_id?: string;  // Current loaded vehicle ID (primary vehicle if multiple are loaded)
  participant_id?: string;  // 설문 매칭/중복 사용자 처리용
}

export interface ChatResponse {
  response: string;
  tool_calls: Array<{
    name: string;
    arguments: Record<string, any>;
    result?: any;
  }>;
  evidence: Array<{
    title: string;
    snippet: string;
    asOf?: string;
    href?: string;
  }>;
}

export interface CameraControlRequest {
  preset?: string;
  position?: [number, number, number];
  zoom_direction?: 'in' | 'out';
  zoom_level?: number;
}

export interface CameraState {
  preset: string;
  position: [number, number, number];
  zoom: number;
}

export interface ConvertRequest {
  ply_path: string;
  output_path?: string;
  format?: 'gltf' | 'glb';
}

export interface ConvertResponse {
  success: boolean;
  output_path: string;
  message: string;
}

export interface GetSceneResponse {
  success: boolean;
  output_path: string;
  message: string;
}

export interface HeatmapRequest {
  query: string;
  cf3_path?: string;
  reference_path?: string;
  vehicle_id?: string;
}

export interface HeatmapResponse {
  success: boolean;
  output_path: string;
  message: string;
  target_position?: [number, number, number];
  surface_normal?: [number, number, number];
  camera_position?: [number, number, number];
  surface_center?: [number, number, number];
  camera_up?: [number, number, number];  // ✅ up vector 추가
}


export interface LoadDataResponse {
  status: string;
  service: string;
}

export interface LoadDataRequest {
  clear_existing: boolean;
}

export async function loadData(request: LoadDataRequest): Promise<LoadDataResponse> {
    try {
      const response = await fetch(`${API_BASE_URL}/api/data/load`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(error.detail || `HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Load data error:', error);
    throw error;
  }
}


/**
 * Send chat message to backend
 */
export async function sendChatMessage(
  request: ChatRequest
): Promise<ChatResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
      credentials: 'include', // 쿠키를 포함하여 전송 (세션 ID 포함)
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(error.detail || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Chat API error:', error);
    throw error;
  }
}

/**
 * Streaming chat message types
 */
export interface ChatStreamEvent {
  type: 'stage' | 'dialogue_chunk' | 'tool_calls' | 'evidence' | 'complete' | 'error';
  stage?: 'planner' | 'evaluator' | 'executor';
  status?: 'started' | 'completed' | 'skipped';
  reason?: string;
  plans_count?: number;
  text?: string;
  accumulated?: string;
  tool_calls?: Array<{
    name: string;
    arguments: Record<string, any>;
    result?: any;
  }>;
  evidence?: Array<{
    title: string;
    snippet: string;
    asOf?: string;
    href?: string;
  }>;
  message?: string;
}

/**
 * Vanilla chat request interface (simplified, no vehicle_id)
 */
export interface VanillaChatRequest {
  message: string;
  session_id?: string;
  conversation_history?: Array<{ role: string; content: string }>;
  participant_id?: string;
}

/**
 * Vanilla chat stream event types
 */
export interface VanillaChatStreamEvent {
  type: 'stage' | 'dialogue_chunk' | 'evidence' | 'complete' | 'error';
  stage?: 'rag_search' | 'response_generation';
  status?: 'started' | 'completed' | 'skipped';
  text?: string;
  accumulated?: string;
  evidence?: Array<{
    title: string;
    snippet: string;
    asOf?: string;
    href?: string;
  }>;
  message?: string;
}

/**
 * Send vanilla chat message with streaming dialogue (RAG-only chatbot)
 * 
 * @param request Vanilla chat request
 * @param onDialogueChunk Callback for each dialogue chunk
 * @param onStageUpdate Callback for stage updates
 * @param onEvidence Callback when evidence is received
 * @param onComplete Callback when streaming is complete
 * @param onError Callback for errors
 */
export async function sendVanillaChatMessageStream(
  request: VanillaChatRequest,
  callbacks: {
    onDialogueChunk?: (text: string, accumulated: string) => void;
    onStageUpdate?: (stage: string, status: string, data?: any) => void;
    onEvidence?: (evidence: Array<any>) => void;
    onComplete?: () => void;
    onError?: (error: Error) => void;
  }
): Promise<void> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/vanilla-chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
      credentials: 'include', // 쿠키를 포함하여 전송 (세션 ID 포함)
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(error.detail || `HTTP error! status: ${response.status}`);
    }

    if (!response.body) {
      throw new Error('Response body is null');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      
      // 마지막 줄은 완전하지 않을 수 있으므로 버퍼에 보관
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const jsonStr = line.slice(6).trim();
            if (!jsonStr) continue;

            const event: VanillaChatStreamEvent = JSON.parse(jsonStr);

            switch (event.type) {
              case 'stage':
                callbacks.onStageUpdate?.(
                  event.stage || 'unknown', 
                  event.status || 'unknown', 
                  {}
                );
                break;

              case 'dialogue_chunk':
                if (event.text && event.accumulated) {
                  callbacks.onDialogueChunk?.(event.text, event.accumulated);
                }
                break;

              case 'evidence':
                if (event.evidence) {
                  callbacks.onEvidence?.(event.evidence);
                }
                break;

              case 'complete':
                callbacks.onComplete?.();
                break;

              case 'error':
                callbacks.onError?.(new Error(event.message || 'Unknown error'));
                break;
            }
          } catch (parseError) {
            console.error('Failed to parse SSE event:', parseError, line);
          }
        }
      }
    }

    // 남은 버퍼 처리
    if (buffer.trim()) {
      const line = buffer.trim();
      if (line.startsWith('data: ')) {
        try {
          const jsonStr = line.slice(6).trim();
          if (jsonStr) {
            const event: VanillaChatStreamEvent = JSON.parse(jsonStr);
            if (event.type === 'complete') {
              callbacks.onComplete?.();
            }
          }
        } catch (parseError) {
          console.error('Failed to parse final SSE event:', parseError);
        }
      }
    }
  } catch (error) {
    console.error('Vanilla chat stream API error:', error);
    callbacks.onError?.(error instanceof Error ? error : new Error(String(error)));
    throw error;
  }
}

/**
 * Send chat message with streaming dialogue
 * 
 * @param request Chat request
 * @param onDialogueChunk Callback for each dialogue chunk (word)
 * @param onStageUpdate Callback for stage updates
 * @param onToolCalls Callback when tool calls are received
 * @param onEvidence Callback when evidence is received
 * @param onComplete Callback when streaming is complete
 * @param onError Callback for errors
 */
export async function sendChatMessageStream(
  request: ChatRequest,
  callbacks: {
    onDialogueChunk?: (text: string, accumulated: string) => void;
    onStageUpdate?: (stage: string, status: string, data?: any) => void;
    onToolCalls?: (toolCalls: Array<any>) => void;
    onEvidence?: (evidence: Array<any>) => void;
    onComplete?: () => void;
    onError?: (error: Error) => void;
  }
): Promise<void> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
      credentials: 'include', // 쿠키를 포함하여 전송 (세션 ID 포함)
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(error.detail || `HTTP error! status: ${response.status}`);
    }

    if (!response.body) {
      throw new Error('Response body is null');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      
      // 마지막 줄은 완전하지 않을 수 있으므로 버퍼에 보관
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const jsonStr = line.slice(6).trim();
            if (!jsonStr) continue;

            const event: ChatStreamEvent = JSON.parse(jsonStr);

            switch (event.type) {
              case 'stage':
                callbacks.onStageUpdate?.(
                  event.stage || 'unknown', 
                  event.status || 'unknown', 
                  {
                    reason: event.reason,
                    plans_count: event.plans_count,
                  }
                );
                break;

              case 'dialogue_chunk':
                if (event.text && event.accumulated) {
                  callbacks.onDialogueChunk?.(event.text, event.accumulated);
                }
                break;

              case 'tool_calls':
                if (event.tool_calls) {
                  callbacks.onToolCalls?.(event.tool_calls);
                }
                break;

              case 'evidence':
                if (event.evidence) {
                  callbacks.onEvidence?.(event.evidence);
                }
                break;

              case 'complete':
                callbacks.onComplete?.();
                break;

              case 'error':
                callbacks.onError?.(new Error(event.message || 'Unknown error'));
                break;
            }
          } catch (parseError) {
            console.error('Failed to parse SSE event:', parseError, line);
          }
        }
      }
    }

    // 남은 버퍼 처리
    if (buffer.trim()) {
      const line = buffer.trim();
      if (line.startsWith('data: ')) {
        try {
          const jsonStr = line.slice(6).trim();
          if (jsonStr) {
            const event: ChatStreamEvent = JSON.parse(jsonStr);
            if (event.type === 'complete') {
              callbacks.onComplete?.();
            }
          }
        } catch (parseError) {
          console.error('Failed to parse final SSE event:', parseError);
        }
      }
    }
  } catch (error) {
    console.error('Chat stream API error:', error);
    callbacks.onError?.(error instanceof Error ? error : new Error(String(error)));
    throw error;
  }
}

/**
 * Save session log to file
 */
export async function saveSessionLog(sessionId: string): Promise<{ success: boolean; filepath: string }> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/chat/session-log/${sessionId}/save`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(error.detail || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Save session log error:', error);
    throw error;
  }
}

/**
 * Health check
 * 
 * NOTE: API_BASE_URL을 사용하여 직접 백엔드로 요청 (다른 API와 일관성 유지)
 */
export async function healthCheck(): Promise<{ status: string; service: string }> {
  try {
    // 다른 API와 동일하게 API_BASE_URL 사용 (직접 백엔드로 요청)
    const response = await fetch(`${API_BASE_URL}/api/health`, {
      method: 'GET',
      credentials: 'include', // 쿠키를 포함하여 전송 (세션 ID 포함)
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Health check error:', error);
    throw error;
  }
}

/**
 * Get scene - returns glTF file path or converts PLY to glTF
 * @param inputPath Path to PLY file (will be converted) or glTF file (will be returned)
 * @param format Output format "gltf" or "glb" (default: "gltf")
 * @returns GetSceneResponse
 */
export async function getScene(
  inputPath: string,
  format: 'gltf' | 'glb' | 'ply' | 'ksplat' = 'ply'
): Promise<GetSceneResponse> {
  try {
    const params = new URLSearchParams({
      input_path: inputPath,
      format: format,
    });

    const response = await fetch(`${API_BASE_URL}/api/scene?${params.toString()}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(error.detail || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Get scene error:', error);
    throw error;
  }
}
/**
 * Convert PLY to KSplat
 */
export async function convertPLYToKSplat(request: ConvertRequest): Promise<ConvertResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/convert/ply-to-ksplat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(error.detail || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Convert PLY to KSplat error:', error);
    throw error;
  }
}


/**
 * Convert PLY to GLTF
 */
export async function convertPLYToGLTF(request: ConvertRequest): Promise<ConvertResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/convert/ply-to-gltf`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(error.detail || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Convert PLY to GLTF error:', error);
    throw error;
  }
}

/**
 * Generate heatmap PLY file based on text query
 * @param request Heatmap request with query and optional paths
 * @returns HeatmapResponse with output PLY file path
 */
export async function generateHeatmap(
  request: HeatmapRequest
): Promise<HeatmapResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/heatmap/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      let errorDetail = 'Unknown error';
      try {
        const errorData = await response.json();
        errorDetail = errorData.detail || errorData.message || `HTTP error! status: ${response.status}`;
      } catch (e) {
        // If response is not JSON, try to get text
        try {
          errorDetail = await response.text() || `HTTP error! status: ${response.status}`;
        } catch (e2) {
          errorDetail = `HTTP error! status: ${response.status}`;
        }
      }
      throw new Error(errorDetail);
    }

    return await response.json();
  } catch (error) {
    console.error('Generate heatmap error:', error);
    // Re-throw with more context if it's a network error
    if (error instanceof TypeError && error.message === 'Failed to fetch') {
      throw new Error('Unable to connect to the server. Please check if the backend is running.');
    }
    throw error;
  }
}

/**
 * 프롬프트 관련 타입 및 API
 */
export interface PromptsResponse {
  sales_knowledge: string;
  common_tool_guidelines: string;
  planner_tool_guidelines: string;
  planner_role: string;
  evaluator_role: string;
}

export interface PromptUpdateRequest {
  sales_knowledge?: string;
  common_tool_guidelines?: string;
  planner_tool_guidelines?: string;
  planner_role?: string;
  evaluator_role?: string;
}

export interface PromptUpdateResponse {
  status: string;
  message: string;
  updates: Record<string, string>;
}

/**
 * 모든 프롬프트 조회
 */
export async function getPrompts(): Promise<PromptsResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/prompts`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(error.detail || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Get prompts error:', error);
    throw error;
  }
}

/**
 * 프롬프트 업데이트
 */
export async function updatePrompts(
  request: PromptUpdateRequest
): Promise<PromptUpdateResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/prompts`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(error.detail || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Update prompts error:', error);
    throw error;
  }
}

/**
 * TTS 스트리밍 요청 타입
 */
export interface TTSStreamRequest {
  segments: Array<{
    text: string;
    timestamp: number;
    actions?: any[];
  }>;
  voice?: "alloy" | "echo" | "fable" | "onyx" | "nova" | "shimmer";
  model?: "gpt-4o-mini-tts" | "tts-1" | "tts-1-hd";
  speed?: number;
}

/**
 * TTS 스트리밍 엔드포인트 호출
 * Note: 실제 스트리밍은 OpenAITTSStream 클래스에서 처리
 */
export async function streamTTS(request: TTSStreamRequest): Promise<Response> {
  return fetch(`${API_BASE_URL}/api/chat/tts/stream-chunks`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  });
}

/**
 * Get initial camera position for a vehicle from JSON preset file
 * @param vehicleId - Vehicle ID to load camera preset for
 * @returns Initial camera position or default position if not found
 */
export async function getInitialCameraPosition(
  vehicleId: string
): Promise<[number, number, number] | null> {
  try {
    // Try to fetch the point_cloud_pruned_pose.json file
    // Use Cloudflare if available, otherwise use Next.js rewrites
    const staticBaseUrl = process.env.NEXT_PUBLIC_STATIC_BASE_URL;
    const url = staticBaseUrl
      ? `${staticBaseUrl}/static/source/${vehicleId}/point_cloud_pruned_pose.json`
      : `/static/source/${vehicleId}/point_cloud_pruned_pose.json`;
    
    const response = await fetch(url);
    
    if (!response.ok) {
      console.log(`No camera preset file found for ${vehicleId}`);
      return null;
    }
    
    const data = await response.json();
    
    // Check if there's an initial_camera_position field
    if (data.initial_camera_position && Array.isArray(data.initial_camera_position)) {
      const pos = data.initial_camera_position as [number, number, number];
      console.log(`📷 Loaded initial camera position for ${vehicleId}:`, pos);
      return pos;
    }
    
    // Fallback: use first query preset's camera_position if available
    if (data.query_presets) {
      const firstPreset = Object.values(data.query_presets)[0] as any;
      if (firstPreset?.camera_position) {
        const pos = firstPreset.camera_position as [number, number, number];
        console.log(`📷 Using first preset camera position for ${vehicleId}:`, pos);
        return pos;
      }
    }
    
    return null;
  } catch (error) {
    console.error(`Error loading camera preset for ${vehicleId}:`, error);
    return null;
  }
}

/**
 * Fetch camera pose data from JSON file
 * @param vehicleId - Vehicle ID to fetch camera pose data for
 * @returns CameraPoseData object or null if not found
 */
export async function getCameraPoseData(
  vehicleId: string
): Promise<CameraPoseData | null> {
  const data = await fetchPointCloudJson(vehicleId);
  if (!data) return null;

  console.log("[getCameraPoseData] Loaded via shared cache:", {
    hasCenter: !!data.center,
    hasForward: !!data.forward,
    hasUp: !!data.up,
    hasSide: !!data.side,
    hasBbox: !!data.bbox_local
  });

  return {
    center: data.center,
    forward: data.forward,
    up: data.up,
    side: data.side,
    bbox_local: data.bbox_local,
  };
}

// Shared in-memory cache for point_cloud_pruned_pose.json per vehicle
const pointCloudJsonCache: Record<string, Promise<any | null>> = {};

async function fetchPointCloudJson(
  vehicleId: string,
  forceReload: boolean = false
): Promise<any | null> {
  if (!forceReload && await pointCloudJsonCache[vehicleId]) {
    return pointCloudJsonCache[vehicleId];
  }

  const url = `${API_BASE_URL}/static/source/${vehicleId}/point_cloud_pruned_pose.json`;
  pointCloudJsonCache[vehicleId] = (async () => {
    try {
      console.log(`[PointCloudJSON] Fetching from: ${url}`);
      const response = await fetch(url);
      console.log(`[PointCloudJSON] Response status: ${response.status}`);
      if (!response.ok) {
        console.error("[PointCloudJSON] Failed to fetch - status:", response.status);
        return null;
      }
      const data = await response.json();
      return data;
    } catch (error) {
      console.error("[PointCloudJSON] Error fetching data:", error);
      return null;
    }
  })();

  try {
    return await pointCloudJsonCache[vehicleId];
  } catch (e) {
    delete pointCloudJsonCache[vehicleId];
    throw e;
  }
}

/**
 * Camera preset type
 */
export interface CameraPreset {
  camera_position: [number, number, number];
  target: [number, number, number];
  up?: [number, number, number];  // ✅ up vector 추가 (optional)
  description?: string;
}

/**
 * Get all camera presets for a vehicle
 * Auto-generates presets if they don't exist
 * @param vehicleId Vehicle ID to load presets for
 * @returns Dictionary of presets or null if not found
 */
export async function getCameraPresets(
  vehicleId: string
): Promise<Record<string, CameraPreset> | null> {
  try {
    const data = await fetchPointCloudJson(vehicleId);

    // Check if camera_presets exists
    if (data?.camera_presets && Object.keys(data.camera_presets).length > 0) {
      console.log("[getCameraPresets] Loaded presets:", Object.keys(data.camera_presets));
      return data.camera_presets;
    }
    
    // If no presets found, try to auto-generate via backend API
    // Use relative path to go through Next.js API route proxy (same as other API calls)
    try {
      const generateUrl = `/api/camera-presets/generate/${vehicleId}`;
      const generateResponse = await fetch(generateUrl, { method: 'POST' });
      
      if (generateResponse.ok) {
        console.log("[getCameraPresets] Auto-generation successful, refetching...");
        const refetchData = await fetchPointCloudJson(vehicleId, true);
        if (refetchData?.camera_presets) {
          console.log("[getCameraPresets] Loaded auto-generated presets:", Object.keys(refetchData.camera_presets));
          return refetchData.camera_presets;
        }
      } else {
        console.warn("[getCameraPresets] Auto-generation failed:", generateResponse.status);
      }
    } catch (genError) {
      console.warn("[getCameraPresets] Error during auto-generation:", genError);
    }
    
    return null;
  } catch (error) {
    console.error("[getCameraPresets] Error fetching camera presets:", error);
    return null;
  }
}

/**
 * Get a specific camera preset
 * @param vehicleId Vehicle ID
 * @param presetName Name of the preset (e.g., 'front', 'top', 'left')
 * @returns Camera preset or null if not found
 */
export async function getCameraPreset(
  vehicleId: string,
  presetName: string
): Promise<CameraPreset | null> {
  try {
    const presets = await getCameraPresets(vehicleId);
    
    if (!presets || !presets[presetName]) {
      return null;
    }
    
    return presets[presetName];
  } catch (error) {
    console.error("[getCameraPreset] Error fetching camera preset:", error);
    return null;
  }
}