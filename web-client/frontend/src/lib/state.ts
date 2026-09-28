import { create } from "zustand";
import { CameraPreset, VehicleViewer, ViewerState, PresentationScript } from "./types";
import { getCameraPresets } from "./api";
import { transformToPLY } from "./camera-utils";
// PresentationPlayer is imported dynamically to avoid circular dependency

export interface PresentationState {
  isPlaying: boolean;
  isPaused: boolean;
  currentScript: PresentationScript | null;
  currentSegmentIndex: number;
  currentTimestamp: number;
  currentReadingText: string; // 현재 읽고 있는 텍스트
  player: any | null; // PresentationPlayer - using any to avoid circular dependency
}

interface ViewerStore extends ViewerState {
  // 새로운 뷰어 관리 메서드
  addViewer: (vehicleId: string, sourcePath: string) => Promise<string>; // 뷰어 추가, ID 반환
  removeViewer: (viewerId: string) => void;
  updateViewer: (viewerId: string, updates: Partial<VehicleViewer>) => void;
  setShow3DView: (show: boolean) => void;
  getViewer: (viewerId: string) => VehicleViewer | undefined;
  getViewerByVehicleId: (vehicleId: string) => VehicleViewer | undefined;  
  // -------------------------
  setCamera: (preset: CameraPreset) => Promise<void>;
  setCameraPosition: (position: [number, number, number]) => void;
  setCameraTarget: (target: [number, number, number]) => void;
  setCameraUp: (up: [number, number, number]) => void;
  setCameraZoom: (zoom: number) => void;
  zoomCamera: (direction: "in" | "out") => void;
  // 뷰어별 카메라 제어 메서드
  setCameraForViewer: (viewerId: string, preset: CameraPreset) => Promise<void>;
  setCameraPositionForViewer: (viewerId: string, position: [number, number, number]) => void;
  setCameraZoomForViewer: (viewerId: string, zoom: number) => void;
  animateCameraToPresetForViewer: (viewerId: string, preset: CameraPreset, duration?: number, zoom?: number) => void;
  animateCameraToPositionForViewer: (viewerId: string, position: [number, number, number], duration?: number, center?: [number, number, number], zoom?: number) => void;
  animateCameraZoomForViewer: (viewerId: string, level: number, duration?: number) => void;
  // Camera animation with duration (for Presentation Script)
  cameraAnimationRequest: CameraAnimationRequest | null;
  animateCameraToPreset: (preset: CameraPreset, duration?: number, zoom?: number) => void;
  animateCameraToPosition: (position: [number, number, number], duration?: number, center?: [number, number, number], zoom?: number) => void;
  animateCameraZoom: (level: number, duration?: number) => void;
  animateCameraRotation: (target: [number, number, number], duration?: number) => void;
  clearCameraAnimationRequest: () => void;
  // -------------------------
  // Presentation 관련 상태 및 메서드
  presentationState: PresentationState;
  startPresentation: (script: PresentationScript) => void;
  stopPresentation: () => Promise<void>;
  pausePresentation: () => void;
  resumePresentation: () => void;
  updatePresentationState: (updates: Partial<PresentationState>) => void;
  // -------------------------
  reset: () => void;
}

const MAX_VIEWERS = 2;

const createDefaultViewer = (vehicleId: string | null = null, sourcePath: string | null = null): VehicleViewer => ({
  id: `viewer-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
  vehicleId,
  sourcePath,
  cameraPreset: "left",
  cameraZoom: 90,
  cameraPosition: [8, 2, 0],
});

const initialState: ViewerState = {
  viewers: [],
  show3DViewer: false,
  cameraPreset: "left",
  cameraZoom: 80,
  cameraPosition: [8, 2, 0],
  cameraTarget: [8, 2, 0],
  cameraUp: [8, 2, 0],
};

const initialPresentationState: PresentationState = {
  isPlaying: false,
  isPaused: false,
  currentScript: null,
  currentSegmentIndex: 0,
  currentTimestamp: 0,
  currentReadingText: "",
  player: null,
};

// Camera animation request for CameraAnimator
export interface CameraAnimationRequest {
  type: "preset" | "position" | "zoom" | "rotation";
  preset?: CameraPreset;
  position?: [number, number, number];
  zoom?: number;
  target?: [number, number, number];
  center?: [number, number, number]; // Optional: orbit center for arc movement
  duration?: number;
  timestamp: number; // For request deduplication
  viewerId?: string; // Optional: 특정 뷰어에만 적용
}

export const useViewerStore = create<ViewerStore>((set, get) => ({
  ...initialState,
  // 새로운 뷰어 관리 메서드
  addViewer: async (vehicleId: string, sourcePath: string) => {
    const state = get();
    
    // ✅ 해당 차량의 camera pose data 로드 (차량별 up vector 적용)
    let cameraData = {
      position: state.cameraPosition,
      target: state.cameraTarget,
      up: state.cameraUp
    };
    
    if (vehicleId) {
      try {
        const { getCameraPoseData } = await import('@/lib/api');
        const { calculateInitialCameraView } = await import('@/lib/camera-utils');
        const cameraPoseData = await getCameraPoseData(vehicleId);
        
        if (cameraPoseData) {
          const calculated = calculateInitialCameraView(cameraPoseData);
          cameraData = calculated;
          console.log(`📷 [addViewer] Loaded camera data for ${vehicleId}:`, {
            position: cameraData.position,
            target: cameraData.target,
            up: cameraData.up
          });
        } else {
          console.log(`📷 [addViewer] No camera pose data for ${vehicleId}, using current state`);
        }
      } catch (error) {
        console.warn(`⚠️ [addViewer] Failed to load camera pose for ${vehicleId}:`, error);
      }
    }
    
    if (state.viewers.length >= MAX_VIEWERS) {
      console.warn(`Maximum ${MAX_VIEWERS} viewers allowed. Removing oldest viewer.`);
      // 가장 오래된 뷰어 제거
      const newViewers = state.viewers.slice(1);
      const newViewer = {
        ...createDefaultViewer(vehicleId, sourcePath),
        cameraPosition: cameraData.position,
        cameraTarget: cameraData.target,
        cameraUp: cameraData.up,  // ✅ 차량별 up 사용
      };
      set({
        viewers: [...newViewers, newViewer],
        cameraPosition: cameraData.position,
        cameraTarget: cameraData.target,
        cameraUp: cameraData.up,  // ✅ 글로벌 state도 업데이트
        show3DViewer: true,
      });
      return newViewer.id;
    }
    
    const newViewer = {
      ...createDefaultViewer(vehicleId, sourcePath),
      cameraPosition: cameraData.position,
      cameraTarget: cameraData.target,
      cameraUp: cameraData.up,  // ✅ 차량별 up 사용
    };
    set({
      viewers: [...state.viewers, newViewer],
      cameraPosition: cameraData.position,
      cameraTarget: cameraData.target,
      cameraUp: cameraData.up,  // ✅ 글로벌 state도 업데이트
      show3DViewer: true,
    });
    console.log(`[addViewer] Created viewer with camera:`, {
      position: cameraData.position,
      target: cameraData.target,
      up: cameraData.up
    });
    return newViewer.id;
  },
  removeViewer: (viewerId: string) => {
    set((state) => {
      const newViewers = state.viewers.filter(v => v.id !== viewerId);
      return {
        viewers: newViewers,
        show3DViewer: newViewers.length > 0,
      };
    });
  },
  updateViewer: (viewerId: string, updates: Partial<VehicleViewer>) => {
    set((state) => ({
      viewers: state.viewers.map(v => 
        v.id === viewerId ? { ...v, ...updates } : v
      ),
    }));
  },
  setShow3DView: (show: boolean) => set({ show3DViewer: show }),
  getViewer: (viewerId) => {
    return get().viewers.find(v => v.id === viewerId);
  },
  
  getViewerByVehicleId: (vehicleId) => {
    return get().viewers.find(v => v.vehicleId === vehicleId);
  },
  
  // -------------------------
  setCamera: async (preset) => {
    // Fallback hardcoded positions (used if JSON presets not available)
    // Note: position = where camera is located, target = where camera looks at
    const fallbackPositions: Record<CameraPreset, [number, number, number]> = {
      initial: [5, 3, 5],
      front: [0, 2, 8],
      back: [0, 2, -8],
      left: [-8, 2, 0],
      right: [8, 2, 0],
      top: [0, 10, 0],
      front_windshield: [0, 4, 8],
      rear_windshield: [0, 4, -8],
    };
    
    let position: [number, number, number] = fallbackPositions[preset] || [8, 2, 0];
    let lookAtTarget: [number, number, number] | null = null;  // 카메라가 바라보는 지점 (look-at point)
    let upVector: [number, number, number] | null = null;  // ✅ up vector 추가
    
    // Try to get position and target from JSON camera presets
    const state = get();
    if (state.viewers.length > 0) {
      const primaryViewer = state.viewers[0];
      if (primaryViewer.vehicleId) {
        try {
          // Import getCameraPreset to get full preset data (camera_position + target)
          const { getCameraPreset } = await import('@/lib/api');
          const presetData = await getCameraPreset(primaryViewer.vehicleId, preset);
          
          if (presetData) {
            // Transform both camera position and look-at target from JSON to PLY coordinates
            position = transformToPLY(presetData.camera_position);
            lookAtTarget = transformToPLY(presetData.target);
            
            // ✅ up vector도 로드 (있으면)
            if (presetData.up) {
              upVector = transformToPLY(presetData.up);
            }
            
            console.log(`📷 [State.setCamera] Loaded preset '${preset}' from JSON:`, {
              cameraPosition: position,
              lookAtTarget: lookAtTarget,
              upVector: upVector,
            });
          } else {
            console.log(`📷 [State.setCamera] Preset '${preset}' not found in JSON, using fallback position`);
          }
        } catch (error) {
          console.warn(`⚠️ [State.setCamera] Failed to load preset '${preset}', using fallback:`, error);
        }
      }
    }
    
    // Update global state
    set({ 
      cameraPreset: preset,
      cameraPosition: position,
      ...(lookAtTarget ? { cameraTarget: lookAtTarget } : {}),  // cameraTarget = 카메라가 바라보는 지점
      ...(upVector ? { cameraUp: upVector } : {}),  // ✅ up vector도 업데이트
    });
    
    // Update primary viewer (first viewer) if exists
    if (state.viewers.length > 0) {
      set({
        viewers: state.viewers.map((v, idx) => 
          idx === 0 ? { 
            ...v, 
            cameraPreset: preset, 
            cameraPosition: position,
            ...(lookAtTarget ? { cameraTarget: lookAtTarget } : {}),
            ...(upVector ? { cameraUp: upVector } : {}),  // ✅ up vector도 업데이트
          } : v
        ),
      });
    }
  },
  setCameraPosition: (position) => {
    console.log(`[setCameraPosition] Setting position to:`, position);
    // Update global state
    set({ cameraPosition: position });
    
    // Update primary viewer (first viewer) if exists
    const state = get();
    console.log(`[setCameraPosition] Current viewers count:`, state.viewers.length);
    if (state.viewers.length > 0) {
      console.log(`[setCameraPosition] Updating first viewer's camera position`);
      set({
        viewers: state.viewers.map((v, idx) => 
          idx === 0 ? { ...v, cameraPosition: position } : v
        ),
      });
    }
  },
  setCameraTarget: (target) => {
    // Update global state
    set({ cameraTarget: target });
    
    // Update primary viewer (first viewer) if exists
    const state = get();
    if (state.viewers.length > 0) {
      set({
        viewers: state.viewers.map((v, idx) => 
          idx === 0 ? { ...v, cameraTarget: target } : v
        ),
      });
    }
  },
  setCameraUp: (up) => {
    // Update global state
    set({ cameraUp: up });
    
    // Update primary viewer (first viewer) if exists
    const state = get();
    if (state.viewers.length > 0) {
      set({
        viewers: state.viewers.map((v, idx) => 
          idx === 0 ? { ...v, cameraUp: up } : v
        ),
      });
    }
  },
  setCameraZoom: (zoom) => {
    const clampedZoom = Math.max(0, Math.min(100, zoom));
    // Update global state
    set({ cameraZoom: clampedZoom });
    
    // Update primary viewer (first viewer) if exists
    const state = get();
    if (state.viewers.length > 0) {
      set({
        viewers: state.viewers.map((v, idx) => 
          idx === 0 ? { ...v, cameraZoom: clampedZoom } : v
        ),
      });
    }
  },
  zoomCamera: (direction) =>
    set((state) => {
      const delta = direction === "in" ? 10 : -10;
      const newZoom = Math.max(0, Math.min(100, state.cameraZoom + delta));
      return { cameraZoom: newZoom };
    }),
  // -------------------------
  // Camera animation with duration (for Presentation Script)
  cameraAnimationRequest: null,
  animateCameraToPreset: async (preset, duration = 2.0, zoom?: number) => {
    const state = get();
    const primaryViewer = state.viewers[0];
    let position: [number, number, number] | null = null;
    let target: [number, number, number] | null = null;
    let upVector: [number, number, number] | null = null;  // ✅ up vector 추가
    
    // Try to load preset from JSON file if viewer exists
    if (primaryViewer?.vehicleId) {
      try {
        const { getCameraPreset } = await import('@/lib/api');
        const { transformToPLY } = await import('@/lib/camera-utils');
        const presetData = await getCameraPreset(primaryViewer.vehicleId, preset);
        
        if (presetData) {
          position = transformToPLY(presetData.camera_position);
          target = transformToPLY(presetData.target);
          
          // ✅ up vector도 로드 (있으면)
          if (presetData.up) {
            upVector = transformToPLY(presetData.up);
          }
          
          console.log(`📷 [State] Loaded preset '${preset}' from JSON:`, { 
            position, 
            target,
            upVector 
          });
        }
      } catch (error) {
        console.log(`⚠️ [State] Failed to load preset '${preset}' from JSON, using fallback`, error);
      }
    }
    
    // Fallback to hardcoded positions if JSON preset not found
    if (!position) {
      const presetPositions: Record<CameraPreset, [number, number, number]> = {
        initial: [5, 3, 5],
        front: [0, 2, 8],
        back: [0, 2, -8],
        left: [-8, 2, 0],
        right: [8, 2, 0],
        top: [0, 10, 0],
        front_windshield: [0, 4, 8],  // Fallback: elevated front view
        rear_windshield: [0, 4, -8],  // Fallback: elevated rear view
      };
      position = presetPositions[preset] || [8, 2, 0];
      console.log(`📷 [State] Using fallback position for preset '${preset}':`, position);
    }
    
    // Update global state
    set({ 
      cameraPreset: preset,
      cameraPosition: position,
      ...(target ? { cameraTarget: target } : {}),
      ...(upVector ? { cameraUp: upVector } : {}),  // ✅ up vector도 업데이트
      cameraAnimationRequest: {
        type: "preset",
        preset,
        position,  // ← 실제 로드된 position 전달
        ...(target ? { target } : {}),  // ← 실제 로드된 target 전달 (있을 때만)
        ...(zoom !== undefined ? { zoom } : {}),  // ← zoom 파라미터 추가
        duration,
        timestamp: Date.now(),
      },
    });
    
    // Update primary viewer (first viewer) if exists
    if (state.viewers.length > 0) {
      set({
        viewers: state.viewers.map((v, idx) => 
          idx === 0 ? { 
            ...v, 
            cameraPreset: preset, 
            cameraPosition: position,
            ...(target ? { cameraTarget: target } : {}),
            ...(upVector ? { cameraUp: upVector } : {}),  // ✅ up vector도 업데이트
          } : v
        ),
      });
    }
  },
  animateCameraToPosition: (position, duration = 2.0, center?: [number, number, number], zoom?: number) => {
    // Update global state
    set({
      cameraPosition: position,
      cameraAnimationRequest: {
        type: "position",
        position,
        duration,
        center,
        zoom,
        timestamp: Date.now(),
      },
    });
    
    // Update primary viewer (first viewer) if exists
    const state = get();
    if (state.viewers.length > 0) {
      set({
        viewers: state.viewers.map((v, idx) => 
          idx === 0 ? { ...v, cameraPosition: position } : v
        ),
      });
    }
  },
  animateCameraZoom: (level, duration = 1.0) => {
    // Trigger animation request only (즉시 zoom 값 반영하지 않음)
    set({
      cameraAnimationRequest: {
        type: "zoom",
        zoom: level,
        duration,
        timestamp: Date.now(),
      },
    });
  },
  animateCameraRotation: (target, duration = 2.0) => {
    set({
      cameraAnimationRequest: {
        type: "rotation",
        target,
        duration,
        timestamp: Date.now(),
      },
    });
  },
  clearCameraAnimationRequest: () => set({ cameraAnimationRequest: null }),
  // -------------------------
  // 뷰어별 카메라 제어 메서드
  setCameraForViewer: async (viewerId, preset) => {
    // Fallback hardcoded positions (used if JSON presets not available)
    const fallbackPositions: Record<CameraPreset, [number, number, number]> = {
      initial: [5, 3, 5],
      front: [0, 2, 8],
      back: [0, 2, -8],
      left: [-8, 2, 0],
      right: [8, 2, 0],
      top: [0, 10, 0],
      front_windshield: [0, 4, 8],
      rear_windshield: [0, 4, -8],
    };
    
    let position: [number, number, number] = fallbackPositions[preset] || [8, 2, 0];
    
    // Try to get position from JSON camera presets
    const state = get();
    const viewer = state.viewers.find(v => v.id === viewerId);
    if (viewer && viewer.vehicleId) {
      try {
        const cameraPresetsData = await getCameraPresets(viewer.vehicleId);
        if (cameraPresetsData && cameraPresetsData[preset]) {
          // Transform from JSON coordinate system to PLY coordinate system
          const jsonPosition = cameraPresetsData[preset].camera_position as [number, number, number];
          position = transformToPLY(jsonPosition);
          console.log(`📷 [State] Using JSON preset for viewer '${viewerId}', preset '${preset}':`, jsonPosition, '→ PLY:', position);
        } else {
          console.log(`📷 [State] Using fallback position for viewer '${viewerId}', preset '${preset}':`, position);
        }
      } catch (error) {
        console.warn(`⚠️ [State] Failed to load camera presets for viewer '${viewerId}', using fallback:`, error);
      }
    }
    
    set((state) => ({
      viewers: state.viewers.map((v) =>
        v.id === viewerId ? { ...v, cameraPreset: preset, cameraPosition: position } : v
      ),
    }));
  },
  setCameraPositionForViewer: (viewerId, position) => {
    set((state) => ({
      viewers: state.viewers.map((v) =>
        v.id === viewerId ? { ...v, cameraPosition: position } : v
      ),
    }));
  },
  setCameraZoomForViewer: (viewerId, zoom) => {
    const clampedZoom = Math.max(0, Math.min(100, zoom));
    set((state) => ({
      viewers: state.viewers.map((v) =>
        v.id === viewerId ? { ...v, cameraZoom: clampedZoom } : v
      ),
    }));
  },
  animateCameraToPresetForViewer: async (viewerId, preset, duration = 2.0, zoom?: number) => {
    const state = get();
    const viewer = state.viewers.find(v => v.id === viewerId);
    let position: [number, number, number] | null = null;
    let target: [number, number, number] | null = null;
    let upVector: [number, number, number] | null = null;
    
    // Try to load preset from JSON file if viewer exists
    if (viewer?.vehicleId) {
      try {
        const { getCameraPreset } = await import('@/lib/api');
        const { transformToPLY } = await import('@/lib/camera-utils');
        const presetData = await getCameraPreset(viewer.vehicleId, preset);
        
        if (presetData) {
          position = transformToPLY(presetData.camera_position);
          target = transformToPLY(presetData.target);
          
          if (presetData.up) {
            upVector = transformToPLY(presetData.up);
          }
          
          console.log(`📷 [State] Loaded preset '${preset}' from JSON for viewer '${viewerId}':`, { 
            position, 
            target,
            upVector 
          });
        }
      } catch (error) {
        console.log(`⚠️ [State] Failed to load preset '${preset}' from JSON for viewer '${viewerId}', using fallback`, error);
      }
    }
    
    // Fallback to hardcoded positions if JSON preset not found
    if (!position) {
      const presetPositions: Record<CameraPreset, [number, number, number]> = {
        initial: [5, 3, 5],
        front: [0, 2, 8],
        back: [0, 2, -8],
        left: [-8, 2, 0],
        right: [8, 2, 0],
        top: [0, 10, 0],
        front_windshield: [0, 4, 8],
        rear_windshield: [0, 4, -8],
      };
      position = presetPositions[preset] || [8, 2, 0];
      console.log(`📷 [State] Using fallback position for viewer '${viewerId}', preset '${preset}':`, position);
    }
    
    set((state) => ({
      viewers: state.viewers.map((v) =>
        v.id === viewerId ? { 
          ...v, 
          cameraPreset: preset, 
          cameraPosition: position!,
          ...(target ? { cameraTarget: target } : {}),
          ...(upVector ? { cameraUp: upVector } : {}),
        } : v
      ),
      cameraAnimationRequest: {
        type: "preset",
        preset,
        position,  // ← 실제 로드된 position 전달
        ...(target ? { target } : {}),  // ← 실제 로드된 target 전달 (있을 때만)
        ...(zoom !== undefined ? { zoom } : {}),  // ← zoom 파라미터 추가
        duration,
        timestamp: Date.now(),
        viewerId, // 뷰어 ID 추가
      },
    }));
  },
  animateCameraToPositionForViewer: (viewerId, position, duration = 2.0, center?: [number, number, number], zoom?: number) => {
    set((state) => ({
      viewers: state.viewers.map((v) =>
        v.id === viewerId ? { ...v, cameraPosition: position } : v
      ),
      cameraAnimationRequest: {
        type: "position",
        position,
        duration,
        center,
        ...(zoom !== undefined ? { zoom } : {}),  // ← zoom 파라미터 추가
        timestamp: Date.now(),
        viewerId, // 뷰어 ID 추가
      },
    }));
  },
  animateCameraZoomForViewer: (viewerId, level, duration = 1.0) => {
    const clampedZoom = Math.max(0, Math.min(100, level));
    set((state) => ({
      cameraAnimationRequest: {
        type: "zoom",
        zoom: clampedZoom,
        duration,
        timestamp: Date.now(),
        viewerId, // 뷰어 ID 추가
      },
      viewers: state.viewers.map((v) =>
        v.id === viewerId ? { ...v } : v
      ),
    }));
  },
  // -------------------------
  // Presentation 관련 상태 및 메서드
  presentationState: initialPresentationState,
  startPresentation: async (script) => {
    // 먼저 이전 프레젠테이션을 완전히 정리
    const currentState = get();
    if (currentState.presentationState.player) {
      console.log("[ViewerStore] Stopping previous presentation before starting new one");
      try {
        await currentState.presentationState.player.stop();
      } catch (error) {
        console.warn("[ViewerStore] Error stopping previous presentation:", error);
      }
      // TTS 스트리밍 완전 정리를 위한 추가 대기
      await new Promise((resolve) => setTimeout(resolve, 150));
    }
    
    const { PresentationPlayer } = await import("@/components/presentation-player");
    const player = new PresentationPlayer(script, {
      onSegmentChange: (index) => {
        // Note: timestamp is not used - actual time is calculated in onTimestampUpdate
        set((state) => ({
          presentationState: {
            ...state.presentationState,
            currentSegmentIndex: index,
            // currentTimestamp will be updated by onTimestampUpdate with actual playback time
          },
        }));
      },
      onTimestampUpdate: (timestamp) => {
        // 시간 경과에 따라 currentTimestamp 업데이트
        // total_duration이 없거나 현재 시간보다 작으면 동적으로 업데이트
        set((state) => {
          const currentScript = state.presentationState.currentScript;
          let updatedTotalDuration = currentScript?.total_duration || 0;
          
          // 현재 재생 시간이 total_duration보다 크면 total_duration을 업데이트
          if (timestamp > updatedTotalDuration) {
            updatedTotalDuration = timestamp;
          }
          
          // total_duration이 0이거나 없으면 최소한 현재 시간 + 여유 시간(5초)로 설정
          if (updatedTotalDuration <= 0) {
            updatedTotalDuration = timestamp + 5;
          }
          
          return {
            presentationState: {
              ...state.presentationState,
              currentTimestamp: timestamp,
              currentScript: currentScript ? {
                ...currentScript,
                total_duration: updatedTotalDuration,
              } : null,
            },
          };
        });
      },
      onTextUpdate: (text, index) => {
        // 현재 읽고 있는 텍스트 업데이트
        set((state) => ({
          presentationState: {
            ...state.presentationState,
            currentReadingText: text,
            currentSegmentIndex: index,
          },
        }));
      },
      onComplete: () => {
        set((state) => {
          const currentScript = state.presentationState.currentScript;
          const finalTimestamp = state.presentationState.currentTimestamp;
          
          // 최종 재생 시간을 total_duration으로 설정
          const finalTotalDuration = finalTimestamp > 0 ? finalTimestamp : (currentScript?.total_duration || 0);
          
          return {
            presentationState: {
              ...state.presentationState,
              isPlaying: false,
              isPaused: false,
              currentReadingText: "",
              currentScript: currentScript ? {
                ...currentScript,
                total_duration: finalTotalDuration,
              } : null,
            },
          };
        });
      },
      onError: (error) => {
        console.error("Presentation error:", error);
        set((state) => ({
          presentationState: {
            ...state.presentationState,
            isPlaying: false,
            isPaused: false,
            currentReadingText: "",
          },
        }));
      },
    });

    set({
      presentationState: {
        isPlaying: true,
        isPaused: false,
        currentScript: script,
        currentSegmentIndex: 0,
        currentTimestamp: 0,
        currentReadingText: "",
        player,
      },
    });

    // Start playing
    player.play().catch((error) => {
      console.error("Failed to start presentation:", error);
      set((state) => ({
        presentationState: {
          ...state.presentationState,
          isPlaying: false,
        },
      }));
    });
  },
  stopPresentation: async () => {
    const state = get();
    if (state.presentationState.player) {
      try {
        await state.presentationState.player.stop();
      } catch (error) {
        console.warn("[ViewerStore] Error stopping presentation:", error);
      }
    }
    set({
      presentationState: {
        ...initialPresentationState,
      },
    });
  },
  pausePresentation: async () => {
    const state = get();
    if (state.presentationState.player) {
      try {
        await state.presentationState.player.pause();
        set((state) => ({
          presentationState: {
            ...state.presentationState,
            isPaused: true,
          },
        }));
      } catch (error) {
        console.error("Failed to pause presentation:", error);
        set((state) => ({
          presentationState: {
            ...state.presentationState,
            isPaused: true,
          },
        }));
      }
    }
  },
  resumePresentation: async () => {
    const state = get();
    if (state.presentationState.player) {
      // 상태를 먼저 업데이트하여 UI가 즉시 반응하도록 함
      set((state) => ({
        presentationState: {
          ...state.presentationState,
          isPaused: false,
        },
      }));
      
      try {
        await state.presentationState.player.resume();
      } catch (error) {
        console.error("Failed to resume presentation:", error);
        // 에러 발생 시 다시 pause 상태로 되돌림
        set((state) => ({
          presentationState: {
            ...state.presentationState,
            isPaused: true,
          },
        }));
      }
    }
  },
  updatePresentationState: (updates) =>
    set((state) => ({
      presentationState: {
        ...state.presentationState,
        ...updates,
      },
    })),
  // -------------------------
  reset: () => set({ 
    ...initialState, 
    presentationState: initialPresentationState,
    cameraAnimationRequest: null,
  }),
}));
