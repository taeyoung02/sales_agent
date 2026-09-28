export interface Vehicle {
  id: string;
  make: string;
  model: string;
  year: number;
  trim: string;
  body: string;
  drivetrain: string;
  mileage: number;
  price: number;
  colorOptions: string[];
  packages: string[];
  specs: Record<string, string | number>;
  evidence: Evidence[];
  thumbnail?: string;
}

export interface Evidence {
  title: string;
  snippet: string;
  asOf: string;
  href?: string;
}

export type Variant = "base" | "sport" | "lux";
export type CameraPreset = "initial" | "front" | "back" | "left" | "right" | "top" | "front_windshield" | "rear_windshield";

export interface ViewerState {
  viewers: VehicleViewer[]; // 최대 2개 뷰어
  show3DViewer: boolean; // 3D 뷰어 표시 여부(default: false)
  cameraPreset: CameraPreset;
  cameraZoom: number; // 0-100
  cameraPosition: [number, number, number];
  cameraTarget?: [number, number, number];
  cameraUp?: [number, number, number];
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  evidence?: Evidence[];
  timestamp: Date;
}

export interface SessionLog {
  id: string;
  timestamp: Date;
  action: string;
  vehicle?: string;
  details: string;
}


export interface VehicleViewer {
  id: string; // 뷰어 인스턴스 고유 ID
  vehicleId: string | null; // 차량 ID
  sourcePath: string | null; // 3D 파일 경로 (PLY 등)
  cameraPreset: CameraPreset;
  cameraZoom: number; // 0-100
  cameraPosition: [number, number, number];
  cameraTarget?: [number, number, number];
  cameraUp?: [number, number, number];
  isModelLoaded?: boolean; // 3D 모델(.ply) 로드 완료 여부
}

export interface RotateCameraParams {
  target: "front" | "side" | "rear" | "rear_three_quarter" | "top" | string; // 시점 또는 부품명
  shot?: "wide" | "full" | "medium" | "closeup";  // 기본값: full
  angle: "eye_level" | "high" | "top" | "low";
  speed: "slow" | "medium";
}

export interface ZoomCameraParams {
  shot: "medium" | "closeup";
  speed: "slow" | "medium";
}

export interface PresentationAction {
  type: "setCamera" | "zoomCamera" | "rotateCamera" | "generateHeatmap";
  viewer_id?: string; // 특정 뷰어에만 적용 (viewerId 우선)
  vehicle_id?: string; // 특정 차량에만 적용 (viewer_id가 없을 때 사용)
  preset?: CameraPreset;
  position?: [number, number, number];
  direction?: "in" | "out";
  level?: number;
  target?: [number, number, number];
  query?: string;
  duration?: number;
  delay?: number;
  // rotateCamera 파라미터
  rotate?: RotateCameraParams;
  // zoomCamera 파라미터 (신규)
  zoom?: ZoomCameraParams;
}

export interface PresentationSegment {
  timestamp?: number; // Optional: 참고용 (실제 재생 시간은 런타임에 계산됨)
  text: string;
  actions: PresentationAction[];
}

export interface PresentationScript {
  vehicle_id: string;
  total_duration: number;
  segments: PresentationSegment[];
}