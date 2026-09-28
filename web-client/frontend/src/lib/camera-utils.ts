/**
 * Camera utilities for 3D viewer
 * Based on spark's camera calculation logic
 */

export interface CameraPoseData {
  center: [number, number, number];
  forward: [number, number, number];
  up: [number, number, number];
  side: [number, number, number];
  bbox_local?: {
    height: [number, number];
    width: [number, number];
    length: [number, number];
  };
}

/**
 * Transform coordinates from JSON coordinate system to PLY coordinate system
 * PLY coords = [jsonX, -jsonY, -jsonZ] for all vectors
 */
export function transformToPLY(
  jsonCoords: [number, number, number]
): [number, number, number] {
  return [jsonCoords[0], -jsonCoords[1], -jsonCoords[2]];
}

/**
 * Calculate optimal camera distance based on bounding box
 */
export function calculateOptimalDistance(
  bbox?: {
    height: [number, number];
    width: [number, number];
    length: [number, number];
  }
): number {
  if (!bbox) {
    return 5; // Default distance
  }

  // Calculate dimensions
  const height = bbox.height[1] - bbox.height[0];
  const width = bbox.width[1] - bbox.width[0];
  const length = bbox.length[1] - bbox.length[0];

  // Max dimension + 50% margin
  const maxDimension = Math.max(height, width, length);
  const distance = maxDimension * 1.5;

  return distance;
}

/**
 * Calculate camera position from center and axis direction
 * @param center Center point
 * @param axis Axis direction (forward/up/side)
 * @param distance Distance from center
 * @param reverse If true, camera is positioned opposite to axis direction
 * @returns Camera position
 */
export function calculateCameraPosition(
  center: [number, number, number],
  axis: [number, number, number],
  distance: number = 5,
  reverse: boolean = false
): [number, number, number] {
  // Normalize axis
  const len = Math.sqrt(axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2);
  const normalized = axis.map((v) => v / len) as [number, number, number];

  // Reverse direction if needed (for views that look "at" the object from opposite direction)
  const direction = reverse ? -1 : 1;

  // Position camera along axis direction at distance from center
  return [
    center[0] + normalized[0] * distance * direction,
    center[1] + normalized[1] * distance * direction,
    center[2] + normalized[2] * distance * direction,
  ];
}

/**
 * Calculate initial camera view (eye-level front 3/4)
 * This matches spark's initial camera setup
 */
export function calculateInitialCameraView(cameraPose: CameraPoseData): {
  position: [number, number, number];
  target: [number, number, number];
  up: [number, number, number];
} {
  // Transform all vectors from JSON to PLY coordinate system
  const plyCenter = transformToPLY(cameraPose.center);
  const plyForward = transformToPLY(cameraPose.forward);
  const plySide = transformToPLY(cameraPose.side);
  const plyUp = transformToPLY(cameraPose.up);

  // Calculate optimal distance based on bbox
  const optimalDistance = calculateOptimalDistance(cameraPose.bbox_local);

  // Initial view: Eye-level front 3/4 view
  // Combine forward (75%) and side (25%) for 3/4 angle
  const forwardComponent = plyForward.map((v) => v * 0.75) as [
    number,
    number,
    number
  ];
  const sideComponent = plySide.map((v) => v * 0.25) as [
    number,
    number,
    number
  ];
  const combinedDirection: [number, number, number] = [
    forwardComponent[0] + sideComponent[0],
    forwardComponent[1] + sideComponent[1],
    forwardComponent[2] + sideComponent[2],
  ];

  // Calculate position along combined direction
  // Use full shot distance (6.0) as the default initial camera distance
  const fullShotDistance = 6.0;
  const basePos = calculateCameraPosition(
    plyCenter,
    combinedDirection,
    fullShotDistance,
    false
  );

  // Eye level adjustment: Calculate height offset from center
  let heightOffset = 5.0;
  if (cameraPose.bbox_local) {
    // Transform bbox height to PLY coordinate system
    const bboxHeightMin = -cameraPose.bbox_local.height[1]; // Negate and swap
    const bboxHeightMax = -cameraPose.bbox_local.height[0];
    const bboxHeightCenter = (bboxHeightMin + bboxHeightMax) / 2;
    
    // Calculate offset: bbox center relative to plyCenter
    heightOffset = bboxHeightCenter - plyCenter[1];
  }

  // Apply height offset ONLY to camera position, NOT to target
  // Target should always be the actual PLY center
  const position: [number, number, number] = [
    basePos[0],
    basePos[1] + heightOffset,
    basePos[2],
  ];

  // Target is the actual PLY center - no offset!
  const target: [number, number, number] = [
    plyCenter[0],
    plyCenter[1],
    plyCenter[2],
  ];

  return {
    position,
    target: target,
    up: plyUp,
  };
}

/**
 * Calculate camera position for rotate action
 * Camera Presets와 동일한 방식으로 실제 차량의 forward/side/up 벡터 사용
 * @param center: 중심점
 * @param forward: 차량의 forward 벡터 (JSON 좌표계)
 * @param side: 차량의 side 벡터 (JSON 좌표계)
 * @param up: 차량의 up 벡터 (JSON 좌표계)
 * @param target: 시점 (front, side, rear, rear_three_quarter, top) 또는 부품명
 * @param shot: 샷 타입 (wide, full, medium, closeup)
 * @param angle: 각도 (eye_level, high, top, low)
 * @param bbox: 차량의 bounding box (optional, optimal distance 계산에 사용)
 * @returns 카메라 위치 및 대상점
 */
export function calculateRotateCamera(
  center: [number, number, number],
  forward: [number, number, number],
  side: [number, number, number],
  up: [number, number, number],
  target: string,
  shot: "wide" | "full" | "medium" | "closeup",
  angle: "eye_level" | "high" | "top" | "low",
  bbox?: {
    height: [number, number];
    width: [number, number];
    length: [number, number];
  },
  presetLowPos?: [number, number, number], // JSON 프리셋의 camera_position (해당 target)
  presetTopPos?: [number, number, number] // JSON 프리셋의 top camera_position
): {
  position: [number, number, number];
  targetPoint: [number, number, number];
} {
  // 실제 프리셋 값에 맞춰서:
  // front = forward 방향
  // back = forward 반대
  // left = side 방향
  // right = side 반대
  // 시점별 방향 벡터 (실제 프리셋 값과 일치)
  let direction: [number, number, number];
  
  switch (target) {
    case "front":
      // Front: forward 방향 (실제 프리셋 값과 일치)
      direction = [...forward] as [number, number, number];
      break;
    case "back":
      // Back: forward의 반대 방향 (실제 프리셋 값과 일치)
      direction = [-forward[0], -forward[1], -forward[2]] as [number, number, number];
      break;
    case "left":
      // Left: side 방향 (좌측에서 보기, 실제 프리셋 값과 일치)
      direction = [...side] as [number, number, number];
      break;
    case "right":
      // Right: side의 반대 방향 (우측에서 보기, 실제 프리셋 값과 일치)
      direction = [-side[0], -side[1], -side[2]] as [number, number, number];
      break;
    case "top":
      // Top: up 방향
      direction = [...up] as [number, number, number];
      break;
    case "front_windshield":
      // Front windshield: front 방향 사용 (나중에 up 벡터로 elevation 적용)
      direction = [...forward] as [number, number, number];
      break;
    case "rear_windshield":
      // Rear windshield: back 방향 사용 (나중에 up 벡터로 elevation 적용)
      direction = [-forward[0], -forward[1], -forward[2]] as [number, number, number];
      break;
    // 하위 호환성
    case "rear":
      // rear는 back과 동일
      direction = [-forward[0], -forward[1], -forward[2]] as [number, number, number];
      break;
    case "side":
      // side는 right와 동일
      direction = [-side[0], -side[1], -side[2]] as [number, number, number];
      break;
    default:
      // 기본값: front
      direction = [...forward] as [number, number, number];
  }
  
  // 샷 타입별 거리 (각 단계마다 10%씩 가감)
  // wide → full: -10%
  // full → medium: -10%
  // medium → closeup: -10%
  // full = 1.0 (기본 거리, 초기 로드 시 사용)
  const shotMultiplier: Record<string, number> = {
    wide: 1.0 / 0.9,  // full 대비 +11.11% (wide → full: -10%)
    full: 1.0,        // 기준 (기본)
    medium: 0.9,      // full 대비 -10%
    closeup: 0.81,    // medium 대비 -10% (full 대비 -19%)
  };

  // 기본 거리: Camera Presets와 동일하게 optimal distance 사용 (bbox 기반)
  // bbox가 없으면 기본값 6.0 사용
  const baseDistance = bbox ? calculateOptimalDistance(bbox) : 6.0;
  const multiplier = shotMultiplier[shot] ?? 1.0;
  const distance = baseDistance * multiplier;

  // up 벡터 정규화 (차량 PCA 기준 축)
  const upLen = Math.sqrt(up[0] ** 2 + up[1] ** 2 + up[2] ** 2) || 1;
  const upNormalized: [number, number, number] = [
    up[0] / upLen,
    up[1] / upLen,
    up[2] / upLen,
  ];

  // 프리셋 기반 높이 범위 계산: up 축으로 투영하여 차량 자체 좌표계 기준으로 비교
  const projectToUp = (pos: [number, number, number]) =>
    (pos[0] - center[0]) * upNormalized[0] +
    (pos[1] - center[1]) * upNormalized[1] +
    (pos[2] - center[2]) * upNormalized[2];

  let lowHeight = 0;
  if (presetLowPos) {
    lowHeight = projectToUp(presetLowPos);
  } else if (bbox) {
    const heightRange = bbox.height[1] - bbox.height[0];
    lowHeight = bbox.height[0] - 0.1 + heightRange * 0.05; // 바닥보다 약간 위
  } else {
    lowHeight = -0.2; // 기본 값
  }

  let topHeight: number;
  if (presetTopPos) {
    topHeight = projectToUp(presetTopPos);
  } else {
    const heightRange = bbox ? bbox.height[1] - bbox.height[0] : 1.5;
    const distanceBonus = distance * 0.25;
    topHeight = lowHeight + Math.max(heightRange, distanceBonus);
  }

  // angle 비율 (0=low, 1=top)
  const angleFactor: Record<string, number> = {
    low: 0.0,
    eye_level: 0.198, // 기존 대비 60% (상승폭 40% 감소)
    high: 0.396,
    top: 0.6,
  };
  const factor = angleFactor[angle] ?? 0.33;

  // 목표 높이 (up 축 투영값)
  const targetHeight = lowHeight + (topHeight - lowHeight) * factor;

  // 정규화된 방향
  const len = Math.sqrt(direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2);
  const normalized: [number, number, number] = [
    direction[0] / len,
    direction[1] / len,
    direction[2] / len,
  ];

  // 카메라 위치 계산: 방향 벡터를 따라 이동한 후, up 축을 따라 목표 높이로 보정
  const basePosition: [number, number, number] = [
    center[0] + normalized[0] * distance,
    center[1] + normalized[1] * distance,
    center[2] + normalized[2] * distance,
  ];

  // 현재 높이(센터 대비 up 투영)와 목표 높이 차이 계산
  const baseHeight = (basePosition[0] - center[0]) * upNormalized[0] +
    (basePosition[1] - center[1]) * upNormalized[1] +
    (basePosition[2] - center[2]) * upNormalized[2];

  // Windshield 타겟은 추가로 30% elevation
  const windshieldExtra =
    target === "front_windshield" || target === "rear_windshield"
      ? distance * 0.3
      : 0;

  const deltaHeight = targetHeight + windshieldExtra - baseHeight;

  const position: [number, number, number] = [
    basePosition[0] + upNormalized[0] * deltaHeight,
    basePosition[1] + upNormalized[1] * deltaHeight,
    basePosition[2] + upNormalized[2] * deltaHeight,
  ];

  // 보는 목표점 (중심점)
  const targetPoint: [number, number, number] = [center[0], center[1], center[2]];

  return { position, targetPoint };
}

/**
 * Get animation duration based on speed
 */
export function getRotateDuration(speed: "slow" | "medium"): number {
  const durations: Record<string, number> = {
    slow: 3.0,      // 3초
    medium: 2.0,    // 2초
  };
  return durations[speed] || 2.0;
}

/**
 * Zoom level mapping for shot types (0-100 scale)
 * medium: 기본 +15 (fov 축소)
 * closeup: 기본 +30 (더 축소)
 */
export function getZoomLevelForShot(shot: "medium" | "closeup"): number {
  const levels: Record<string, number> = {
    medium: 75,   // 더 큰 확대
    closeup: 90,  // 더 큰 확대
  };
  return levels[shot] ?? 65;
}

/**
 * RotateCamera shot별 기본 줌 레벨 (0-100)
 * wide   : 약간 멀리
 * full   : 기본
 * medium : 조금 가깝게
 * closeup: 많이 가깝게
 */
export function getZoomLevelForRotateShot(
  shot: "wide" | "full" | "medium" | "closeup"
): number {
  const levels: Record<string, number> = {
    wide: 55,
    full: 65,
    medium: 75,
    closeup: 90,
  };
  return levels[shot] ?? 65;
}

/**
 * Zoom duration mapping for speed
 * slow: 3s, medium: 2s
 */
export function getZoomDuration(speed: "slow" | "medium"): number {
  const durations: Record<string, number> = {
    slow: 3.0,
    medium: 2.0,
  };
  return durations[speed] ?? 2.0;
}
