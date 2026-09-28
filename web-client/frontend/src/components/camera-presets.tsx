"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { getCameraPreset, type CameraPreset as APIPreset, getCameraPoseData, getCameraPresets } from "@/lib/api";
import { transformToPLY, calculateRotateCamera, getRotateDuration, getZoomLevelForShot, getZoomDuration, getZoomLevelForRotateShot } from "@/lib/camera-utils";
import { useViewerStore } from "@/lib/state";
import { useState, useEffect } from "react";

interface CameraPresetsProps {
  vehicleId: string;
}

// Preset display names and icons
const PRESET_INFO: Record<string, { label: string; icon: string; color: string }> = {
  front: { label: "Front", icon: "🚗", color: "bg-blue-500" },
  back: { label: "Back", icon: "🚙", color: "bg-blue-600" },
  left: { label: "Left", icon: "⬅️", color: "bg-purple-500" },
  right: { label: "Right", icon: "➡️", color: "bg-purple-600" },
  top: { label: "Top", icon: "⬆️", color: "bg-green-500" },
  front_windshield: { label: "F. Windshield", icon: "🪟", color: "bg-cyan-500" },
  rear_windshield: { label: "R. Windshield", icon: "🪟", color: "bg-cyan-600" },
};

export function CameraPresets({ vehicleId }: CameraPresetsProps) {
  const [availablePresets, setAvailablePresets] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [vehicleCenter, setVehicleCenter] = useState<[number, number, number] | null>(null);
  const [vehicleForward, setVehicleForward] = useState<[number, number, number] | null>(null);
  const [vehicleSide, setVehicleSide] = useState<[number, number, number] | null>(null);
  const [vehicleUp, setVehicleUp] = useState<[number, number, number] | null>(null);
  const [vehicleBbox, setVehicleBbox] = useState<{
    height: [number, number];
    width: [number, number];
    length: [number, number];
  } | null>(null);
  const { setCameraPosition, setCameraTarget, animateCameraToPosition, animateCameraZoom } = useViewerStore();
  
  // Rotate Camera 테스트 상태 (Camera Presets와 동일한 타겟 사용)
  const [testTarget, setTestTarget] = useState<"front" | "back" | "left" | "right" | "top" | "front_windshield" | "rear_windshield" | string>("front");
  const [testShot, setTestShot] = useState<"wide" | "full" | "medium" | "closeup">("full");
  const [testAngle, setTestAngle] = useState<"eye_level" | "high" | "top" | "low">("eye_level");
  const [testSpeed, setTestSpeed] = useState<"slow" | "medium">("slow");
  const [testZoomShot, setTestZoomShot] = useState<"medium" | "closeup">("medium");
  const [testZoomSpeed, setTestZoomSpeed] = useState<"slow" | "medium">("slow");

  useEffect(() => {
    loadAvailablePresets();
    loadVehicleCenter();
  }, [vehicleId]);

  const loadVehicleCenter = async () => {
    try {
      const cameraPoseData = await getCameraPoseData(vehicleId);
      if (cameraPoseData) {
        if (cameraPoseData.center) {
          setVehicleCenter(cameraPoseData.center);
          console.log(`📐 Loaded vehicle center:`, cameraPoseData.center);
        }
        if (cameraPoseData.forward) {
          setVehicleForward(cameraPoseData.forward);
        }
        if (cameraPoseData.side) {
          setVehicleSide(cameraPoseData.side);
        }
        if (cameraPoseData.up) {
          setVehicleUp(cameraPoseData.up);
        }
        if (cameraPoseData.bbox_local) {
          setVehicleBbox(cameraPoseData.bbox_local);
        }
      }
    } catch (error) {
      console.error("Failed to load vehicle center:", error);
    }
  };

  const loadAvailablePresets = async () => {
    setIsLoading(true);
    try {
      const presets = await getCameraPresets(vehicleId);
      if (presets) {
        const presetNames = Object.keys(presets);
        setAvailablePresets(presetNames);
        console.log(`📷 Loaded ${presetNames.length} camera presets for ${vehicleId}`);
      }
    } catch (error) {
      console.error("Failed to load camera presets:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handlePresetClick = async (presetName: string) => {
    try {
      console.log(`📷 Loading preset: ${presetName}`);
      const preset = await getCameraPreset(vehicleId, presetName);
      
      if (!preset) {
        console.error(`Preset ${presetName} not found`);
        return;
      }

      // Transform coordinates from JSON to PLY
      const plyPosition = transformToPLY(preset.camera_position);
      const plyTarget = transformToPLY(preset.target);

      console.log(`📷 Applying preset ${presetName}:`);
      console.log(`  Camera position: ${plyPosition}`);
      console.log(`  Target: ${plyTarget}`);

      // Update camera
      setCameraPosition(plyPosition);
      setCameraTarget(plyTarget);
    } catch (error) {
      console.error(`Failed to load preset ${presetName}:`, error);
    }
  };

  const handleRotateCameraTest = async () => {
    try {
      // 차량 데이터가 로드되지 않았으면 경고
      if (!vehicleCenter || !vehicleForward || !vehicleSide || !vehicleUp) {
        console.warn("⚠️ Vehicle data not loaded yet. Please wait...");
        return;
      }

      const center: [number, number, number] = vehicleCenter;
      const forward: [number, number, number] = vehicleForward;
      const side: [number, number, number] = vehicleSide;
      const up: [number, number, number] = vehicleUp;
      
      // 실제 프리셋 값과 비교를 위해 로드
      const actualPreset = await getCameraPreset(vehicleId, testTarget);
      const topPreset = await getCameraPreset(vehicleId, "top");
      if (actualPreset) {
        console.log(`📷 Actual preset (${testTarget}) from JSON:`, {
          camera_position: actualPreset.camera_position,
          target: actualPreset.target,
        });
      }
      
      console.log(`🧪 Testing rotateCamera:`, {
        target: testTarget,
        shot: testShot,
        angle: testAngle,
        speed: testSpeed,
        center: center,
        forward: forward,
        side: side,
        up: up,
        bbox: vehicleBbox,
      });

      // 카메라 위치 계산 (JSON 좌표계에서, Camera Presets와 동일한 방식)
      // bbox를 전달하여 optimal distance 계산
      // 프리셋 위치(해당 타겟 & top)를 높이 범위로 사용
      const { position, targetPoint } = calculateRotateCamera(
        center,
        forward,
        side,
        up,
        testTarget,
        testShot,
        testAngle,
        vehicleBbox || undefined,
        actualPreset?.camera_position as [number, number, number] | undefined,
        topPreset?.camera_position as [number, number, number] | undefined
      );

      console.log(`📷 Calculated (JSON coords):`, {
        position,
        targetPoint,
      });

      // 실제 프리셋과 비교 (shot이 full이고 angle이 eye_level일 때)
      if (actualPreset && testShot === "full" && testAngle === "eye_level") {
        const posDiff = [
          position[0] - actualPreset.camera_position[0],
          position[1] - actualPreset.camera_position[1],
          position[2] - actualPreset.camera_position[2],
        ];
        const distDiff = Math.sqrt(posDiff[0] ** 2 + posDiff[1] ** 2 + posDiff[2] ** 2);
        console.log(`📊 Comparison with actual preset:`, {
          calculated: position,
          actual: actualPreset.camera_position,
          difference: posDiff,
          distance_diff: distDiff.toFixed(2),
        });
      }

      // PLY 좌표로 변환
      const plyPosition = transformToPLY(position);
      const plyTarget = transformToPLY(targetPoint);
      const plyCenter = transformToPLY(center);

      // 애니메이션 속도에 따른 duration 계산
      const duration = getRotateDuration(testSpeed);

      console.log(`📷 Rotate camera test result (PLY coords):`, {
        position: plyPosition,
        target: plyTarget,
        duration: `${duration}s`,
      });

      // 카메라 애니메이션 실행 (위치 + 샷 기반 줌 동시 보간)
      animateCameraToPosition(plyPosition, duration, plyCenter, getZoomLevelForRotateShot(testShot));
      setCameraTarget(plyTarget);
    } catch (error) {
      console.error(`Failed to test rotateCamera:`, error);
    }
  };

  const handleZoomTest = () => {
    const level = getZoomLevelForShot(testZoomShot);
    const duration = getZoomDuration(testZoomSpeed);
    console.log(`🧪 Zoom test: shot=${testZoomShot}, speed=${testZoomSpeed}, level=${level}, duration=${duration}s`);
    animateCameraZoom(level, duration);
  };

  if (isLoading) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="text-sm">Camera Presets</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-sm text-muted-foreground">Loading presets...</div>
        </CardContent>
      </Card>
    );
  }

  if (availablePresets.length === 0) {
    return null;
  }

  return (
    <div className="space-y-4">
      {/* Rotate Camera 테스트 섹션 */}
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="text-sm">🧪 Rotate Camera 테스트</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* TARGET 선택 */}
          <div>
            <label className="text-xs font-medium text-muted-foreground mb-2 block">
              TARGET (시점)
            </label>
            <div className="grid grid-cols-3 gap-2">
              {(["front", "back", "left", "right", "top", "front_windshield", "rear_windshield"] as const).map((target) => {
                const info = PRESET_INFO[target] || {
                  label: target === "front_windshield" ? "F. Windshield" : target === "rear_windshield" ? "R. Windshield" : target.charAt(0).toUpperCase() + target.slice(1),
                  icon: "📷",
                };
                return (
                  <Button
                    key={target}
                    onClick={() => setTestTarget(target)}
                    variant={testTarget === target ? "default" : "outline"}
                    size="sm"
                    className="text-xs justify-start gap-1"
                  >
                    <span>{info.icon}</span>
                    <span>{info.label}</span>
                  </Button>
                );
              })}
            </div>
          </div>

          {/* SHOT 선택 */}
          <div>
            <label className="text-xs font-medium text-muted-foreground mb-2 block">
              SHOT (거리)
            </label>
            <div className="grid grid-cols-4 gap-2">
              {(["wide", "full", "medium", "closeup"] as const).map((shot) => (
                <Button
                  key={shot}
                  onClick={() => setTestShot(shot)}
                  variant={testShot === shot ? "default" : "outline"}
                  size="sm"
                  className="text-xs"
                >
                  {shot.charAt(0).toUpperCase() + shot.slice(1)}
                </Button>
              ))}
            </div>
          </div>

          {/* ANGLE 선택 */}
          <div>
            <label className="text-xs font-medium text-muted-foreground mb-2 block">
              ANGLE (높이)
            </label>
            <div className="grid grid-cols-4 gap-2">
              {(["eye_level", "high", "top", "low"] as const).map((angle) => (
                <Button
                  key={angle}
                  onClick={() => setTestAngle(angle)}
                  variant={testAngle === angle ? "default" : "outline"}
                  size="sm"
                  className="text-xs"
                >
                  {angle === "eye_level" ? "Eye Level" : angle.charAt(0).toUpperCase() + angle.slice(1)}
                </Button>
              ))}
            </div>
          </div>

          {/* SPEED 선택 */}
          <div>
            <label className="text-xs font-medium text-muted-foreground mb-2 block">
              SPEED (속도)
            </label>
            <div className="grid grid-cols-2 gap-2">
              {(["slow", "medium"] as const).map((speed) => (
                <Button
                  key={speed}
                  onClick={() => setTestSpeed(speed)}
                  variant={testSpeed === speed ? "default" : "outline"}
                  size="sm"
                  className="text-xs"
                >
                  {speed.charAt(0).toUpperCase() + speed.slice(1)} ({getRotateDuration(speed)}s)
                </Button>
              ))}
            </div>
          </div>

          {/* 현재 설정 표시 */}
          <div className="pt-2 border-t">
            <div className="text-xs text-muted-foreground mb-2">
              현재 설정: <span className="font-medium">{testTarget}</span> → <span className="font-medium">{testShot}</span> shot, <span className="font-medium">{testAngle}</span> angle, <span className="font-medium">{testSpeed}</span> speed
            </div>
            <Button
              onClick={handleRotateCameraTest}
              className="w-full"
              size="sm"
            >
              🎬 테스트 실행
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Zoom 테스트 섹션 */}
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="text-sm">🧪 Zoom 테스트</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* SHOT 선택 */}
          <div>
            <label className="text-xs font-medium text-muted-foreground mb-2 block">
              SHOT (Zoom level)
            </label>
            <div className="grid grid-cols-2 gap-2">
              {(["medium", "closeup"] as const).map((shot) => (
                <Button
                  key={shot}
                  onClick={() => setTestZoomShot(shot)}
                  variant={testZoomShot === shot ? "default" : "outline"}
                  size="sm"
                  className="text-xs"
                >
                  {shot === "medium" ? "Medium" : "Closeup"} (level {getZoomLevelForShot(shot)})
                </Button>
              ))}
            </div>
          </div>

          {/* SPEED 선택 */}
          <div>
            <label className="text-xs font-medium text-muted-foreground mb-2 block">
              SPEED (duration)
            </label>
            <div className="grid grid-cols-2 gap-2">
              {(["slow", "medium"] as const).map((speed) => (
                <Button
                  key={speed}
                  onClick={() => setTestZoomSpeed(speed)}
                  variant={testZoomSpeed === speed ? "default" : "outline"}
                  size="sm"
                  className="text-xs"
                >
                  {speed.charAt(0).toUpperCase() + speed.slice(1)} ({getZoomDuration(speed)}s)
                </Button>
              ))}
            </div>
          </div>

          <Button onClick={handleZoomTest} className="w-full" size="sm">
            🔍 Zoom 실행
          </Button>
        </CardContent>
      </Card>

      {/* 기존 Camera Presets 섹션 */}
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="text-sm">📷 Camera Presets</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-2">
            {availablePresets.map((presetName) => {
              const info = PRESET_INFO[presetName] || {
                label: presetName,
                icon: "📷",
                color: "bg-gray-500"
              };
              
              return (
                <Button
                  key={presetName}
                  onClick={() => handlePresetClick(presetName)}
                  variant="outline"
                  size="sm"
                  className="justify-start gap-2 h-auto py-2"
                >
                  <span className="text-lg">{info.icon}</span>
                  <span className="text-xs font-medium">{info.label}</span>
                </Button>
              );
            })}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
