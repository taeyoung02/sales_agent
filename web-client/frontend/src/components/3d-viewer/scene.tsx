import { CameraPreset } from "@/lib/types";
import { OrbitControls, PerspectiveCamera } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import { Suspense, useEffect, useRef, useState, useCallback } from "react";
import * as THREE from "three";
import SparkSplat from "./spark-splat";
import { CameraAnimator } from "@/lib/camera-animation";
import { useViewerStore } from "@/lib/state";
import { getCameraPresets } from "@/lib/api";
import { transformToPLY } from "@/lib/camera-utils";

export default function Scene({
  sourcePath,
  cameraPreset,
  cameraZoom,
  cameraPosition,
  cameraTarget = [0, 0, 0],
  cameraUp,
  vehicleId,
}: {
  sourcePath: string | null;
  cameraPreset: CameraPreset;
  cameraZoom: number;
  cameraPosition: [number, number, number];
  cameraTarget?: [number, number, number];
  cameraUp?: [number, number, number];
  vehicleId?: string;
}) {
  const cameraRef = useRef<THREE.PerspectiveCamera>(null);
  const controlsRef = useRef<any>(null);
  const cameraAnimatorRef = useRef<CameraAnimator | null>(null);
  const { cameraAnimationRequest, clearCameraAnimationRequest, presentationState, updateViewer, getViewerByVehicleId } = useViewerStore();
  
  // State for camera presets from JSON
  const [cameraPresetsData, setCameraPresetsData] = useState<Record<string, { camera_position: number[], target: number[] }> | null>(null);
  
  // State for smooth camera animation (legacy, for non-presentation animations)
  const [targetPosition, setTargetPosition] = useState<THREE.Vector3>(
    new THREE.Vector3(...cameraPosition)
  );
  const [isAnimating, setIsAnimating] = useState(false);
  const [isInitialized, setIsInitialized] = useState(false);
  const lastPresetRef = useRef<CameraPreset>(cameraPreset);
  const lastCameraPositionRef = useRef<[number, number, number]>(cameraPosition);
  const isUserInteractingRef = useRef(false);
  const userInteractionTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const lastLoggedPositionRef = useRef<THREE.Vector3 | null>(null);
  
  // Load camera presets from JSON when vehicleId changes
  useEffect(() => {
    if (!vehicleId) return;
    
    const loadPresets = async () => {
      try {
        const presets = await getCameraPresets(vehicleId);
        if (presets) {
          setCameraPresetsData(presets);
        }
      } catch (error) {
        console.error(`❌ [Scene] Failed to load camera presets:`, error);
      }
    };
    
    loadPresets();
  }, [vehicleId]);
  
  // Initialize CameraAnimator when camera and controls are ready
  useEffect(() => {
    if (cameraRef.current && controlsRef.current && !cameraAnimatorRef.current) {
      cameraAnimatorRef.current = new CameraAnimator(
        cameraRef.current,
        controlsRef.current
      );
    }
  }, [cameraRef.current, controlsRef.current]);

  // Update OrbitControls target when cameraTarget prop changes
  useEffect(() => {
    if (controlsRef.current && cameraTarget) {
      const targetVec = new THREE.Vector3(...cameraTarget);
      controlsRef.current.target.copy(targetVec);
      controlsRef.current.update();
      // Force camera to look at target
      if (cameraRef.current) {
        cameraRef.current.lookAt(targetVec);
      }
    }
  }, [cameraTarget?.[0], cameraTarget?.[1], cameraTarget?.[2], controlsRef.current]);

  // Update camera up vector when cameraUp prop changes
  useEffect(() => {
    if (cameraRef.current && cameraUp) {
      const upVec = new THREE.Vector3(...cameraUp);
      cameraRef.current.up.copy(upVec);
      cameraRef.current.updateProjectionMatrix();
      if (controlsRef.current) {
        controlsRef.current.update();
      }
    }
  }, [cameraUp?.[0], cameraUp?.[1], cameraUp?.[2]]);

  // Initialize camera position on mount (only once)
  useEffect(() => {
    if (cameraRef.current && !isInitialized) {
      // Apply eye-level adjustment: raise camera by 1.3 units
      const eyeLevelOffset = 1.3;
      const adjustedPosition = [
        cameraPosition[0],
        cameraPosition[1] + eyeLevelOffset,
        cameraPosition[2]
      ] as [number, number, number];
      
      const initialPos = new THREE.Vector3(...adjustedPosition);
      cameraRef.current.position.copy(initialPos);
      
      // targetPosition is for animation, not for looking at!
      // Set it to current camera position so no animation happens initially
      setTargetPosition(initialPos);
      
      setIsInitialized(true);
    }
  }, [cameraPosition, isInitialized]); // Include cameraPosition in dependencies

  // Update target position when cameraPosition prop changes (after initialization)
  useEffect(() => {
    if (cameraRef.current && isInitialized) {
      // Check if cameraPosition actually changed
      const positionChanged = 
        lastCameraPositionRef.current[0] !== cameraPosition[0] ||
        lastCameraPositionRef.current[1] !== cameraPosition[1] ||
        lastCameraPositionRef.current[2] !== cameraPosition[2];
      
      if (positionChanged) {
        const newTarget = new THREE.Vector3(...cameraPosition);
        // IMPORTANT: Use current camera position (not ref) as starting point for animation
        // This ensures animation starts from where camera actually is, not where it should be
        const currentPos = cameraRef.current.position.clone();
        
        // Only animate if position actually changed
        if (currentPos.distanceTo(newTarget) > 0.01) {
          // Set target and start animation from current position
          setTargetPosition(newTarget);
          setIsAnimating(true);
          lastCameraPositionRef.current = cameraPosition;
        }
      }
    }
  }, [cameraPosition, isInitialized]);

  // Handle camera animation requests from Presentation Script
  useEffect(() => {
    if (!cameraAnimationRequest || !cameraAnimatorRef.current) return;
    
    const animator = cameraAnimatorRef.current;
    const request = cameraAnimationRequest;
    
    // Stop any ongoing legacy animation
    setIsAnimating(false);
    
    switch (request.type) {
      case "preset":
        // Use position/target from request if available (loaded from JSON),
        // otherwise fallback to animateToPreset with preset name
        if (request.position) {
          animator.animateToPosition(
            request.position,
            request.duration || 2.0,
            "easeInOut",
            request.center ?? request.target,  // Use provided center (or target) as orbit center
            request.zoom  // ← zoom 파라미터 추가
          );
        } else if (request.preset) {
          // preset만 있을 때도 zoom이 있으면 함께 처리
          animator.animateToPreset(request.preset, request.duration || 2.0, "easeInOut", request.zoom);
        }
        break;
      case "position":
        if (request.position) {
          animator.animateToPosition(
            request.position,
            request.duration || 2.0,
            "easeInOut",
            request.center ?? request.target,  // Use provided center (or target) as orbit center
            request.zoom
          );
        }
        break;
      case "zoom":
        if (request.zoom !== undefined) {
          animator.animateZoom(request.zoom, request.duration || 1.0);
        }
        break;
      case "rotation":
        if (request.target) {
          animator.animateRotation(request.target, request.duration || 2.0);
        }
        break;
    }
    
    // Clear the request after processing
    clearCameraAnimationRequest();
  }, [cameraAnimationRequest, clearCameraAnimationRequest]);

  // Smooth camera animation using useFrame (runs every frame)
  useFrame((state, delta) => {
    if (cameraRef.current) {
      const currentPos = cameraRef.current.position;
      
      // Log camera position when it changes (throttled to avoid too many logs)
      if (!lastLoggedPositionRef.current || 
          currentPos.distanceTo(lastLoggedPositionRef.current) > 0.1) {
        lastLoggedPositionRef.current = currentPos.clone();
      }
      
      // Update CameraAnimator if it exists (for Presentation Script animations or camera animation requests)
      // Also update if CameraAnimator is currently animating (even after request is cleared)
      if (cameraAnimatorRef.current && 
          (presentationState.isPlaying || cameraAnimationRequest || cameraAnimatorRef.current.isAnimating())) {
        cameraAnimatorRef.current.update(delta);
        if (controlsRef.current) {
          controlsRef.current.update();
        }
        return; // Skip legacy animation when using CameraAnimator
      }
      
      // Legacy animation (for non-presentation camera movements)
      // Only animate if animation is active and user is not manually controlling
      if (isAnimating && !isUserInteractingRef.current) {
        const distance = currentPos.distanceTo(targetPosition);
        
        if (distance > 0.01) {
          // Use delta time for consistent animation speed regardless of framerate
          // Animation speed: 2.0 units per second
          const animationSpeed = 2.0;
          const lerpFactor = Math.min(0.15, delta * animationSpeed);
          currentPos.lerp(targetPosition, lerpFactor);
          
          // Update controls target to maintain smooth interaction
          if (controlsRef.current) {
            // Temporarily disable controls during animation to prevent interference
            controlsRef.current.update();
          }
        } else {
          // Animation complete - snap to final position
          currentPos.copy(targetPosition);
          setIsAnimating(false);
          
          // Re-enable controls after animation
          if (controlsRef.current) {
            controlsRef.current.update();
          }
        }
      } else {
        // Update controls normally when not animating
        if (controlsRef.current) {
          controlsRef.current.update();
        }
      }
    }
  });

  // Update camera FOV based on zoom (zoom: 0-100, FOV: 75-30)
  useEffect(() => {
    if (cameraRef.current) {
      const fov = 75 - (cameraZoom / 100) * 45; // 75 to 30 degrees
      cameraRef.current.fov = fov;
      cameraRef.current.updateProjectionMatrix();
    }
  }, [cameraZoom]);

  // Handle model load completion - use useCallback to prevent recreation on every render
  const handleModelLoaded = useCallback(() => {
    if (vehicleId) {
      const viewer = getViewerByVehicleId(vehicleId);
      if (viewer && !viewer.isModelLoaded) {
        console.log(`[Scene] Model loaded for vehicle ${vehicleId}, updating viewer ${viewer.id}`);
        updateViewer(viewer.id, { isModelLoaded: true });
      } else if (!viewer) {
        console.warn(`[Scene] Viewer not found for vehicleId: ${vehicleId}`);
      }
      // If already loaded, don't update (prevents infinite loop)
    } else {
      console.warn(`[Scene] vehicleId not provided, cannot update model loaded state`);
    }
  }, [vehicleId, getViewerByVehicleId, updateViewer]);

  // Reset model loaded state when sourcePath changes
  useEffect(() => {
    if (vehicleId && sourcePath) {
      const viewer = getViewerByVehicleId(vehicleId);
      if (viewer && viewer.isModelLoaded) {
        // Only reset if it was previously loaded
        updateViewer(viewer.id, { isModelLoaded: false });
      }
    }
  }, [sourcePath]); // Only depend on sourcePath, not on functions

  // Update camera preset position (only when preset actually changes)
  useEffect(() => {
    if (cameraRef.current && isInitialized && lastPresetRef.current !== cameraPreset) {
      // Fallback hardcoded positions (used if JSON presets not available)
      const fallbackPositions: Record<CameraPreset, [number, number, number]> = {
        initial: [5, 3, 5],
        front: [0, 2, 8],
        back: [0, 2, -8],
        left: [-8, 2, 0],
        right: [8, 2, 0],
        top: [0, 10, 0],
        front_windshield: [0, 1.5, 4],
        rear_windshield: [0, 1.5, -4],
      };
      
      let targetPos: [number, number, number];
      
      // Try to use JSON camera preset data first
      if (cameraPresetsData && cameraPresetsData[cameraPreset]) {
        const presetData = cameraPresetsData[cameraPreset];
        const jsonPosition = presetData.camera_position as [number, number, number];
        // Transform from JSON coordinate system to PLY coordinate system
        targetPos = transformToPLY(jsonPosition);
      } else {
        // Fallback to hardcoded positions
        targetPos = fallbackPositions[cameraPreset] || [8, 2, 0];
      }
      
      const newTarget = new THREE.Vector3(...targetPos);
      const currentPos = cameraRef.current.position;
      
      // Only animate if position actually changed
      if (currentPos.distanceTo(newTarget) > 0.01) {
        setTargetPosition(newTarget);
        setIsAnimating(true);
        lastPresetRef.current = cameraPreset;
        // Update cameraPosition ref to match preset
        lastCameraPositionRef.current = targetPos;
      }
    }
  }, [cameraPreset, isInitialized, cameraPresetsData]);

  return (
    <>
      <PerspectiveCamera 
        ref={cameraRef} 
        makeDefault 
        fov={75 - (cameraZoom / 100) * 45}
        near={0.1}
        far={1000}
      />
      <OrbitControls 
        ref={controlsRef}
        target={cameraTarget ? new THREE.Vector3(...cameraTarget) : undefined}
        enableDamping 
        dampingFactor={0.05}
        minDistance={1}
        maxDistance={100}
        maxPolarAngle={Math.PI / 2}  // Limit camera to upper hemisphere (prevent going below XZ plane)
        minPolarAngle={0}  // Allow camera to go straight up
        onStart={() => {
          // User started interacting - mark as user interaction
          isUserInteractingRef.current = true;
          
          // Stop any ongoing animations
          if (isAnimating) {
            setIsAnimating(false);
            if (cameraRef.current) {
              const currentPos = cameraRef.current.position;
              setTargetPosition(currentPos.clone());
            }
          }
          
          // Stop CameraAnimator if active
          if (cameraAnimatorRef.current) {
            cameraAnimatorRef.current.stop();
          }
          
          // Clear any existing timeout
          if (userInteractionTimeoutRef.current) {
            clearTimeout(userInteractionTimeoutRef.current);
          }
        }}
        onChange={() => {
          // Update last position ref when user controls camera
          if (cameraRef.current && isUserInteractingRef.current) {
            const currentPos = cameraRef.current.position;
            lastCameraPositionRef.current = [
              currentPos.x,
              currentPos.y,
              currentPos.z,
            ] as [number, number, number];
            
            // Log camera position when user manually controls
            lastLoggedPositionRef.current = currentPos.clone();
            console.log(`[Scene] Camera position changed:`, currentPos);
          }
        }}
        onEnd={() => {
          // User finished interacting - clear flag after a short delay
          // This prevents animation from restarting immediately after user interaction
          if (userInteractionTimeoutRef.current) {
            clearTimeout(userInteractionTimeoutRef.current);
          }
          userInteractionTimeoutRef.current = setTimeout(() => {
            isUserInteractingRef.current = false;
          }, 100);
        }}
      />

      <ambientLight intensity={0.5} />

      {/* <Grid args={[10, 10]} cellColor="#6f6f6f" sectionColor="#9d4b4b" fadeDistance={50} fadeStrength={1} /> */}
      {/* X, Y, Z 축 표시 */}
      {/* <axesHelper args={[5]} /> */}
      
      <directionalLight
        position={[10, 10, 5]}
        intensity={1}
        castShadow
        shadow-mapSize-width={2048}
        shadow-mapSize-height={2048}
      />
      <pointLight position={[-10, -10, -5]} intensity={0.3} />

      {/* Render model using Spark.js */}
      <Suspense fallback={null}>
        {sourcePath && <SparkSplat key={sourcePath} path={sourcePath} onLoaded={handleModelLoaded} />}
      </Suspense>
    </>
  );
}
