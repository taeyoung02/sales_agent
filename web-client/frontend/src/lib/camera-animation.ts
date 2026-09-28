/**
 * Camera Animator for smooth, duration-based camera animations
 * Supports preset transitions, zoom, and rotation with easing functions
 */

import * as THREE from "three";
import { CameraPreset } from "./types";

export type EasingFunction = "linear" | "easeIn" | "easeOut" | "easeInOut";

interface AnimationState {
  isAnimating: boolean;
  startTime: number;
  duration: number;
  startPosition: THREE.Vector3;
  targetPosition: THREE.Vector3;
  startZoom: number;
  targetZoom: number;
  easing: EasingFunction;
  orbitCenter?: THREE.Vector3;
}

export class CameraAnimator {
  private camera: THREE.PerspectiveCamera;
  private controls: any; // OrbitControls
  private animationState: AnimationState | null = null;
  private rotationAnimation: {
    isAnimating: boolean;
    startTime: number;
    duration: number;
    startRotation: THREE.Euler;
    targetLookAt: THREE.Vector3;
    easing: EasingFunction;
  } | null = null;

  constructor(camera: THREE.PerspectiveCamera, controls: any) {
    this.camera = camera;
    this.controls = controls;
  }

  /**
   * Animate camera to a preset position
   * targetZoom: 0-100 (optional) - 위치 이동과 줌을 동일 duration으로 보간
   */
  animateToPreset(
    preset: CameraPreset,
    duration: number = 2.0,
    easing: EasingFunction = "easeInOut",
    targetZoom?: number
  ): void {
    const presetPositions: Record<CameraPreset, [number, number, number]> = {
      initial: [5, 3, 5],
      front: [0, 2, 8],
      back: [0, 2, -8],
      left: [-8, 2, 0],
      right: [8, 2, 0],
      top: [0, 10, 0],
      front_windshield: [0, 2, 5],
      rear_windshield: [0, 2, -5]
    };

    const target = presetPositions[preset];
    if (!target || !Array.isArray(target) || target.length !== 3) {
      console.error(`[CameraAnimator] Invalid preset: ${preset}`);
      return;
    }
    this.animateToPosition(target, duration, easing, undefined, targetZoom);
  }

  /**
   * Animate camera to a specific position
   * targetZoom: 0-100 (optional) - 위치 이동과 줌을 동일 duration으로 보간
   */
  animateToPosition(
    position: [number, number, number],
    duration: number = 2.0,
    easing: EasingFunction = "easeInOut",
    orbitCenter?: [number, number, number], // 있으면 중심을 기준으로 원호 이동
    targetZoom?: number
  ): void {
    if (!this.camera) return;

    // 위치 애니메이션 시작 시, 진행 중이던 회전 애니메이션은 중단하여 최종 각도 튐 방지
    if (this.rotationAnimation) {
      this.rotationAnimation.isAnimating = false;
    }

    const targetPosition = new THREE.Vector3(...position);
    const startPosition = this.camera.position.clone();
    const startZoom = this.getCurrentZoom();
    const nextTargetZoom = targetZoom !== undefined ? Math.max(0, Math.min(100, targetZoom)) : startZoom;

    // If already animating, start from current position
    if (this.animationState?.isAnimating) {
      this.animationState.startPosition = this.camera.position.clone();
      this.animationState.startZoom = this.getCurrentZoom();
    }

    // Convert orbitCenter to THREE.Vector3 if provided
    const orbitCenterVec = orbitCenter
      ? new THREE.Vector3(...orbitCenter)
      : undefined;

    this.animationState = {
      isAnimating: true,
      startTime: performance.now(),
      duration: duration * 1000, // Convert to milliseconds
      startPosition,
      targetPosition,
      startZoom,
      targetZoom: nextTargetZoom,
      easing,
      orbitCenter: orbitCenterVec,
    };
  }

  /**
   * Animate camera zoom level
   */
  animateZoom(
    level: number,
    duration: number = 1.0,
    easing: EasingFunction = "easeInOut"
  ): void {
    if (!this.camera) return;

    const startZoom = this.getCurrentZoom();
    const targetZoom = Math.max(0, Math.min(100, level));

    // If already animating, update zoom animation
    if (this.animationState?.isAnimating) {
      this.animationState.startZoom = this.getCurrentZoom();
      this.animationState.targetZoom = targetZoom;
      this.animationState.startTime = performance.now();
      this.animationState.duration = duration * 1000;
      this.animationState.easing = easing;
    } else {
      this.animationState = {
        isAnimating: true,
        startTime: performance.now(),
        duration: duration * 1000,
        startPosition: this.camera.position.clone(),
        targetPosition: this.camera.position.clone(), // Keep position unchanged
        startZoom,
        targetZoom,
        easing,
      };
    }
  }

  /**
   * Animate camera rotation to look at a target point
   */
  animateRotation(
    target: [number, number, number],
    duration: number = 2.0,
    easing: EasingFunction = "easeInOut"
  ): void {
    if (!this.camera || !this.controls) return;

    // Validate target parameter
    if (!target || !Array.isArray(target) || target.length !== 3) {
      console.error(`[CameraAnimator] Invalid target:`, target);
      return;
    }

    // Validate that all elements are numbers
    if (!target.every((val) => typeof val === "number" && !isNaN(val))) {
      console.error(`[CameraAnimator] Target contains invalid numbers:`, target);
      return;
    }

    const targetLookAt = new THREE.Vector3(...target);
    const startRotation = this.camera.rotation.clone();

    this.rotationAnimation = {
      isAnimating: true,
      startTime: performance.now(),
      duration: duration * 1000,
      startRotation,
      targetLookAt,
      easing,
    };
  }

  /**
   * Update animation (should be called in useFrame or animation loop)
   */
  update(deltaTime?: number): void {
    const currentTime = performance.now();

    // Update position/zoom animation
    if (this.animationState?.isAnimating) {
      const elapsed = currentTime - this.animationState.startTime;
      const progress = Math.min(elapsed / this.animationState.duration, 1.0);
      const easedProgress = this.applyEasing(progress, this.animationState.easing);

      if (progress < 1.0) {
        // Interpolate position
        let currentPosition: THREE.Vector3;
        
        // 원호 이동: orbitCenter가 있으면 중심을 기준으로 원호 이동, 없으면 선형 보간
        if (this.animationState.orbitCenter) {
          const center = this.animationState.orbitCenter;
          const startVec = this.animationState.startPosition.clone().sub(center);
          const targetVec = this.animationState.targetPosition.clone().sub(center);

          // 평면 반경 및 각도
          const startPlanar = new THREE.Vector2(startVec.x, startVec.z);
          const targetPlanar = new THREE.Vector2(targetVec.x, targetVec.z);
          const startRadius = startPlanar.length();
          const targetRadius = targetPlanar.length();
          // shot 변경 시 거리(A)가 달라도 반경을 시간에 따라 보간하여 튐을 방지
          const interpolatedRadius = THREE.MathUtils.lerp(
            startRadius,
            targetRadius,
            easedProgress
          );
          const useRadius = interpolatedRadius > 1e-4 ? interpolatedRadius : targetRadius;

          const startAngle = Math.atan2(startPlanar.y, startPlanar.x);
          const targetAngle = Math.atan2(targetPlanar.y, targetPlanar.x);
          let delta = targetAngle - startAngle;
          // 가장 짧은 경로
          if (delta > Math.PI) delta -= 2 * Math.PI;
          if (delta < -Math.PI) delta += 2 * Math.PI;
          const angle = startAngle + delta * easedProgress;

          const planarX = Math.cos(angle) * useRadius;
          const planarZ = Math.sin(angle) * useRadius;

          const height = THREE.MathUtils.lerp(startVec.y, targetVec.y, easedProgress);

          currentPosition = new THREE.Vector3(
            center.x + planarX,
            center.y + height,
            center.z + planarZ
          );
        } else {
          // 선형 보간
          currentPosition = this.animationState.startPosition.clone().lerp(
            this.animationState.targetPosition,
            easedProgress
          );
        }
        this.camera.position.copy(currentPosition);

        // 원호 이동 시에는 매 프레임 중심을 바라보도록 강제
        if (this.animationState.orbitCenter) {
          this.camera.lookAt(this.animationState.orbitCenter);
          if (this.controls) {
            this.controls.target.copy(this.animationState.orbitCenter);
            this.controls.update();
          }
        }
        
        // Log camera position during animation (throttled - only log at key progress points)
        if (Math.floor(progress * 10) !== Math.floor((progress - 0.01) * 10)) {
          console.log(`[Camera Position - Animation] x: ${currentPosition.x.toFixed(2)}, y: ${currentPosition.y.toFixed(2)}, z: ${currentPosition.z.toFixed(2)} (progress: ${(progress * 100).toFixed(1)}%)`);
        }

        // Interpolate zoom
        const currentZoom = THREE.MathUtils.lerp(
          this.animationState.startZoom,
          this.animationState.targetZoom,
          easedProgress
        );
        this.setZoom(currentZoom);

        // Update controls
        if (this.controls) {
          this.controls.update();
        }
      } else {
        // Animation complete
        this.camera.position.copy(this.animationState.targetPosition);
        this.setZoom(this.animationState.targetZoom);
        // 완료 시에도 중심을 바라보도록 정리
        if (this.animationState.orbitCenter) {
          this.camera.lookAt(this.animationState.orbitCenter);
          if (this.controls) {
            this.controls.target.copy(this.animationState.orbitCenter);
            this.controls.update();
          }
        }
        console.log(`[Camera Position - Animation Complete] x: ${this.animationState.targetPosition.x.toFixed(2)}, y: ${this.animationState.targetPosition.y.toFixed(2)}, z: ${this.animationState.targetPosition.z.toFixed(2)}`);
        this.animationState.isAnimating = false;
        if (this.controls) {
          this.controls.update();
        }
      }
    }

    // Update rotation animation
    if (this.rotationAnimation?.isAnimating) {
      const elapsed = currentTime - this.rotationAnimation.startTime;
      const progress = Math.min(elapsed / this.rotationAnimation.duration, 1.0);
      const easedProgress = this.applyEasing(progress, this.rotationAnimation.easing);

      if (progress < 1.0) {
        // Calculate direction to target
        const direction = new THREE.Vector3()
          .subVectors(this.rotationAnimation.targetLookAt, this.camera.position)
          .normalize();

        // Calculate target rotation
        const targetRotation = new THREE.Euler().setFromQuaternion(
          new THREE.Quaternion().setFromUnitVectors(
            new THREE.Vector3(0, 0, -1), // Camera forward direction
            direction
          )
        );

        // Interpolate rotation
        const currentRotation = new THREE.Euler();
        currentRotation.x = THREE.MathUtils.lerp(
          this.rotationAnimation.startRotation.x,
          targetRotation.x,
          easedProgress
        );
        currentRotation.y = THREE.MathUtils.lerp(
          this.rotationAnimation.startRotation.y,
          targetRotation.y,
          easedProgress
        );
        currentRotation.z = THREE.MathUtils.lerp(
          this.rotationAnimation.startRotation.z,
          targetRotation.z,
          easedProgress
        );

        this.camera.rotation.copy(currentRotation);

        // Update controls to look at target
        if (this.controls) {
          this.controls.target.copy(this.rotationAnimation.targetLookAt);
          this.controls.update();
        }
      } else {
        // Animation complete
        const direction = new THREE.Vector3()
          .subVectors(this.rotationAnimation.targetLookAt, this.camera.position)
          .normalize();
        const targetRotation = new THREE.Euler().setFromQuaternion(
          new THREE.Quaternion().setFromUnitVectors(
            new THREE.Vector3(0, 0, -1),
            direction
          )
        );
        this.camera.rotation.copy(targetRotation);
        if (this.controls) {
          this.controls.target.copy(this.rotationAnimation.targetLookAt);
          this.controls.update();
        }
        this.rotationAnimation.isAnimating = false;
      }
    }
  }

  /**
   * Stop all animations
   */
  stop(): void {
    if (this.animationState) {
      this.animationState.isAnimating = false;
    }
    if (this.rotationAnimation) {
      this.rotationAnimation.isAnimating = false;
    }
  }

  /**
   * Check if currently animating
   */
  isAnimating(): boolean {
    return (
      (this.animationState?.isAnimating ?? false) ||
      (this.rotationAnimation?.isAnimating ?? false)
    );
  }

  /**
   * Apply easing function to progress (0-1)
   */
  private applyEasing(t: number, easing: EasingFunction): number {
    switch (easing) {
      case "linear":
        return t;
      case "easeIn":
        return t * t;
      case "easeOut":
        return t * (2 - t);
      case "easeInOut":
        return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
      default:
        return t;
    }
  }

  /**
   * Get current zoom level (0-100)
   */
  private getCurrentZoom(): number {
    if (!this.camera) return 50;
    // FOV: 75 (zoom 0) to 30 (zoom 100)
    const fov = this.camera.fov;
    return ((75 - fov) / 45) * 100;
  }

  /**
   * Set zoom level (0-100)
   */
  private setZoom(level: number): void {
    if (!this.camera) return;
    const clampedLevel = Math.max(0, Math.min(100, level));
    const fov = 75 - (clampedLevel / 100) * 45; // 75 to 30 degrees
    this.camera.fov = fov;
    this.camera.updateProjectionMatrix();
  }
}

