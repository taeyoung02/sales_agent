"use client";

import { CameraPreset } from "@/lib/types";
import { Canvas } from "@react-three/fiber";
import { Suspense, useEffect, useState } from "react";
import Scene from "./3d-viewer/scene";

interface ThreeViewerProps {
  sourcePath: string | null;
  cameraPreset: CameraPreset;
  cameraZoom?: number;
  cameraPosition?: [number, number, number];
  cameraTarget?: [number, number, number];
  cameraUp?: [number, number, number];
  vehicleId?: string;
  onLoaded?: () => void;
}

export function ThreeViewer({
  sourcePath,
  cameraPreset,
  cameraZoom = 50,
  cameraPosition = [5, 3, 5],
  cameraTarget,
  cameraUp,
  vehicleId,
  onLoaded,
}: ThreeViewerProps) {
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    if (!isLoaded) {
      setIsLoaded(true);
      onLoaded?.();
    }
  }, [isLoaded, onLoaded]);

  return (
    <div 
      className="h-full w-full rounded-xl relative overflow-hidden"
      style={{
        background: `
          radial-gradient(circle at 30% 30%, rgba(99, 39, 242, 0.15) 0%, transparent 50%),
          radial-gradient(circle at 70% 70%, rgba(139, 92, 246, 0.2) 0%, transparent 50%),
          radial-gradient(circle at 50% 50%, rgba(99, 39, 242, 0.15) 0%, rgba(99, 39, 242, 0.08) 40%, rgba(255, 255, 255, 0.95) 80%, rgba(255, 255, 255, 0.98) 100%),
          linear-gradient(135deg, rgba(99, 39, 242, 0.05) 0%, rgba(255, 255, 255, 0.98) 100%)
        `,
      }}
    >
      {/* <img src="/background5.png" alt="3d-viewer-bg" className="absolute top-0 left-0 w-full h-[100%] object-cover" /> */}
      <Canvas shadows>
        <Suspense fallback={null}>
          <Scene 
            sourcePath={sourcePath}
            cameraPreset={cameraPreset}
            cameraZoom={cameraZoom}
            cameraPosition={cameraPosition}
            cameraTarget={cameraTarget}
            cameraUp={cameraUp}
            vehicleId={vehicleId}
          />
        </Suspense>
      </Canvas>
    </div>
  );
}
