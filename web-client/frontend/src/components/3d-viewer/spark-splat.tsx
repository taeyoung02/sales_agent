// web-client/frontend/src/components/3d-viewer/spark-splat.tsx

import { getFullPath } from "@/lib/utils";
import { SplatMesh } from "@sparkjsdev/spark";
import { useEffect, useRef, useState } from "react";
import * as THREE from "three";

// 전역 메모리 캐시: 파일 경로 -> 로드된 mesh
const meshCache = new Map<string, SplatMesh>();
const meshLoadingPromises = new Map<string, Promise<SplatMesh>>();

function SparkSplat({ path, onLoaded }: { path: string | null; onLoaded?: () => void }) {
  const fullPath = path ? getFullPath(path) : null;

  const meshRef = useRef<SplatMesh | null>(null);
  const [isReady, setIsReady] = useState(false);
  const previousPathRef = useRef<string | null>(null);
  const onLoadedCalledRef = useRef<boolean>(false); // Track if onLoaded has been called for current path
  
  useEffect(() => {
    // Reset onLoaded flag when path changes
    onLoadedCalledRef.current = false;
    
    // If path is null or empty, don't load anything
    if (!fullPath) {
      if (meshRef.current) {
        // 캐시에서 제거하지 않고 유지 (다음에 재사용)
        meshRef.current = null;
      }
      setIsReady(false);
      previousPathRef.current = null;
      return;
    }

    // If path hasn't changed, don't reload
    // if (previousPathRef.current === fullPath && meshRef.current) {
    // console.log(3333)

    //   return;
    // }

    // 캐시에서 확인
    // const cachedMesh = meshCache.get(fullPath);
    // if (cachedMesh) {
    //   console.log('✅ Using cached mesh for:', fullPath);
    //   meshRef.current = cachedMesh;
    //   setIsReady(true);
    //   previousPathRef.current = fullPath;
    //   onLoaded?.();
    //   return;
    // }

    // 이미 로딩 중인 경우 대기
    // const loadingPromise = meshLoadingPromises.get(fullPath);
    // if (loadingPromise) {
    //   console.log('⏳ Waiting for mesh to load:', fullPath);
    //   loadingPromise.then((mesh) => {
    //     if (previousPathRef.current === fullPath) {
    //       console.log(4444)
    //       meshRef.current = mesh;
    //       setIsReady(true);
    //       onLoaded?.();
    //     }
    //   });

    //   console.log(5555)

    //   previousPathRef.current = fullPath;
    //   console.log(6666, previousPathRef.current)
    //   return;
    // }

    // 새로 로드
    let isMounted = true;
    let loadCheckInterval: NodeJS.Timeout | null = null;
    let maxTimeout: NodeJS.Timeout | null = null;
    
    console.log('🔄 Loading new SplatMesh from:', fullPath);
    setIsReady(false);
    previousPathRef.current = fullPath;
    
    // Create loading promise
    const loadPromise = new Promise<SplatMesh>((resolve, reject) => {
      try {
        // Create Spark.js SplatMesh
        const mesh = new SplatMesh({ url: fullPath });
        
        // Initial position
        mesh.position.set(0, 0, 0);
        mesh.quaternion.set(1, 0, 0, 0);
        
        // Handle loading completion
        const checkLoaded = () => {
          if (!isMounted) return;
          
          if (mesh.children && mesh.children.length > 0) {
            // Model loaded successfully
            // Keep model at original position (do not center to origin)
            // Camera will use point_cloud_pruned_pose.json center as target
            console.log('SplatMesh loaded at original position:', fullPath);
            
            // 캐시에 저장
            meshCache.set(fullPath, mesh);
            meshLoadingPromises.delete(fullPath);
            
            if (isMounted && previousPathRef.current === fullPath && !onLoadedCalledRef.current) {
              meshRef.current = mesh;
              setIsReady(true);
              onLoadedCalledRef.current = true; // Mark as called to prevent duplicate calls
              onLoaded?.();
            }
            
            resolve(mesh);
            if (loadCheckInterval) {
              clearInterval(loadCheckInterval);
              loadCheckInterval = null;
            }
            if (maxTimeout) {
              clearTimeout(maxTimeout);
              maxTimeout = null;
            }
            return;
          }
        };
        
        // Start checking periodically (더 빠른 체크)
        loadCheckInterval = setInterval(checkLoaded, 50); // 100ms -> 50ms
        
        // Timeout
        maxTimeout = setTimeout(() => {
          console.log('⏱️ timeout fired for', fullPath);  
          if (isMounted && !onLoadedCalledRef.current) {
            console.log('⏱️ SplatMesh load timeout, using mesh anyway:', fullPath);
            meshCache.set(fullPath, mesh);
            meshLoadingPromises.delete(fullPath);
            
            if (previousPathRef.current === fullPath) {
              meshRef.current = mesh;
              setIsReady(true);
              onLoadedCalledRef.current = true; // Mark as called to prevent duplicate calls
              onLoaded?.();
            }
            
            resolve(mesh);
            if (loadCheckInterval) {
              clearInterval(loadCheckInterval);
              loadCheckInterval = null;
            }
          }
        }, 3000); // 5초 -> 3초
        
      } catch (error) {
        meshLoadingPromises.delete(fullPath);
        reject(error);
      }
    });
    
    meshLoadingPromises.set(fullPath, loadPromise);
    
    return () => {
      isMounted = false;
      if (loadCheckInterval) {
        clearInterval(loadCheckInterval);
      }
      if (maxTimeout) {
        clearTimeout(maxTimeout);
      }
      
      // Cleanup: mesh는 캐시에 남겨두고 ref만 정리
      if (previousPathRef.current !== fullPath) {
        meshRef.current = null;
      }
    };
  }, [fullPath]); // Remove onLoaded from dependencies to prevent infinite loop
  
  // Render Spark.js mesh as a primitive in react-three/fiber
  if (!fullPath || !meshRef.current) {
    return null;
  }
  
  return <primitive object={meshRef.current} />;
}

export default SparkSplat;