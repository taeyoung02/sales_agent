"use client";

import { CameraPresets } from "@/components/camera-presets";
import { ChatPanel } from "@/components/chat-panel";
import { PresentationControls } from "@/components/presentation-controls";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent } from "@/components/ui/tabs";
import { getCameraPoseData, healthCheck } from "@/lib/api";
import { useViewerStore } from "@/lib/state";
import { Evidence } from "@/lib/types";
import { ChevronDown, ChevronUp, X } from "lucide-react";
import dynamic from "next/dynamic";
import { Suspense, useEffect, useState } from "react";

import { calculateInitialCameraView } from "@/lib/camera-utils";
import { PresentationScript } from "@/lib/types";


// PLY 파일 경로 (source 폴더 기준)

const PARTICIPANT_ID_KEY = "chat_participant_id";

const ThreeViewer = dynamic(
  () => import("@/components/three-viewer").then((mod) => mod.ThreeViewer),
  {
    ssr: false,
    loading: () => (
      <div className="flex h-full items-center justify-center bg-muted/30">
        <div className="text-muted-foreground">Loading 3D Viewer...</div>
      </div>
    ),
  }
);

export default function ViewerPage() {
  const {
    viewers,
    show3DViewer,
    setShow3DView,
    removeViewer,
  } = useViewerStore();
  const [evidence, setEvidence] = useState<Evidence[]>([]);
  const [apiStatus, setApiStatus] = useState<"checking" | "online" | "offline">("checking");
  const [isTestToolsOpen, setIsTestToolsOpen] = useState<boolean>(false);
  const [showEmailModal, setShowEmailModal] = useState<boolean>(false);
  const [emailInput, setEmailInput] = useState<string>("");

  const { startPresentation } = useViewerStore();

  const testVehicles = [
    { id: "test", path: "/static/source/test/point_cloud.ply", name: "Test Vehicle" },
    { id: "test", path: "/static/source/test/volvo_with_floor.ply", name: "Test Vehicle with Floor" },
    { id: "test", path: "/static/source/test/volvo_without_floor.ply", name: "Test Vehicle without Floor" },
    { id: "toycar", path: "/static/source/toycar/point_cloud.ply", name: "Toy Car" },
    { id: "super_car", path: "/static/source/super_car/point_cloud.ply", name: "Super Car" },
  ];

  const handleLoadTestVehicle = async (vehicleId: string, path: string) => {
    const state = useViewerStore.getState();
    const existingViewer = state.getViewerByVehicleId(vehicleId);
    
    // Calculate camera position before loading vehicle
    try {
      const cameraPoseData = await getCameraPoseData(vehicleId);
      if (cameraPoseData) {
        const { position, target, up } = calculateInitialCameraView(cameraPoseData);
        console.log("📷 [UI Button] Calculated initial camera view:", { position, target, up });
        state.setCameraPosition(position);
        state.setCameraTarget(target);
        state.setCameraUp(up);
      }
    } catch (error) {
      console.error("Failed to load camera pose data:", error);
      // Continue with default camera if pose data fails
    }
    
    if (existingViewer) {
      state.updateViewer(existingViewer.id, { sourcePath: path });
    } else {
      state.addViewer(vehicleId, path);
    }
    state.setShow3DView(true);
  };

  const testPresentationScript = () => {
    const testScript: PresentationScript = {
      vehicle_id: "WBAZV7EWXRT7F3CL5",
      total_duration: 30,
      segments: [
        {
          timestamp: 0,
          text: "고객님, 지금 보고 계신 차량은 2024년식 볼보 더 뉴 S90 B5 얼티메이트 브라이트 모델입니다.",
          actions: [
            {
              type: "setCamera",
              preset: "initial",
              duration: 2.0
            }
          ]
        },
        {
          timestamp: 3,
          text: "해당 모델은 가족의 안전과 편의성을 최우선으로 생각하는 고객님께 정말 최적의 선택이 될 겁니다.",
          actions: [
            {
              type: "rotateCamera",
              rotate: {
                target: "left",
                shot: "medium",
                angle: "high",
                speed: "medium"
              },
              duration: 2.0
            },
            {
              type: "zoomCamera",
              level: 70,
              duration: 1.5
            }
          ]
        },
        {
          timestamp: 6,
          text: "전면부에는 LED 헤드램프와 어댑티브 헤드램프가 적용되어 있습니다.",
          actions: [
            {
              type: "rotateCamera",
              rotate: {
                target: "front",
                shot: "full",
                angle: "high",
                speed: "medium"
              },
              duration: 2.0
            },
            {
              type: "zoomCamera",
              level: 70,
              duration: 1.5
            }
          ]
        },
        {
          timestamp: 9,
          text: "측면의 유려한 실루엣과 알루미늄 휠이 돋보입니다.",
          actions: [
            {
              type: "rotateCamera",
              rotate: {
                target: "left",
                shot: "closeup",
                angle: "low",
                speed: "medium"
              },
              duration: 2.0
            },
            {
              type: "zoomCamera",
              level: 80,
              duration: 1.5
            }
          ]
        },
        {
          timestamp: 12,
          text: "또한 파노라마 썬루프가 적용되어 있어 실내로 자연광이 풍부하게 들어옵니다.",
          actions: [
            {
              type: "rotateCamera",
              rotate: {
                target: "right",
                shot: "medium",
                angle: "high",
                speed: "slow"
              },
              duration: 2.0
            }
          ]
        },
        {
          timestamp: 14,
          text: "후면부로 이동해보면, 볼보만의 독특한 테일램프 디자인을 확인하실 수 있습니다.",
          actions: [
            {
              type: "setCamera",
              preset: "back",
              duration: 2.0
            }
          ]
        },
        {
          timestamp: 17,
          text: "트렁크 용량은 500리터로 넉넉한 수납 공간을 제공하여 가족 여행에도 충분합니다.",
          actions: [
            {
              type: "setCamera",
              preset: "back",
              duration: 2.0
            },
            {
              type: "zoomCamera",
              level: 90,
              duration: 1.5
            }
          ]
        },
        {
          timestamp: 20,
          text: "다시 전면부를 보시면, 볼보의 상징적인 그릴과 함께 프리미엄한 디자인을 느끼실 수 있습니다.",
          actions: [
            {
              type: "rotateCamera",
              rotate: {
                target: "front",
                shot: "medium",
                angle: "high",
                speed: "medium"
              },
              duration: 2.0
            },
            {
              type: "zoomCamera",
              level: 60,
              duration: 1.5
            }
          ]
        },
        {
          timestamp: 23,
          text: "이 차량은 주행거리 5,321km의 저주행 차량으로, 경미한 외판 사고 이력만 있을 뿐 주요 골격은 무사고 상태입니다.",
          actions: [
            {
              type: "setCamera",
              preset: "initial",
              duration: 2.0
            },
            {
              type: "zoomCamera",
              level: 80,
              duration: 1.5
            }
          ]
        }
      ]
    };
    
    startPresentation(testScript);
  };


  // 이메일 입력 모달 표시 여부 확인 (컴포넌트 마운트 시)
  useEffect(() => {
    if (typeof window !== "undefined") {
      const existingParticipantId = sessionStorage.getItem(PARTICIPANT_ID_KEY);
      // URL 쿼리 파라미터에서 participant_id 확인
      const urlParams = new URLSearchParams(window.location.search);
      const participantIdFromUrl = urlParams.get("participant_id");
      
      // sessionStorage에 participant_id가 없고, URL에도 없으면 모달 표시
      if (!existingParticipantId && !participantIdFromUrl) {
        setShowEmailModal(true);
      } else if (participantIdFromUrl && !existingParticipantId) {
        // URL에 있으면 sessionStorage에 저장
        sessionStorage.setItem(PARTICIPANT_ID_KEY, participantIdFromUrl);
      }
    }
  }, []);

  // 백엔드 API 헬스 체크 (컴포넌트 마운트 시)
  useEffect(() => {
    const checkApiHealth = async () => {
      try {
        const result = await healthCheck();
        if (result.status === "healthy") {
          setApiStatus("online");
        }
      } catch (error) {
        console.error("Backend API is not available:", error);
        setApiStatus("offline");
      }
    };

    checkApiHealth();
  }, []);

  // 이메일 입력 완료 핸들러
  const handleEmailSubmit = () => {
    const email = emailInput.trim();
    if (!email) {
      alert("이메일을 입력해주세요.");
      return;
    }
    
    // 이메일 형식 간단 검증
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      alert("올바른 이메일 형식을 입력해주세요.");
      return;
    }
    
    // sessionStorage에 저장
    if (typeof window !== "undefined") {
      sessionStorage.setItem(PARTICIPANT_ID_KEY, email);
      console.log(`[ViewerPage] Participant ID saved: ${email}`);
    }
    
    // 모달 닫기
    setShowEmailModal(false);
    setEmailInput("");
  };

  // Heatmap 생성 콜백 (개발 모드에서만 사용 - PLY 파일 로드용)
  // 프로덕션에서는 카메라만 이동하므로 이 콜백은 사용하지 않음
  const handleHeatmapGenerated = (path: string) => {
    // 개발 모드에서만 PLY 파일을 로드 (HEATMAP_GENERATE_PLY=true일 때만)
    const isDevMode = process.env.NODE_ENV === 'development';
    if (!isDevMode) {
      return;
    }
    
    // 뷰어가 없으면 새로 생성, 있으면 첫 번째 뷰어 업데이트
    const state = useViewerStore.getState();
    if (state.viewers.length === 0) {
      state.addViewer("", path);
    } else {
      state.updateViewer(state.viewers[0].id, { sourcePath: path });
    }
  };

  // 차량 정보 요청시 3D 뷰 활성화 콜백
  const handleVehicleRequested = (
    vehicleId: string, 
    sourcePath: string,
    compareWithVehicleId?: string,
    compareWithSourcePath?: string
  ) => {
    const state = useViewerStore.getState();
    
    // 첫 번째 차량 로드
    const existingViewer1 = state.getViewerByVehicleId(vehicleId);
    if (existingViewer1) {
      state.updateViewer(existingViewer1.id, { sourcePath });
    } else {
      state.addViewer(vehicleId, sourcePath);
    }
    
    // 두 번째 차량이 있으면 로드 (비교 모드)
    if (compareWithVehicleId && compareWithSourcePath) {
      const existingViewer2 = state.getViewerByVehicleId(compareWithVehicleId);
      if (existingViewer2) {
        state.updateViewer(existingViewer2.id, { sourcePath: compareWithSourcePath });
      } else {
        state.addViewer(compareWithVehicleId, compareWithSourcePath);
      }
    }
    
    state.setShow3DView(true);
  };

  return (
    <div className="flex h-screen flex-col bg-gradient-to-br from-purple-50/50 via-white to-purple-50/30">
      {process.env.NODE_ENV === 'development' && (
        <div className="fixed top-4 right-4 z-50 bg-card border border-border rounded-lg shadow-lg overflow-hidden">
          <button
            onClick={() => setIsTestToolsOpen(!isTestToolsOpen)}
            className="w-full px-4 py-2 flex items-center justify-between hover:bg-accent transition-colors"
          >
            <h3 className="text-sm font-semibold">테스트 도구</h3>
            {isTestToolsOpen ? (
              <ChevronUp className="h-4 w-4" />
            ) : (
              <ChevronDown className="h-4 w-4" />
            )}
          </button>
          {isTestToolsOpen && (
            <div className="p-4 pt-2 flex flex-col gap-2 border-t border-border max-h-[600px] overflow-y-auto">
              {testVehicles.map((vehicle) => (
                <Button
                  key={vehicle.id}
                  onClick={() => handleLoadTestVehicle(vehicle.id, vehicle.path)}
                  variant="outline"
                  size="sm"
                >
                  {vehicle.name} 로드
                </Button>
              ))}
              <Button 
                onClick={testPresentationScript}
                variant="outline"
                size="sm"
                className="mt-2"
              >
                프레젠테이션 테스트
              </Button>
              
              {viewers.length > 0 && viewers[0].vehicleId && (
                <div className="mt-4 pt-4 border-t border-border">
                  <CameraPresets vehicleId={viewers[0].vehicleId} />
                </div>
              )}
            </div>
          )}
        </div>
      )}
      <PresentationControls />
      <div className="flex flex-1 overflow-hidden">
        {/* 3D Viewer 영역 - show3DView가 true일 때만 표시 */}
        {show3DViewer && viewers.length > 0 && (
          <div className="flex flex-1 flex-col p-4 h-full transition-all duration-500 ease-in-out">
            <div className="mb-4">
              <div className="flex items-center justify-between">
                <div>
                  <h1 className="text-2xl font-bold">3D Viewer</h1>
                  <p className="text-sm text-muted-foreground">
                    {viewers.length} viewer{viewers.length > 1 ? 's' : ''} active
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  {/* API 상태 표시 */}
                  <div className="flex items-center gap-2">
                    <div
                      className={`h-2 w-2 rounded-full ${
                        apiStatus === "online"
                          ? "bg-green-500"
                          : apiStatus === "offline"
                          ? "bg-red-500"
                          : "bg-yellow-500"
                      }`}
                    />
                    <span className="text-xs text-muted-foreground">
                      {apiStatus === "online"
                        ? "API Connected"
                        : apiStatus === "offline"
                        ? "API Offline"
                        : "Checking..."}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* 3D 파일이 없거나 못 불러온 경우 메시지 표시 */}
            {viewers.length === 0 || viewers.every(v => !v.sourcePath) ? (
              <div className="flex flex-1 items-center justify-center">
                <p className="text-muted-foreground">차량 3D 파일을 찾지 못 했어요.</p>
              </div>
            ) : (
            /* 뷰어 그리드 - 최대 2개 (세로 스택, 동일 높이) */
            <div className="grid grid-flow-row auto-rows-fr grid-cols-1 gap-4 flex-1 min-h-0">
              {viewers.map((viewer) => (
                <div
                  key={viewer.id}
                  className="relative h-full min-h-0 overflow-hidden rounded-xl"
                >
                  <Suspense fallback={
                    <div className="flex h-full items-center justify-center bg-muted">
                      Loading...
                    </div>
                  }>
                    <ThreeViewer
                      sourcePath={viewer.sourcePath}
                      cameraPreset={viewer.cameraPreset}
                      cameraZoom={viewer.cameraZoom}
                      cameraPosition={viewer.cameraPosition}
                      cameraTarget={viewer.cameraTarget}
                      cameraUp={viewer.cameraUp}
                      vehicleId={viewer.vehicleId || undefined}
                    />
                  </Suspense>
                  {/* 뷰어 닫기 버튼 */}
                  {viewers.length > 0 && (
                    <Button
                      variant="ghost"
                      size="icon"
                      className="absolute top-2 right-2 h-6 w-6"
                      onClick={() => {
                        removeViewer(viewer.id);
                        if (viewers.length === 1) {
                          setShow3DView(false);
                        }
                      }}
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  )}
                </div>
              ))}
            </div>
            )}
          </div>
        )}

        {/* Right Panel - Tabs */}
        <div className={`${show3DViewer ? 'w-[400px] border-l border-border' : 'w-full flex justify-center items-center'} flex flex-col h-full min-h-0 transition-all duration-500 ease-in-out`}>
          <div className={`${show3DViewer ? 'w-full' : 'w-1/2 max-w-2xl'} flex flex-col h-full min-h-0 transition-all duration-500 ease-in-out`}>
            <Tabs defaultValue="chat" className="flex h-full flex-col min-h-0">
              <TabsContent value="chat" className="flex-1 overflow-hidden min-h-0">
                <ChatPanel
                  onEvidenceUpdate={setEvidence}
                  onHeatmapGenerated={handleHeatmapGenerated}
                  onVehicleRequested={handleVehicleRequested}
                  vehicleId={viewers[0]?.vehicleId || undefined}
                />
              </TabsContent>

              <TabsContent
                value="evidence"
                className="flex-1 overflow-hidden px-0 min-h-0"
              >
                {/* <EvidencePanel
                  evidence={[...currentVehicle.evidence, ...evidence]}
                /> */}
              </TabsContent>

              <TabsContent
                value="details"
                className="flex-1 overflow-auto p-4 min-h-0"
              >
                <Card>
                  <CardHeader>
                    <CardTitle>Vehicle Specifications</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    {/* {Object.entries(currentVehicle.specs).map(([key, value]) => (
                      <div
                        key={key}
                        className="flex justify-between border-b border-border pb-2 last:border-0"
                      >
                        <span className="text-sm font-medium capitalize text-muted-foreground">
                          {key.replace(/([A-Z])/g, " $1")}
                        </span>
                        <span className="text-sm font-semibold">{value}</span>
                      </div>
                    ))} */}
                  </CardContent>
                </Card>

                <Card className="mt-4">
                  <CardHeader>
                    <CardTitle>Available Packages</CardTitle>
                  </CardHeader>
                  <CardContent>
                    {/* <div className="flex flex-wrap gap-2">
                      {currentVehicle.packages.map((pkg, i) => (
                        <Button key={i} variant="outline" size="sm">
                          <Package className="mr-2 h-4 w-4" />
                          {pkg}
                        </Button>
                      ))}
                    </div> */}
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          </div>
        </div>
      </div>
    </div>
  );
}
