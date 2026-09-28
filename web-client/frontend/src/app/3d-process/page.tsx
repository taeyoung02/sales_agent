"use client"

import { useState, useCallback, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Progress } from "@/components/ui/progress"
import { ThemeToggle } from "@/components/theme-toggle"
import { 
  Upload, 
  ChevronDown, 
  ChevronUp, 
  CheckCircle2, 
  XCircle, 
  Loader2,
  FileImage,
  Video
} from "lucide-react"
import { cn } from "@/lib/utils"

type ProcessStatus = "pending" | "processing" | "success" | "error"

interface ProcessStep {
  id: string
  title: string
  status: ProcessStatus
  progress: number
  logs: string[]
  showLogs: boolean
}

type DataType = "image" | "video"

// 이미지 처리 단계 (run_full_pipeline.sh에 맞춤)
const IMAGE_STEPS: Omit<ProcessStep, "showLogs">[] = [
  { id: "preprocess", title: "Phase 0: 이미지 전처리", status: "pending", progress: 0, logs: [] },
  { id: "colmap", title: "Phase 1: COLMAP 카메라 포즈 추정", status: "pending", progress: 0, logs: [] },
  { id: "rembg", title: "Phase 2: 배경 제거", status: "pending", progress: 0, logs: [] },
  { id: "3dgs", title: "Phase 3: 3D Gaussian Splatting 학습", status: "pending", progress: 0, logs: [] },
  { id: "pruning", title: "Phase 3.5: Point Cloud 정리", status: "pending", progress: 0, logs: [] },
  { id: "langsplat", title: "Phase 4: LangSplat 특징 추출", status: "pending", progress: 0, logs: [] },
  { id: "cf3", title: "Phase 5: CF3 Feature Field 학습", status: "pending", progress: 0, logs: [] },
]

// 비디오 처리 단계 (비디오 추출 단계 추가)
const VIDEO_STEPS: Omit<ProcessStep, "showLogs">[] = [
  { id: "video_extract", title: "Phase -1: 비디오에서 이미지 추출", status: "pending", progress: 0, logs: [] },
  ...IMAGE_STEPS,
]

const ThreeProcessPage = () => {
  const [dataType, setDataType] = useState<DataType>("image")
  const [selectedFiles, setSelectedFiles] = useState<FileList | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [steps, setSteps] = useState<ProcessStep[]>([])
  const [jobId, setJobId] = useState<string | null>(null)
  const [serverLogs, setServerLogs] = useState<string[]>([])
  const [currentTqdm, setCurrentTqdm] = useState<{
    percent: number
    current: number
    total: number
    label?: string
  } | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const eventSourceRef = useRef<EventSource | null>(null)
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const logsEndRef = useRef<HTMLDivElement>(null)
  
  // 설정 상태
  const [showSettings, setShowSettings] = useState(true)
  const [projectName, setProjectName] = useState("")
  const [iterations, setIterations] = useState(15000)
  const [maxWidth, setMaxWidth] = useState(3240)
  const [frameInterval, setFrameInterval] = useState(30)
  const [useEsrganPreprocess, setUseEsrganPreprocess] = useState(false)
  const [autoRemoveBackground, setAutoRemoveBackground] = useState(true)
  const [debugMode, setDebugMode] = useState(false)

  // 파일 선택 핸들러
  const handleFileSelect = (files: FileList | null) => {
    if (!files || files.length === 0) return
    
    // 파일 타입 검증
    const file = files[0]
    const isVideo = file.type.startsWith("video/")
    const isImage = file.type.startsWith("image/")
    
    if (!isVideo && !isImage) {
      alert("영상 또는 이미지 파일만 업로드 가능합니다.")
      return
    }
    
    // 비디오인 경우 단일 파일만 허용
    if (isVideo && files.length > 1) {
      alert("비디오 파일은 하나만 업로드 가능합니다.")
      return
    }
    
    setSelectedFiles(files)
    // 파일 타입에 따라 dataType 설정
    setDataType(isVideo ? "video" : "image")
  }

  // 드래그 앤 드롭 핸들러
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    
    const files = e.dataTransfer.files
    handleFileSelect(files)
  }, [])

  // 로그 토글
  const toggleLogs = (stepId: string) => {
    setSteps(prev => 
      prev.map(step => 
        step.id === stepId 
          ? { ...step, showLogs: !step.showLogs }
          : step
      )
    )
  }

  // tqdm 출력 파싱 함수
  const parseTqdmLog = (log: string): {
    percent: number
    current: number
    total: number
    label?: string
  } | null => {
    // tqdm 패턴: "31%|███▏      | 9440/30000 [12:54<59:24,  5.77it/s, Loss=0.0121852, Depth Loss=0.0000000]"
    // 또는 "Training progress:  31%|███▏      | 9440/30000..."
    const tqdmPattern = /(\d+)%\s*\|\s*[█▏▎▍▌▋▊▉\s]+\|\s*(\d+)\/(\d+)/;
    const match = log.match(tqdmPattern);
    
    if (match) {
      const percent = parseInt(match[1], 10);
      const current = parseInt(match[2], 10);
      const total = parseInt(match[3], 10);
      
      // 라벨 추출 (예: "Training progress:")
      const labelMatch = log.match(/^([^:]+):/);
      const label = labelMatch ? labelMatch[1].trim() : undefined;
      
      return { percent, current, total, label };
    }
    
    return null;
  }

  // tqdm 로그인지 확인
  const isTqdmLog = (log: string): boolean => {
    return /(\d+)%\s*\|\s*[█▏▎▍▌▋▊▉\s]+\|\s*(\d+)\/(\d+)/.test(log);
  }

  // 로그 추가 헬퍼
  const addLog = (stepId: string, log: string) => {
    setSteps(prev =>
      prev.map(step =>
        step.id === stepId
          ? { ...step, logs: [...step.logs, log] }
          : step
      )
    )
  }

  // 단계 상태 업데이트 헬퍼
  const updateStepStatus = (stepId: string, status: ProcessStatus, progress?: number) => {
    setSteps(prev =>
      prev.map(step =>
        step.id === stepId
          ? { ...step, status, progress: progress !== undefined ? progress : step.progress }
          : step
      )
    )
  }

  // 서버 로그 폴링
  const pollServerLogs = useCallback(async (jobId: string) => {
    try {
      const response = await fetch(`/api/3d-process/status/${jobId}`)
      if (!response.ok) {
        throw new Error("상태 조회 실패")
      }
      const data = await response.json()

      // 각 단계의 상태 업데이트
      if (data.steps) {
        data.steps.forEach((stepData: any) => {
          updateStepStatus(stepData.id, stepData.status, stepData.progress)
          if (stepData.logs && stepData.logs.length > 0) {
            // 기존 로그와 비교하여 새 로그만 추가
            setSteps(prev => {
              const step = prev.find(s => s.id === stepData.id)
              if (step) {
                const existingLogs = new Set(step.logs)
                const newLogs = stepData.logs.filter((log: string) => !existingLogs.has(log))
                if (newLogs.length > 0) {
                  return prev.map(s =>
                    s.id === stepData.id
                      ? { ...s, logs: [...s.logs, ...newLogs] }
                      : s
                  )
                }
              }
              return prev
            })
          }
        })
      }

      // 전체 서버 로그 업데이트 (tqdm 제외)
      if (data.logs && data.logs.length > 0) {
        // tqdm 로그를 필터링하고 별도로 처리
        const nonTqdmLogs: string[] = []
        let latestTqdm: ReturnType<typeof parseTqdmLog> = null
        
        data.logs.forEach((log: string) => {
          if (isTqdmLog(log)) {
            const tqdmData = parseTqdmLog(log)
            if (tqdmData) {
              latestTqdm = tqdmData
            }
          } else {
            nonTqdmLogs.push(log)
          }
        })
        
        setServerLogs(nonTqdmLogs)
        if (latestTqdm) {
          setCurrentTqdm(latestTqdm)
        }
        
        // 전체 로그 추가 (현재 처리 중인 단계에, tqdm 제외)
        setSteps(prev => {
          const currentStep = prev.find(s => s.status === "processing")
          if (currentStep) {
            const existingLogs = new Set(currentStep.logs)
            const newLogs = nonTqdmLogs.filter((log: string) => !existingLogs.has(log))
            if (newLogs.length > 0) {
              return prev.map(s =>
                s.id === currentStep.id
                  ? { ...s, logs: [...s.logs, ...newLogs] }
                  : s
              )
            }
          }
          return prev
        })
      }

      // 완료 또는 에러 처리
      if (data.status === "completed") {
        setIsProcessing(false)
        setCurrentTqdm(null) // tqdm 초기화
        if (pollingIntervalRef.current) {
          clearInterval(pollingIntervalRef.current)
          pollingIntervalRef.current = null
        }
        if (eventSourceRef.current) {
          eventSourceRef.current.close()
          eventSourceRef.current = null
        }
        alert("3D 변환이 완료되었습니다!")
      } else if (data.status === "error") {
        setIsProcessing(false)
        setCurrentTqdm(null) // tqdm 초기화
        if (pollingIntervalRef.current) {
          clearInterval(pollingIntervalRef.current)
          pollingIntervalRef.current = null
        }
        if (eventSourceRef.current) {
          eventSourceRef.current.close()
          eventSourceRef.current = null
        }
        alert(`처리 중 오류가 발생했습니다: ${data.error || "알 수 없는 오류"}`)
      }
    } catch (error) {
      console.error("상태 조회 오류:", error)
    }
  }, [])

  // 처리 시작
  const startProcessing = async () => {
    if (!selectedFiles || selectedFiles.length === 0) {
      alert("파일을 먼저 선택해주세요.")
      return
    }

    setIsProcessing(true)
    
    // 파일 타입 확인
    const firstFile = selectedFiles[0]
    const isVideo = firstFile.type.startsWith("video/")
    const fileDataType: DataType = isVideo ? "video" : "image"
    
    // 단계 초기화 (비디오인 경우 비디오 추출 단계 포함)
    const stepsToUse = fileDataType === "video" ? VIDEO_STEPS : IMAGE_STEPS
    const initialSteps = stepsToUse.map(step => ({
      ...step,
      showLogs: false,
    }))
    setSteps(initialSteps)
    setServerLogs([]) // 서버 로그 초기화
    setCurrentTqdm(null) // tqdm 초기화

    try {
      // 프로젝트 이름 검증
      const finalProjectName = projectName.trim() || `project_${Date.now()}`
      
      // 파일 업로드
      const formData = new FormData()
      Array.from(selectedFiles).forEach((file) => {
        formData.append("files", file)
      })
      formData.append("project_name", finalProjectName)
      formData.append("use_esrgan_preprocess", useEsrganPreprocess ? "Y" : "N")
      formData.append("auto_remove_background", autoRemoveBackground ? "Y" : "N")
      formData.append("debug_mode", debugMode ? "Y" : "N")
      formData.append("iterations", iterations.toString())
      formData.append("max_width", maxWidth.toString())
      // 비디오 처리 옵션 추가
      formData.append("extract_from_video", isVideo ? "Y" : "N")
      formData.append("frame_interval", frameInterval.toString())

      const response = await fetch("/api/3d-process/start", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        const errorMessage = errorData.detail || errorData.error || `서버 오류: ${response.status}`
        console.error("API Error:", errorData)
        throw new Error(errorMessage)
      }

      const data = await response.json()
      setJobId(data.job_id)

      // SSE로 실시간 로그 수신 시도
      if (data.sse_url) {
        try {
          // 백엔드 URL 구성
          const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://100.93.57.71:8000'
          const sseUrl = `${backendUrl}${data.sse_url}`
          
          const eventSource = new EventSource(sseUrl)
          eventSourceRef.current = eventSource

          eventSource.addEventListener("log", (event) => {
            try {
              const logData = JSON.parse(event.data)
              if (logData.message) {
                const logMessage = logData.message
                
                // tqdm 로그인지 확인
                if (isTqdmLog(logMessage)) {
                  const tqdmData = parseTqdmLog(logMessage)
                  if (tqdmData) {
                    setCurrentTqdm(tqdmData)
                  }
                  // tqdm 로그는 서버 로그에 추가하지 않음
                } else {
                  // 일반 로그는 서버 로그에 추가
                  setServerLogs(prev => [...prev, logMessage])
                  
                  // 현재 처리 중인 단계에 로그 추가
                  setSteps(prev => {
                    const currentStep = prev.find(s => s.status === "processing")
                    if (currentStep) {
                      return prev.map(s =>
                        s.id === currentStep.id
                          ? { ...s, logs: [...s.logs, logMessage] }
                          : s
                      )
                    }
                    return prev
                  })
                }
              }
            } catch (e) {
              console.error("로그 파싱 오류:", e)
            }
          })

          eventSource.addEventListener("step_update", (event) => {
            try {
              const stepData = JSON.parse(event.data)
              if (stepData.step_id && stepData.status) {
                updateStepStatus(stepData.step_id, stepData.status, stepData.progress)
              }
            } catch (e) {
              console.error("단계 업데이트 파싱 오류:", e)
            }
          })

          eventSource.addEventListener("complete", (event) => {
            try {
              const completeData = JSON.parse(event.data)
              setIsProcessing(false)
              setCurrentTqdm(null) // tqdm 초기화
              if (eventSourceRef.current) {
                eventSourceRef.current.close()
                eventSourceRef.current = null
              }
              if (completeData.status === "error") {
                alert(`처리 중 오류가 발생했습니다: ${completeData.error || "알 수 없는 오류"}`)
              } else {
                alert("3D 변환이 완료되었습니다!")
              }
            } catch (e) {
              console.error("완료 이벤트 파싱 오류:", e)
            }
          })

          eventSource.onerror = (error) => {
            console.error("SSE 연결 오류:", error)
            // SSE 실패 시 폴링으로 전환
            if (eventSourceRef.current) {
              eventSourceRef.current.close()
              eventSourceRef.current = null
            }
            pollingIntervalRef.current = setInterval(() => pollServerLogs(data.job_id), 2000)
          }
        } catch (e) {
          console.warn("SSE 연결 실패, 폴링으로 전환:", e)
          // SSE 실패 시 폴링으로 전환
          pollingIntervalRef.current = setInterval(() => pollServerLogs(data.job_id), 2000)
        }
      } else {
        // SSE URL이 없으면 폴링 사용
        pollingIntervalRef.current = setInterval(() => pollServerLogs(data.job_id), 2000)
      }

      // 초기 상태 조회
      await pollServerLogs(data.job_id)
    } catch (error) {
      console.error("처리 시작 오류:", error)
      setIsProcessing(false)
      alert(error instanceof Error ? error.message : "처리 시작에 실패했습니다.")
    }
  }

  // 서버 로그가 업데이트될 때마다 스크롤을 맨 아래로
  useEffect(() => {
    if (logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: "smooth" })
    }
  }, [serverLogs])

  // 컴포넌트 언마운트 시 정리
  useEffect(() => {
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current)
      }
      if (eventSourceRef.current) {
        eventSourceRef.current.close()
      }
    }
  }, [])

  // 상태 아이콘 렌더링
  const renderStatusIcon = (status: ProcessStatus) => {
    switch (status) {
      case "processing":
        return <Loader2 className="h-4 w-4 animate-spin text-blue-500" />
      case "success":
        return <CheckCircle2 className="h-4 w-4 text-green-500" />
      case "error":
        return <XCircle className="h-4 w-4 text-red-500" />
      default:
        return <div className="h-2 w-2 rounded-full bg-gray-300" />
    }
  }

  return (
    <div className="flex h-screen flex-col bg-background">
      {/* 헤더 */}
      <header className="border-b border-border px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">3D Process</h1>
            <p className="text-sm text-muted-foreground">
              2D 이미지/영상을 3D 파일로 자동 변환
            </p>
          </div>
          {/* <ThemeToggle /> */}
        </div>
      </header>

      {/* 메인 컨텐츠 */}
      <main className="flex-1 overflow-auto p-6">
        <div className="mx-auto max-w-7xl space-y-6">
          {/* 파일 업로드 카드 */}
          <Card>
            <CardHeader>
              <CardTitle>파일 업로드</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* 데이터 타입 정보 */}
              <div className="space-y-2">
                <Label>데이터 타입</Label>
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  {dataType === "video" ? (
                    <>
                      <Video className="h-4 w-4" />
                      비디오 (이미지 추출 후 처리)
                    </>
                  ) : (
                    <>
                      <FileImage className="h-4 w-4" />
                      이미지
                    </>
                  )}
                </div>
              </div>

              {/* 드래그 앤 드롭 영역 */}
              <div
                className={cn(
                  "relative flex min-h-[200px] cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed transition-colors",
                  isDragging
                    ? "border-primary bg-primary/10"
                    : "border-border hover:border-primary/50",
                  isProcessing && "pointer-events-none opacity-50"
                )}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => !isProcessing && fileInputRef.current?.click()}
              >
                <Upload className="h-12 w-12 text-muted-foreground mb-4" />
                <p className="text-sm font-medium">
                  파일을 드래그하거나 클릭하여 선택하세요
                </p>
                <p className="text-xs text-muted-foreground mt-2">
                  {dataType === "video" 
                    ? "비디오 파일을 업로드해주세요 (단일 파일만 가능)"
                    : "이미지 파일을 업로드해주세요 (여러 파일 선택 가능)"}
                </p>
                <Input
                  ref={fileInputRef}
                  type="file"
                  className="hidden"
                  accept={dataType === "video" ? "video/*" : "image/*"}
                  onChange={(e) => handleFileSelect(e.target.files)}
                  disabled={isProcessing}
                  multiple={dataType !== "video"}
                />
              </div>

              {/* 선택된 파일 표시 */}
              {selectedFiles && selectedFiles.length > 0 && (
                <div className="rounded-lg bg-muted p-4">
                  <p className="text-sm font-medium mb-2">선택된 파일:</p>
                  <div className="max-h-32 overflow-y-auto">
                    <ul className="space-y-1">
                      {Array.from(selectedFiles).map((file, idx) => (
                        <li key={idx} className="text-sm text-muted-foreground">
                          • {file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}

              {/* 설정 섹션 */}
              <div className="space-y-4 border-t pt-4">
                <div className="flex items-center justify-between">
                  <Label className="text-base font-semibold">처리 설정</Label>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={() => setShowSettings(!showSettings)}
                    disabled={isProcessing}
                  >
                    {showSettings ? (
                      <>
                        <ChevronUp className="h-4 w-4 mr-1" />
                        설정 숨기기
                      </>
                    ) : (
                      <>
                        <ChevronDown className="h-4 w-4 mr-1" />
                        설정 보기
                      </>
                    )}
                  </Button>
                </div>

                {showSettings && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 p-4 rounded-lg bg-muted/50">
                    {/* 프로젝트 이름 */}
                    <div className="space-y-2">
                      <Label htmlFor="project_name">
                        프로젝트 이름 <span className="text-red-500">*</span>
                      </Label>
                      <Input
                        id="project_name"
                        type="text"
                        placeholder="예: tesla_new"
                        value={projectName}
                        onChange={(e) => setProjectName(e.target.value)}
                        disabled={isProcessing}
                      />
                      <p className="text-xs text-muted-foreground">
                        비어있으면 자동 생성됩니다
                      </p>
                    </div>

                    {/* ITERATIONS */}
                    <div className="space-y-2">
                      <Label htmlFor="iterations">학습 반복 횟수 (ITERATIONS)</Label>
                      <Input
                        id="iterations"
                        type="number"
                        min="1000"
                        max="100000"
                        step="1000"
                        value={iterations}
                        onChange={(e) => setIterations(parseInt(e.target.value) || 15000)}
                        disabled={isProcessing}
                      />
                      <p className="text-xs text-muted-foreground">
                        기본값: 15000 (스크립트 고정값)
                      </p>
                    </div>

                    {/* MAX_WIDTH */}
                    <div className="space-y-2">
                      <Label htmlFor="max_width">최대 이미지 너비 (MAX_WIDTH)</Label>
                      <Input
                        id="max_width"
                        type="number"
                        min="512"
                        max="8192"
                        step="256"
                        value={maxWidth}
                        onChange={(e) => setMaxWidth(parseInt(e.target.value) || 3240)}
                        disabled={isProcessing}
                      />
                      <p className="text-xs text-muted-foreground">
                        기본값: 3240 (스크립트 고정값)
                      </p>
                    </div>

                    {/* FRAME_INTERVAL (비디오인 경우만 표시) */}
                    {dataType === "video" && (
                      <div className="space-y-2">
                        <Label htmlFor="frame_interval">프레임 간격 (FRAME_INTERVAL)</Label>
                        <Input
                          id="frame_interval"
                          type="number"
                          min="1"
                          max="100"
                          value={frameInterval}
                          onChange={(e) => setFrameInterval(parseInt(e.target.value) || 30)}
                          disabled={isProcessing}
                        />
                        <p className="text-xs text-muted-foreground">
                          기본값: 30 (스크립트 고정값)
                        </p>
                      </div>
                    )}

                    {/* USE_ESRGAN_PREPROCESS */}
                    <div className="space-y-2">
                      <Label htmlFor="use_esrgan">빛번짐/반사 제거 (ESRGAN 전처리)</Label>
                      <div className="flex items-center gap-2">
                        <input
                          id="use_esrgan"
                          type="checkbox"
                          checked={useEsrganPreprocess}
                          onChange={(e) => setUseEsrganPreprocess(e.target.checked)}
                          disabled={isProcessing}
                          className="h-4 w-4 rounded border-gray-300"
                        />
                        <span className="text-sm text-muted-foreground">
                          {`사용 여부: ${useEsrganPreprocess ? "Y" : "N"} (기본값: N)`}
                        </span>
                      </div>
                    </div>

                    {/* AUTO_REMOVE_BACKGROUND */}
                    <div className="space-y-2">
                      <Label htmlFor="auto_remove_bg">자동 배경 제거</Label>
                      <div className="flex items-center gap-2">
                        <input
                          id="auto_remove_bg"
                          type="checkbox"
                          checked={autoRemoveBackground}
                          onChange={(e) => setAutoRemoveBackground(e.target.checked)}
                          disabled={isProcessing}
                          className="h-4 w-4 rounded border-gray-300"
                        />
                        <span className="text-sm text-muted-foreground">
                          {autoRemoveBackground ? "사용 (Y)" : "사용 안 함 (n)"}
                        </span>
                      </div>
                      <p className="text-xs text-muted-foreground">
                        기본값: 사용 (Y)
                      </p>
                    </div>

                    {/* DEBUG_MODE */}
                    <div className="space-y-2">
                      <Label htmlFor="debug_mode">디버그 모드 (중간 파일 유지)</Label>
                      <div className="flex items-center gap-2">
                        <input
                          id="debug_mode"
                          type="checkbox"
                          checked={debugMode}
                          onChange={(e) => setDebugMode(e.target.checked)}
                          disabled={isProcessing}
                          className="h-4 w-4 rounded border-gray-300"
                        />
                        <span className="text-sm text-muted-foreground">
                          {`사용 여부: ${debugMode ? "Y" : "N"} (기본값: N)`}
                        </span>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* 처리 시작 버튼 */}
              <Button
                onClick={startProcessing}
                disabled={!selectedFiles || isProcessing}
                className="w-full"
                size="lg"
              >
                {isProcessing ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    처리 중...
                  </>
                ) : (
                  "3D 변환 시작"
                )}
              </Button>
            </CardContent>
          </Card>

          {/* 처리 단계 리스트와 서버 로그 */}
          {steps.length > 0 && (
            <div className="flex gap-6 h-[calc(100vh-280px)]">
              {/* 진행 상황 리스트 (2/3 너비) */}
              <div className="flex-[2] min-w-0">
                <Card className="h-full flex flex-col">
                  <CardHeader>
                    <CardTitle>처리 진행 상황</CardTitle>
                  </CardHeader>
                  <CardContent className="flex-1 overflow-y-auto space-y-4">
                    {steps.map((step, idx) => (
                      <div
                        key={step.id}
                        className="rounded-lg border border-border p-4 space-y-3"
                      >
                        {/* 단계 헤더 */}
                        <div className="flex items-center gap-3">
                          {/* 상태 표시 */}
                          <div className="flex-shrink-0">
                            {step.status === "processing" ? (
                              <div className="h-3 w-3 rounded-full bg-green-500 animate-pulse" />
                            ) : (
                              renderStatusIcon(step.status)
                            )}
                          </div>

                          {/* 제목 */}
                          <div className="flex-1 min-w-0">
                            <h3 className="font-medium truncate">{step.title}</h3>
                          </div>

                          {/* 진행률 또는 상태 */}
                          <div className="flex items-center gap-3 flex-shrink-0">
                            {step.status === "processing" && (
                              <span className="text-sm text-muted-foreground whitespace-nowrap">
                                {step.progress}%
                              </span>
                            )}
                            {step.status === "success" && (
                              <span className="text-sm text-green-600 dark:text-green-400 whitespace-nowrap">
                                완료
                              </span>
                            )}
                            {step.status === "error" && (
                              <span className="text-sm text-red-600 dark:text-red-400 whitespace-nowrap">
                                오류
                              </span>
                            )}

                            {/* 로그 토글 버튼 */}
                            {step.logs.length > 0 && (
                              <Button
                                variant="ghost"
                                size="icon-sm"
                                onClick={() => toggleLogs(step.id)}
                              >
                                {step.showLogs ? (
                                  <ChevronUp className="h-4 w-4" />
                                ) : (
                                  <ChevronDown className="h-4 w-4" />
                                )}
                              </Button>
                            )}
                          </div>
                        </div>

                        {/* 진행률 바 */}
                        {(step.status === "processing" || step.status === "success") && (
                          <div className="space-y-2">
                            <Progress value={step.progress} className="h-2" />
                            {/* 현재 단계가 처리 중이고 tqdm 정보가 있으면 표시 */}
                            {step.status === "processing" && currentTqdm && (
                              <div className="space-y-1">
                                {currentTqdm.label && (
                                  <p className="text-xs text-muted-foreground font-medium">
                                    {currentTqdm.label}
                                  </p>
                                )}
                                <div className="flex items-center justify-between text-xs text-muted-foreground">
                                  <span>
                                    {currentTqdm.current.toLocaleString()} / {currentTqdm.total.toLocaleString()} ({currentTqdm.percent}%)
                                  </span>
                                </div>
                                <Progress value={currentTqdm.percent} className="h-1.5" />
                              </div>
                            )}
                          </div>
                        )}

                        {/* 로그 표시 */}
                        {step.showLogs && step.logs.length > 0 && (
                          <div className="rounded-md bg-muted p-3 max-h-40 overflow-y-auto">
                            <div className="space-y-1 font-mono text-xs">
                              {step.logs.map((log, logIdx) => (
                                <div key={logIdx} className="text-muted-foreground break-words">
                                  {log}
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </CardContent>
                </Card>
              </div>

              {/* 서버 로그 (1/3 너비) */}
              <div className="flex-1 min-w-0">
                <Card className="h-full flex flex-col">
                  <CardHeader>
                    <CardTitle>서버 로그</CardTitle>
                  </CardHeader>
                  <CardContent className="flex-1 overflow-hidden flex flex-col">
                    <div className="flex-1 overflow-y-auto rounded-md bg-muted p-3">
                      <div className="space-y-1 font-mono text-xs">
                        {serverLogs.length === 0 ? (
                          <div className="text-muted-foreground italic">
                            로그가 없습니다...
                          </div>
                        ) : (
                          <>
                            {serverLogs
                              .filter(log => !isTqdmLog(log)) // tqdm 로그는 서버 로그에서 제외
                              .map((log, logIdx) => {
                                // 로그 타입에 따라 색상 구분
                                const isError = log.toLowerCase().includes("error") || log.includes("[ERROR]")
                                const isWarning = log.toLowerCase().includes("warning") || log.includes("[WARNING]")
                                const isProgress = log.includes("[Progress]")
                                
                                return (
                                  <div
                                    key={logIdx}
                                    className={cn(
                                      "text-muted-foreground break-words whitespace-pre-wrap",
                                      isError && "text-red-500 dark:text-red-400",
                                      isWarning && "text-yellow-600 dark:text-yellow-400",
                                      isProgress && "text-blue-600 dark:text-blue-400"
                                    )}
                                  >
                                    {log}
                                  </div>
                                )
                              })}
                            <div ref={logsEndRef} />
                          </>
                        )}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  )
}

export default ThreeProcessPage
