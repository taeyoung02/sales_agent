'use client'

import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { ScrollArea } from '@/components/ui/scroll-area'
import { getPrompts, updatePrompts, type PromptsResponse, type PromptUpdateRequest } from '@/lib/api'
import { Check, Copy } from 'lucide-react'
import { useEffect, useState } from 'react'

const PROMPT_LABELS: Record<keyof PromptsResponse, string> = {
  sales_knowledge: '세일즈 지식 (Sales Knowledge)',
  common_tool_guidelines: '공통 Tool 가이드라인 (Common Tool Guidelines)',
  planner_tool_guidelines: 'Planner Tool 가이드라인 (Planner Tool Guidelines)',
  planner_role: 'Planner 역할 (Planner Role)',
  evaluator_role: 'Evaluator 역할 (Evaluator Role)',
}

export default function PromptsPage() {
  const [prompts, setPrompts] = useState<PromptsResponse | null>(null)
  const [updates, setUpdates] = useState<PromptUpdateRequest>({})
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)
  const [copiedStates, setCopiedStates] = useState<Record<string, boolean>>({})

  useEffect(() => {
    loadPrompts()
  }, [])

  const loadPrompts = async () => {
    try {
      setLoading(true)
      const data = await getPrompts()
      setPrompts(data)
      setUpdates({})
      setMessage(null)
    } catch (error) {
      setMessage({
        type: 'error',
        text: error instanceof Error ? error.message : '프롬프트를 불러오는데 실패했습니다.',
      })
    } finally {
      setLoading(false)
    }
  }

  const handleUpdate = (key: keyof PromptsResponse, value: string) => {
    setUpdates((prev) => ({
      ...prev,
      [key]: value || undefined,
    }))
  }

  const handleCopy = async (text: string, key: string, type: 'current' | 'update') => {
    try {
      await navigator.clipboard.writeText(text)
      const stateKey = `${key}-${type}`
      setCopiedStates((prev) => ({ ...prev, [stateKey]: true }))
      setTimeout(() => {
        setCopiedStates((prev) => ({ ...prev, [stateKey]: false }))
      }, 2000)
    } catch (error) {
      console.error('복사 실패:', error)
    }
  }

  const handleSave = async () => {
    try {
      setSaving(true)
      setMessage(null)

      // 빈 문자열은 undefined로 변환 (현재 값 유지)
      const updateRequest: PromptUpdateRequest = {}
      for (const [key, value] of Object.entries(updates)) {
        if (value && value.trim()) {
          updateRequest[key as keyof PromptUpdateRequest] = value.trim()
        }
      }

      const result = await updatePrompts(updateRequest)
      setMessage({
        type: 'success',
        text: result.message || '프롬프트가 성공적으로 업데이트되었습니다.',
      })

      // 업데이트 후 최신 프롬프트 다시 로드 (페이지 리로드 대신)
      await loadPrompts()
      
      // updates state 초기화하여 텍스트 영역도 비우기
      setUpdates({})
    } catch (error) {
      setMessage({
        type: 'error',
        text: error instanceof Error ? error.message : '프롬프트 업데이트에 실패했습니다.',
      })
    } finally {
      setSaving(false)
    }
  }
  
  if (loading) {
    return (
      <div className="container mx-auto p-6">
        <div className="flex items-center justify-center min-h-screen">
          <p className="text-muted-foreground">프롬프트를 불러오는 중...</p>
        </div>
      </div>
    )
  }

  if (!prompts) {
    return (
      <div className="container mx-auto p-6">
        <div className="flex items-center justify-center min-h-screen">
          <div className="text-center">
            <p className="text-destructive mb-4">프롬프트를 불러올 수 없습니다.</p>
            <Button onClick={loadPrompts}>다시 시도</Button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="container mx-auto p-6 max-w-7xl">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2">프롬프트 관리</h1>
        <p className="text-muted-foreground">
          각 프롬프트를 업데이트할 수 있습니다. 업데이트하지 않으면 현재 값이 유지됩니다.
        </p>
      </div>

      {message && (
        <div
          className={`mb-4 p-4 rounded-md ${
            message.type === 'success'
              ? 'bg-green-50 text-green-800 border border-green-200'
              : 'bg-red-50 text-red-800 border border-red-200'
          }`}
        >
          {message.text}
        </div>
      )}

      <div className="space-y-6">
        {(Object.keys(prompts) as Array<keyof PromptsResponse>).map((key) => {
          const currentValue = prompts[key]
          const updateValue = updates[key] ?? ''

          return (
            <Card key={key} className="p-6">
              <div className="mb-4">
                <h2 className="text-xl font-semibold mb-2">{PROMPT_LABELS[key]}</h2>
                <p className="text-sm text-muted-foreground mb-4">
                  현재 프롬프트 (읽기 전용)
                </p>
                <div className="relative">
                  <ScrollArea className="h-48 w-full border rounded-md p-4 bg-muted/50">
                    <pre className="text-sm whitespace-pre-wrap font-mono">
                      {currentValue}
                    </pre>
                  </ScrollArea>
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    className="absolute top-2 right-2 h-8 w-8"
                    onClick={() => handleCopy(currentValue, key, 'current')}
                    title="현재 프롬프트 복사"
                  >
                    {copiedStates[`${key}-current`] ? (
                      <Check className="h-4 w-4 text-green-600" />
                    ) : (
                      <Copy className="h-4 w-4" />
                    )}
                  </Button>
                </div>
              </div>

              <div>
                <label className="text-sm font-medium mb-2 block">
                  업데이트할 프롬프트 (비워두면 현재 값 유지)
                </label>
                <div className="relative">
                  <textarea
                    value={updateValue}
                    onChange={(e) => handleUpdate(key, e.target.value)}
                    placeholder="새로운 프롬프트를 입력하세요. 비워두면 현재 값이 유지됩니다..."
                    className="w-full min-h-[200px] p-4 border rounded-md font-mono text-sm resize-y bg-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:border-ring"
                  />
                  {updateValue && (
                    <Button
                      variant="ghost"
                      size="icon-sm"
                      className="absolute top-2 right-2 h-8 w-8"
                      onClick={() => handleCopy(updateValue, key, 'update')}
                      title="업데이트할 프롬프트 복사"
                    >
                      {copiedStates[`${key}-update`] ? (
                        <Check className="h-4 w-4 text-green-600" />
                      ) : (
                        <Copy className="h-4 w-4" />
                      )}
                    </Button>
                  )}
                </div>
                {updateValue && (
                  <p className="text-xs text-muted-foreground mt-2">
                    {updateValue.length}자 입력됨
                  </p>
                )}
              </div>
            </Card>
          )
        })}
      </div>

      <div className="mt-6 flex gap-4 justify-end">
        <Button variant="outline" onClick={loadPrompts} disabled={saving}>
          새로고침
        </Button>
        <Button onClick={handleSave} disabled={saving}>
          {saving ? '저장 중...' : '저장하기'}
        </Button>
      </div>
    </div>
  )
}

