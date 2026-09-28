"use client";

import { VanillaChatPanel } from "@/components/vanilla-chat-panel";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useEffect, useState } from "react";

const PARTICIPANT_ID_KEY = "vanilla_chat_participant_id";

/**
 * URL 쿼리 파라미터에서 participant_id 읽어서 sessionStorage에 저장
 * Google Form에서 링크를 통해 들어올 때 사용
 */
function initializeParticipantMetadata(): void {
  if (typeof window === "undefined") return;

  const urlParams = new URLSearchParams(window.location.search);
  
  // participant_id 처리
  const participantId = urlParams.get("participant_id");
  if (participantId) {
    // URL에 있으면 sessionStorage에 저장 (기존 값이 없을 때만)
    if (!sessionStorage.getItem(PARTICIPANT_ID_KEY)) {
      sessionStorage.setItem(PARTICIPANT_ID_KEY, participantId);
      console.log(`[VanillaChatPage] Participant ID from URL: ${participantId}`);
    }
  }
}

export default function VanillaChatPage() {
  const [showEmailModal, setShowEmailModal] = useState<boolean>(false);
  const [emailInput, setEmailInput] = useState<string>("");

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
      console.log(`[VanillaChatPage] Participant ID saved: ${email}`);
    }
    
    // 모달 닫기
    setShowEmailModal(false);
    setEmailInput("");
  };

  return (
    <div className="flex h-screen w-full flex-col">
      <div className="flex-1 overflow-hidden">
        <VanillaChatPanel locale="ko" />
      </div>
    </div>
  );
}

