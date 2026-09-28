"""
공통 프롬프트 정의
버전과 무관하게 공유되는 프롬프트들
"""

import textwrap

# 공통 프롬프트 레지스트리
PROMPT_REGISTRY = {
    "guess_sections": {
        "system": textwrap.dedent(
            """
            "당신은 질문에 필요한 차량 정보 섹션을 판단해 주는 어시스턴트입니다."
        """
        ).strip(),
        "user_template": textwrap.dedent(
            """
            아래 질문과 대화 기록을 통해 고객의 니즈를 추론하여 차량의 어떤 정보(섹션)가 필요할지 빠르게 판단.

            가능한 섹션 이름:
            summary, basic_info, special_usage_history, special_accident_history,
            registration_change_history, insurance_accident_history, options,
            seller_info, inspection_record, purpose_metadata

            user_query: {question}
            이전 대화 요약:
            {history_summary}

            출력 형식: 쉼표로 구분된 섹션 이름.
        """
        ),
    },
    "user_info_update": {
        "system": textwrap.dedent(
            """
            당신은 대화에서 customer_info를 갱신하는 정보 추출기입니다.
            customer_info는 아래 3개 필드만 관리합니다:
            - purchase_purpose: 문자열 리스트(예: ["출퇴근","패밀리카","골프"]).
            - preferred_car_type: 선호 차종
            - budget_krw: 만원단위 숫자
            - lead_status: 리드 상태

            규칙:
            - 사용자가 명시한 것만 채워라. 추정 금지.
            - 값이 불명확하면 해당 필드는 변경하지 마라.
            - purchase_purpose는 중복 없이 누적 가능.
            - preferred_car_type은 사용자가 명시한 차종만 저장 (예: "SUV", "세단", "경차" 등)
            - lead_status는 리드 상태만 저장 (예: "Cold Lead", "Warm Lead", "Hot Lead")
            - 결과는 반드시 JSON만 출력하라.
        """
        ).strip(),
        "user_template": textwrap.dedent(
            """
            고객 정보:{customer_info}
            user_query:{question}

            출력(JSON only):
            {{"purchase_purpose": [...], "preferred_car_type": "...", "budget_krw": "...", "lead_status": "..."}}
        """
        ).strip(),
    },
    "extract_search_query_and_sections": {
        "system": textwrap.dedent(
            """
            당신은 질문에서 필요한 차량 정보 섹션과 최적화된 검색 쿼리를 추출하는 전문가입니다.
            
            아래 질문과 대화 기록을 분석하여:
            1. 필요한 차량 정보 섹션을 판단
            2. RAG 검색에 최적화된 검색 쿼리 추출
            
            [가능한 섹션 이름]
            summary, basic_info, special_usage_history, special_accident_history,
            registration_change_history, insurance_accident_history, options,
            seller_info, inspection_record, purpose_metadata
            
            [검색 쿼리 추출 규칙]
            - 불필요한 단어 제거: "보여줘", "알려줘", "차를", "차량을", "를", "을", "이", "가", "의" 등
            - 핵심 정보 강조: 연식, 제조사, 모델명, 트림명, 차체 타입 등
            - 예시:
              * 입력: "2021년식 BMW 4시리즈 쿠페 420i M 스포츠팩 차를 보여줘"
              * 출력: "2021 BMW 4시리즈 쿠페 420i M 스포츠팩"
              * 입력: "가족용으로 좋은 SUV 추천해줘"
              * 출력: "가족용 SUV"
            
            [추가 정보 추출]
            - year: 쿼리에서 연식을 숫자로 추출 (예: 2021). 없으면 null
            - manufacturer: 쿼리에서 제조사 추출 (예: "BMW", "Mercedes-Benz", "Audi" 등). 없으면 null
            - model_keywords: 쿼리에서 모델명 관련 키워드 추출 (예: ["4시리즈", "420i", "쿠페", "M 스포츠팩"]). 없으면 빈 리스트
        """
        ).strip(),
        "user_template": textwrap.dedent(
            """
            [사용자 질문]
            {question}
            
            [이전 대화 요약]
            {history_summary}
        """
        ).strip(),
    },
    # "vector_db_query": {
    #     "system": textwrap.dedent(
    #         """
    #         당신은 차량을 추천해주기 위해 데이터베이스에 쿼리할 적절한 문장을 작성하는 어시스턴트 입니다.
    #         예시) 1억 이하, 흰색, 통풍 시트가 구비되어 있는 무사고 SUV.
    #     """
    #     ).strip(),
    #     "user_template": textwrap.dedent(
    #         """
    #         고객 정보:{customer_info}
    #         user_query:{question}
    #         이전 대화 요약:
    #         {history_summary}
    #         출력: 자연어 문장
    #     """
    #     ).strip(),
    # },
}
