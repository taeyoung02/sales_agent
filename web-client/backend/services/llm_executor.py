# llm_evaluator 에이전트가 선택한 결과를 실행하는 에이전트
# 실행 결과는 3D 모델 조작, 카메라 조작, 3D 파일 경로 등 다양한 functions call 실행하고 결과를 반환

import logging
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from .base_agent import BaseAgent
from utils.logging_context import set_session_id

logger = logging.getLogger(__name__)


class LLMExecutor(BaseAgent):
    """llm_evaluator 에이전트가 선택한 결과를 실행하는 에이전트"""

    def __init__(self, api_key: Optional[str] = None) -> None:
        super().__init__(api_key)

    def execute(
        self,
        selected_plan: Dict[str, Any],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        vehicle_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        선택된 시나리오 실행

        Returns:
            {
                "response": str,  # 최종 텍스트 응답
                "tool_calls": List[Dict],  # 실행할 tool calls
                "evidence": List[Dict],  # RAG 검색 결과 (있는 경우)
            }
        """
        # ContextVar는 스레드 간 전달되지 않으므로, asyncio.to_thread()로 실행될 때를 대비해 명시적으로 설정
        set_session_id(session_id or "default")

        # Step 1: Tool calls 검증 및 정제
        validated_tool_calls = self._validate_tool_calls(
            selected_plan.get("tool_calls", []),
            vehicle_id,
            conversation_history,
        )

        # Step 2: Tool calls 실행
        # 성능 최적화: searchVehicleDatabase tool calls를 병렬 실행
        executed_tool_calls = []
        search_results = []

        # searchVehicleDatabase와 다른 tool들을 분리
        search_vehicle_calls = [
            tc
            for tc in validated_tool_calls
            if tc.get("name") == "searchVehicleDatabase"
        ]
        other_tool_calls = [
            tc
            for tc in validated_tool_calls
            if tc.get("name") != "searchVehicleDatabase"
        ]

        # searchVehicleDatabase 병렬 실행 및 VIN 추출
        extracted_vins = []  # 여러 VIN 추출 (비교 모드 대응)
        query_vin_map = {}  # 쿼리와 VIN 매핑 (우선순위 결정용)
        search_results_with_metadata = []  # (text, metadata) 튜플 리스트 (evidence용)
        if search_vehicle_calls:
            with ThreadPoolExecutor(max_workers=len(search_vehicle_calls)) as executor:
                futures = {
                    executor.submit(
                        self._execute_search_vehicle_database,
                        tc.get("arguments", {}).get("query", ""),
                        vehicle_id,
                        session_id,
                    ): tc
                    for tc in search_vehicle_calls
                }

                for future in as_completed(futures):
                    tool_call = futures[future]
                    query = tool_call.get("arguments", {}).get("query", "")
                    try:
                        result = future.result()
                        if result:
                            extracted_vin = None
                            metadata = None

                            # result는 (search_result_text, metadata) 튜플 또는 search_result_text 문자열
                            if isinstance(result, tuple) and len(result) == 2:
                                search_result_text, metadata = result
                                search_results.append(search_result_text)
                                search_results_with_metadata.append(
                                    (search_result_text, metadata)
                                )

                                # metadata에서 직접 VIN 추출
                                if metadata and metadata.get("vin"):
                                    extracted_vin = metadata["vin"]
                            else:
                                # 하위 호환성: 문자열만 반환된 경우
                                search_result_text = result
                                search_results.append(search_result_text)
                                search_results_with_metadata.append(
                                    (search_result_text, None)
                                )

                                # 텍스트에서 VIN 추출 시도 (fallback)
                                extracted_vin = self._extract_vin_from_text(
                                    search_result_text
                                )

                            # VIN 추출 성공 시 쿼리와 매핑 저장
                            if extracted_vin:
                                query_vin_map[query] = extracted_vin
                                if extracted_vin not in extracted_vins:
                                    extracted_vins.append(extracted_vin)
                                    logger.info(
                                        f"[Executor] metadata에서 VIN 추출: {extracted_vin} (총 {len(extracted_vins)}개), query={query}"
                                    )

                            logger.info(
                                f"[Executor] searchVehicleDatabase 실행 완료: query={query}, 결과 길이: {len(search_result_text) if isinstance(result, tuple) else len(result)}, VIN={extracted_vin or '없음'}"
                            )
                        else:
                            logger.warning(
                                f"[Executor]searchVehicleDatabase 실행 실패: query={query}, 결과 없음"
                            )
                    except Exception as e:
                        logger.error(
                            f"[Executor] searchVehicleDatabase 실행 오류: query={query}, error={e}",
                            exc_info=True,
                        )

        # searchVehicleDatabase에서 추출한 VIN을 loadVehicle / generatePresentationScript에 자동 연결
        # 우선순위: 구체적인 차량명 쿼리 > VIN 직접 검색 > 기타
        if extracted_vins:
            # 1. "test", "placeholder", VIN 문자열 등을 제외한 구체적인 차량명 쿼리 찾기
            specific_queries = [
                q
                for q in query_vin_map.keys()
                if q
                and q not in ["test", "placeholder", "VIN_PLACEHOLDER"]
                and not self._is_valid_vin_or_car_id(
                    q
                )  # VIN이나 car_id가 아닌 구체적인 설명
            ]

            if specific_queries:
                # 구체적인 쿼리가 있으면 해당 VIN을 primary로 사용
                primary_vin = query_vin_map[specific_queries[0]]
                print(
                    f"[Executor] 구체적인 쿼리에서 primary_vin 결정: query='{specific_queries[0]}' → VIN={primary_vin}"
                )
                # 나머지 VIN들을 secondary로 사용 (중복 제거)
                remaining_vins = [vin for vin in extracted_vins if vin != primary_vin]
                secondary_vin = remaining_vins[0] if remaining_vins else None
            else:
                # 구체적인 쿼리가 없으면 첫 번째 VIN 사용 (기존 로직)
                primary_vin = extracted_vins[0]
                secondary_vin = extracted_vins[1] if len(extracted_vins) >= 2 else None
                print(
                    f"[Executor] 첫 번째 결과에서 primary_vin 결정: VIN={primary_vin}"
                )

            # 기존 tool_calls에서 VIN 자동 연결 및 정합성 보정
            for tool_call in other_tool_calls:
                name = tool_call.get("name")
                args = tool_call.get("arguments", {})

                # loadVehicle에 VIN 자동 연결 및 정합성 보정
                if name == "loadVehicle":
                    current_vid = args.get("vehicle_id")
                    current_compare_vid = args.get("compare_with_vehicle_id")

                    # 1) vehicle_id가 없거나 유효하지 않은 경우 → primary_vin 사용
                    # 2) vehicle_id가 존재하지만, search 결과에서 추출한 VIN 리스트와 불일치하는 경우 → primary_vin으로 덮어쓰기
                    if (
                        not current_vid
                        or current_vid == "test"
                        or (
                            current_vid
                            and not self._is_valid_vin_or_car_id(current_vid)
                        )
                        or (current_vid and current_vid not in extracted_vins)
                    ):
                        old_vid = current_vid
                        print(
                            f"[Executor] loadVehicle vehicle_id 보정: {old_vid} → {primary_vin}"
                        )
                        args["vehicle_id"] = primary_vin
                        # source_path는 항상 vehicle_id(VIN)로부터 생성
                        args["source_path"] = self._infer_source_path(primary_vin)

                    # compare_with_vehicle_id가 없거나 유효하지 않은 경우 또는 추출 VIN과 불일치하는 경우 두 번째 VIN 자동 설정
                    if secondary_vin:
                        if (
                            not current_compare_vid
                            or current_compare_vid == "test"
                            or (
                                current_compare_vid
                                and not self._is_valid_vin_or_car_id(
                                    current_compare_vid
                                )
                            )
                            or (
                                current_compare_vid
                                and current_compare_vid not in extracted_vins
                            )
                        ):
                            old_compare = current_compare_vid
                            print(
                                f"[Executor] loadVehicle compare_with_vehicle_id 보정: {old_compare} → {secondary_vin}"
                            )
                            args["compare_with_vehicle_id"] = secondary_vin

                # generatePresentationScript의 vehicle_id도 search 결과 VIN과 맞도록 보정
                # 주의: script 파라미터는 보존해야 함 (수정하지 않음)
                elif name == "generatePresentationScript":
                    gps_vid = args.get("vehicle_id")
                    if not gps_vid or gps_vid not in extracted_vins:
                        old_gps_vid = gps_vid
                        args["vehicle_id"] = primary_vin
                        print(
                            f"[Executor] generatePresentationScript vehicle_id 보정: {old_gps_vid} → {primary_vin}"
                        )
                    # script 파라미터가 있는지 확인 (디버깅용)
                    if "script" not in args:
                        print(
                            f"[Executor] ⚠️  WARNING: generatePresentationScript에 script 파라미터가 없습니다!"
                        )
                    else:
                        # ===== 🔍 Script Dictionary 상세 로깅 (전체 출력, 스킵 없음) =====
                        script_dict = args.get('script', {})
                        segments = script_dict.get('segments', [])
                        logger.info("=" * 80)
                        logger.info(f"[Executor] 📤 generatePresentationScript Dictionary (TTS 전송 직전):")
                        logger.info(f"  - vehicle_id: {args.get('vehicle_id')}")
                        logger.info(f"  - total_duration: {script_dict.get('total_duration')}")
                        logger.info(f"  - segments count: {len(segments)}")
                        logger.info("")
                        
                        # 모든 세그먼트 출력 (스킵 없음)
                        for idx, seg in enumerate(segments):
                            logger.info(f"  [Segment {idx}]")
                            logger.info(f"    - timestamp: {seg.get('timestamp')}")
                            logger.info(f"    - text: {seg.get('text', '')}")  # 전체 텍스트
                            logger.info(f"    - actions: {seg.get('actions')}")  # 전체 actions
                            logger.info("")  # 세그먼트 간 구분
                        
                        logger.info("=" * 80)
                        # ===== End of Logging =====
                        
                        print(
                            f"[Executor] generatePresentationScript script 파라미터 확인: segments={len(segments)}개"
                        )

        # 다른 tool들은 프론트엔드로 전달
        executed_tool_calls = other_tool_calls

        # Step 3: 텍스트 응답 가공
        # Plan 구조에서는 dialogue 필드 사용
        dialogue = selected_plan.get("dialogue", "")
        final_dialogue = self._polish_dialogue(
            dialogue,
            executed_tool_calls,
            conversation_history,
        )

        # Step 4: Evidence 구성 (RAG 검색 결과 + searchVehicleDatabase 결과)
        evidence = []
        if selected_plan.get("rag_context"):
            evidence.extend(self._format_evidence(selected_plan["rag_context"]))

        # searchVehicleDatabase 실행 결과를 evidence에 추가 (source_id, snippet_text 형식)
        for result_data in search_results_with_metadata:
            if isinstance(result_data, tuple) and len(result_data) == 2:
                search_result_text, metadata = result_data
            else:
                # 하위 호환성: 문자열만 있는 경우
                search_result_text = result_data
                metadata = None

            # source_id 추출: metadata에서 VIN 또는 car_id 우선 사용
            source_id = ""
            if metadata:
                source_id = (
                    metadata.get("vin")
                    or metadata.get("car_id")
                    or metadata.get("id")
                    or ""
                )
            # metadata가 없으면 텍스트에서 VIN 추출 시도
            if not source_id:
                extracted_vin = self._extract_vin_from_text(search_result_text)
                if extracted_vin:
                    source_id = extracted_vin

            # snippet_text: 원문 근거 스니펫 (전체 텍스트 또는 일부)
            snippet_text = (
                search_result_text[:500]
                if len(search_result_text) > 500
                else search_result_text
            )

            evidence.append(
                {
                    "source_id": source_id,
                    "snippet_text": snippet_text,
                }
            )

        logger.info(f"[Executor] dialogue: {dialogue}")
        logger.info(f"[Executor] tool_calls: {len(executed_tool_calls)}개")
        logger.info(f"[Executor] final_dialogue 길이={len(final_dialogue)}")

        return {
            "response": final_dialogue,
            "tool_calls": executed_tool_calls,
            "evidence": evidence,
        }

    def _validate_tool_calls(
        self,
        tool_calls: List[Dict],
        vehicle_id: Optional[str],
        conversation_history: Optional[List[Dict[str, str]]],
    ) -> List[Dict]:
        """Tool calls 검증 및 필수 파라미터 보완"""
        validated = []

        # vehicle_id 우선순위: 명시적 전달 > 대화 히스토리에서 추출
        current_vehicle_id = vehicle_id
        if not current_vehicle_id and conversation_history:
            current_vehicle_id = self.extract_vehicle_id_from_history(
                conversation_history
            )

        for tool_call in tool_calls:
            # tool_call이 딕셔너리인지 확인
            if not isinstance(tool_call, dict):
                # Pydantic 모델 객체인 경우 딕셔너리로 변환
                if hasattr(tool_call, "model_dump"):
                    tool_call = tool_call.model_dump(mode="python")
                else:
                    continue  # 변환 불가능한 경우 스킵

            name = tool_call.get("name")

            # 디버깅: 원본 tool_call 확인 (JSON 파싱 전)
            if name == "generatePresentationScript":
                original_args = tool_call.get("arguments", {})
                print(
                    f"[Executor] [DEBUG] generatePresentationScript 원본 arguments (파싱 전): type={type(original_args)}, "
                    f"keys={list(original_args.keys()) if isinstance(original_args, dict) else 'N/A'}, "
                    f"is_str={isinstance(original_args, str)}, str_len={len(str(original_args)) if isinstance(original_args, str) else 0}"
                )
                if isinstance(original_args, str):
                    print(
                        f"[Executor] [DEBUG] 원본 arguments 문자열 (처음 500자): {original_args[:500]}"
                    )

            args = tool_call.get("arguments", {})

            # arguments가 문자열인 경우 Dict로 변환
            if isinstance(args, str):
                import json

                try:
                    parsed_args = json.loads(args)
                    print(
                        f"[Executor] [DEBUG] JSON 파싱 성공: {name}, 파싱된 키: {list(parsed_args.keys()) if isinstance(parsed_args, dict) else 'N/A'}"
                    )
                    args = parsed_args
                except (json.JSONDecodeError, TypeError) as e:
                    print(
                        f"[Executor] ⚠️  JSON 파싱 실패: {name}, error={e}, args={args[:200] if len(str(args)) > 200 else args}"
                    )
                    args = {}
            elif not isinstance(args, dict):
                print(
                    f"[Executor] ⚠️  arguments가 딕셔너리가 아님: {name}, type={type(args)}"
                )
                args = {}

            # 디버깅: generatePresentationScript의 파싱된 arguments 확인
            if name == "generatePresentationScript":
                print(
                    f"[Executor] [DEBUG] generatePresentationScript 파싱된 arguments 키: {list(args.keys()) if isinstance(args, dict) else 'N/A'}"
                )
                if "script" in args:
                    script_segments = args.get("script", {}).get("segments", [])
                    print(
                        f"[Executor] [DEBUG] script segments 개수: {len(script_segments)}"
                    )
                else:
                    print(
                        f"[Executor] [DEBUG] ⚠️  script 파라미터가 파싱된 args에 없습니다. args 전체: {args}"
                    )

            # 공통: vehicle_id 정규화 (car_id -> VIN, test 예외)
            def _canonical_vehicle_id(raw_vehicle_id: Optional[str]) -> Optional[str]:
                if not raw_vehicle_id:
                    return None
                return self._normalize_vehicle_id_to_vin(raw_vehicle_id)

            # loadVehicle 검증
            if name == "loadVehicle":
                if not args.get("vehicle_id") and current_vehicle_id:
                    args["vehicle_id"] = current_vehicle_id

                canonical_vid = _canonical_vehicle_id(args.get("vehicle_id"))
                if canonical_vid:
                    args["vehicle_id"] = canonical_vid
                    # source_path는 항상 vehicle_id(VIN)로부터 생성 (LLM 제공값은 무시/덮어쓰기)
                    args["source_path"] = self._infer_source_path(canonical_vid)

                # compare_with_vehicle_id도 VIN으로 정규화 및 source_path 생성
                if args.get("compare_with_vehicle_id"):
                    original_compare_vid = args.get("compare_with_vehicle_id")
                    canonical_compare_vid = _canonical_vehicle_id(original_compare_vid)
                    if canonical_compare_vid:
                        args["compare_with_vehicle_id"] = canonical_compare_vid
                        # compare_with_vehicle_id의 source_path도 생성 (일관성 유지)
                        args["compare_with_source_path"] = self._infer_source_path(
                            canonical_compare_vid
                        )
                        if original_compare_vid != canonical_compare_vid:
                            print(
                                f"[Executor] compare_with_vehicle_id 정규화: {original_compare_vid} → {canonical_compare_vid}"
                            )

            # generatePresentationScript 검증
            elif name == "generatePresentationScript":
                if not args.get("vehicle_id") and current_vehicle_id:
                    args["vehicle_id"] = current_vehicle_id
                canonical_vid = _canonical_vehicle_id(args.get("vehicle_id"))
                if canonical_vid:
                    args["vehicle_id"] = canonical_vid
                # script 파라미터는 보존해야 함 (수정하지 않음)
                # script가 없는 경우 경고 (디버깅용)
                if "script" not in args:
                    print(
                        f"[Executor] ⚠️  WARNING: _validate_tool_calls에서 generatePresentationScript의 script 파라미터가 없습니다!"
                    )
                    print(
                        f"[Executor] Original args keys: {list(tool_call.get('arguments', {}).keys()) if isinstance(tool_call.get('arguments'), dict) else 'N/A'}"
                    )
                else:
                    segments_count = len(args.get("script", {}).get("segments", []))
                    print(
                        f"[Executor] _validate_tool_calls: generatePresentationScript script 확인: segments={segments_count}개"
                    )

            validated.append({"name": name, "arguments": args})

        return validated

    def _normalize_vehicle_id_to_vin(self, vehicle_id: str) -> str:
        """
        vehicle_id를 VIN으로 정규화합니다.
        - VIN이면 그대로
        - car_id이면 VINMapper로 변환 시도
        - test는 그대로
        """
        if not vehicle_id:
            return vehicle_id
        if vehicle_id == "test":
            return vehicle_id

        from rag.vin_mapper import get_vin_mapper

        vin_mapper = get_vin_mapper()

        if vin_mapper.is_vin(vehicle_id):
            return vehicle_id

        vin = vin_mapper.get_vin_from_car_id(vehicle_id)
        if vin:
            logger.info(f"[Executor] car_id → VIN 변환: {vehicle_id} → {vin}")
            return vin

        # 변환 실패 시 원본 유지 (단, 이 경우 source_path 생성은 잘못될 수 있으므로 로그로 남김)
        logger.warning(f"[Executor] car_id → VIN 변환 실패: {vehicle_id} (원본 유지)")
        return vehicle_id

    def _infer_source_path(self, vehicle_id: Optional[str]) -> str:
        """vehicle_id(VIN)로부터 canonical source_path 생성"""
        if not vehicle_id or vehicle_id == "test":
            return "/static/source/test/point_cloud.ply"
        return f"/static/source/{vehicle_id}/point_cloud.ply"

    def _polish_dialogue(
        self,
        dialogue: str,
        tool_calls: List[Dict],
        conversation_history: Optional[List[Dict[str, str]]],
    ) -> str:
        """dialogue를 Sales Person 형태로 가공"""
        # loadVehicle이 있으면 vehicle_id 정보 추가
        load_vehicle = next(
            (tc for tc in tool_calls if tc.get("name") == "loadVehicle"), None
        )
        if load_vehicle:
            vid = load_vehicle.get("arguments", {}).get("vehicle_id")
            if vid and f"(현재 로드된 차량 ID: {vid})" not in dialogue:
                dialogue += f"\n\n(현재 로드된 차량 ID: {vid})"

        return dialogue

    def _execute_search_vehicle_database(
        self, query: str, vehicle_id: Optional[str], session_id: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        searchVehicleDatabase tool 실행
        구조화된 결과와 함께 metadata(VIN 포함)를 튜플로 반환

        Returns:
            (search_result_text, metadata) 튜플
            - search_result_text: 검색 결과 텍스트
            - metadata: VIN, car_id 등이 포함된 메타데이터 딕셔너리
        """
        # ContextVar는 스레드 간 전달되지 않으므로 명시적으로 설정
        set_session_id(session_id or "default")

        try:
            # vehicle_id와 query가 일치하면 get_vehicle_full_info 사용
            if vehicle_id and query.strip() == vehicle_id:
                logger.info(f"[Executor] vehicle_id로 검색: {vehicle_id}")
                vehicle_info = self.rag_pipeline.get_vehicle_full_info(vehicle_id)
                if vehicle_info and not vehicle_info.startswith("No information found"):
                    # VIN은 vehicle_id 자체일 수 있음
                    metadata = {"vin": vehicle_id} if len(vehicle_id) == 17 else {}
                    return (vehicle_info, metadata)

            # 일반 검색 - search 메서드를 사용하여 metadata(VIN 포함) 접근
            logger.info(f"[Executor] 일반 검색 실행: {query}")
            search_results = self.rag_pipeline.search(query, n_results=3)

            if search_results and len(search_results) > 0:
                # 첫 번째 결과의 document와 metadata 사용
                first_result = search_results[0]
                document = first_result.get("document", "")
                metadata = first_result.get("metadata", {})

                # VIN 정보를 문서에 추가 (하위 호환성)
                formatted_result = document
                if metadata.get("vin"):
                    formatted_result = f"[VIN: {metadata['vin']}]\n{formatted_result}"
                if metadata.get("car_id"):
                    formatted_result = (
                        f"[car_id: {metadata['car_id']}]\n{formatted_result}"
                    )

                if formatted_result and not formatted_result.startswith(
                    "No relevant information found"
                ):
                    return (formatted_result, metadata)

            # fallback: get_context 사용 (하위 호환성)
            context = self.rag_pipeline.get_context(query, n_results=3)
            if context and not context.startswith("No relevant information found"):
                # get_context는 문자열만 반환하므로 metadata 없음
                return (context, None)

            return (None, None)
        except Exception as e:
            logger.error(
                f"[Executor] searchVehicleDatabase 실행 오류: {e}", exc_info=True
            )
            return (None, None)

    def _extract_vin_from_text(self, text: str) -> Optional[str]:
        """
        텍스트에서 VIN 추출 (fallback용)
        주로 하위 호환성을 위한 메서드로, 가능하면 metadata에서 직접 추출하는 것을 권장
        """
        import re

        try:
            # 텍스트에서 17자리 VIN 패턴 찾기
            vin_pattern = r"\b([A-Z0-9]{17})\b"
            matches = re.findall(vin_pattern, text, re.IGNORECASE)
            if matches:
                vin = matches[0].upper()
                logger.info(f"[Executor] 텍스트에서 VIN 추출: {vin}")
                return vin
            return None
        except Exception as e:
            logger.warning(f"[Executor] 텍스트에서 VIN 추출 오류: {e}")
            return None

    def _is_valid_vin_or_car_id(self, vehicle_id: str) -> bool:
        """
        vehicle_id가 유효한 VIN 또는 car_id인지 확인
        """
        if not vehicle_id or vehicle_id == "test":
            return False

        from rag.vin_mapper import get_vin_mapper

        vin_mapper = get_vin_mapper()
        if vin_mapper.is_vin(vehicle_id):
            return True

        # car_id인지 확인 (VINMapper에 있으면 유효)
        vin = vin_mapper.get_vin_from_car_id(vehicle_id)
        return vin is not None

    def _format_evidence(self, rag_context: str) -> List[Dict]:
        """
        RAG 검색 결과를 evidence 형식으로 변환
        evidence 스키마: {source_id: str, snippet_text: str}
        """
        # rag_context는 문자열이므로 source_id 추출이 어려움
        # 텍스트에서 VIN이나 car_id 패턴을 찾아 source_id로 사용 시도
        source_id = ""
        # VIN 패턴 찾기 (17자리 알파벳+숫자)
        import re

        vin_pattern = r"\b([A-Z0-9]{17})\b"
        vin_matches = re.findall(vin_pattern, rag_context, re.IGNORECASE)
        if vin_matches:
            source_id = vin_matches[0].upper()
        else:
            # car_id 패턴 찾기 (일반적으로 숫자나 특정 형식)
            car_id_pattern = r"\[car_id:\s*([^\]]+)\]"
            car_id_matches = re.findall(car_id_pattern, rag_context)
            if car_id_matches:
                source_id = car_id_matches[0].strip()

        # snippet_text: 원문 근거 스니펫 (전체 또는 일부)
        snippet_text = rag_context[:500] if len(rag_context) > 500 else rag_context

        return [
            {
                "source_id": source_id,
                "snippet_text": snippet_text,
            }
        ]
