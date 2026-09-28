"""
Streamlit 예측 앱
사용자 입력을 받아 LGBM 모델로 중고차 판매가 예측
"""
import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import re
from pathlib import Path
from datetime import datetime
import sys

# 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
MODEL_DIR = BASE_DIR / "00_data" / "03_result" / "lgbm_models" / "saved_models"
PREPARE_MODULE = BASE_DIR / "02_model" / "00_prepare_for_LGBM.py"

# 파일 존재 확인
if not PREPARE_MODULE.exists():
    raise FileNotFoundError(
        f"전처리 모듈을 찾을 수 없습니다: {PREPARE_MODULE}\n"
        f"BASE_DIR: {BASE_DIR}\n"
        f"APP_DIR: {APP_DIR}"
    )

# 전처리 모듈 import
sys.path.append(str(BASE_DIR / "02_model"))
import importlib.util
spec = importlib.util.spec_from_file_location("prepare_for_LGBM", PREPARE_MODULE)
prepare_for_LGBM = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_for_LGBM)

# 전처리 함수들
drop_columns_for_lgbm = prepare_for_LGBM.drop_columns_for_lgbm
categorize_displacement = prepare_for_LGBM.categorize_displacement
convert_binned_features_to_ordinal = prepare_for_LGBM.convert_binned_features_to_ordinal
convert_to_categorical = prepare_for_LGBM.convert_to_categorical
remove_nan_rows_in_binned_features = prepare_for_LGBM.remove_nan_rows_in_binned_features

# 전처리 유틸리티 import
from preprocess_utils import (
    load_mapping_dicts,
    parse_연월식,
    calculate_car_age,
    calculate_temporal_features,
    create_price_interval,
    create_displacement_interval,
    map_grade_code_info,
    get_cluster_from_grade_displacement_price,
    get_name_from_code,
    calculate_km_per_month
)


@st.cache_data
def load_hierarchy_dicts():
    """계층화된 딕셔너리 로드"""
    hierarchy_path = DATA_DIR / "hierarchy_dicts.json"
    if hierarchy_path.exists():
        with open(hierarchy_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


@st.cache_data
def load_name_mapping():
    """이름 매핑 딕셔너리 로드"""
    name_mapping_path = DATA_DIR / "name_mapping.json"
    if name_mapping_path.exists():
        with open(name_mapping_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


@st.cache_resource
def load_lgbm_models():
    """LGBM 모델들 로드"""
    models = {}
    for cluster_num in range(6):
        model_path = MODEL_DIR / f"lgbm_model_cluster_{cluster_num}.txt"
        if model_path.exists():
            try:
                model = lgb.Booster(model_file=str(model_path))
                models[cluster_num] = model
            except Exception as e:
                st.error(f"Cluster {cluster_num} 모델 로드 실패: {str(e)}")
    return models


def format_code_name(code_type: str, code: int, name_mapping: dict) -> str:
    """코드와 이름을 함께 표시하는 형식으로 변환"""
    name = get_name_from_code(code_type, code, {'name_mapping': name_mapping})
    if name:
        return f"{name}({code})"
    return str(code)


def extract_code_from_formatted_string(formatted_str: str) -> int:
    """
    "이름(코드)" 형식의 문자열에서 코드를 안전하게 추출
    이름에 괄호가 포함되어 있어도 마지막 괄호 쌍에서 코드를 추출합니다.
    """
    # 마지막 괄호 쌍에서 숫자를 찾음
    match = re.search(r'\((\d+)\)\s*$', formatted_str)
    if match:
        return int(match.group(1))
    # 괄호가 없으면 전체 문자열이 코드인 경우
    try:
        return int(formatted_str)
    except ValueError:
        raise ValueError(f"코드를 추출할 수 없습니다: {formatted_str}")


def main():
    st.set_page_config(
        page_title="중고차 판매가 예측",
        page_icon="🚗",
        layout="wide"
    )
    
    st.title("🚗 중고차 판매가 예측 시스템")
    st.markdown("---")
    
    # 딕셔너리 로드
    hierarchy_dicts = load_hierarchy_dicts()
    name_mapping = load_name_mapping()
    mapping_dicts = load_mapping_dicts()
    
    if not hierarchy_dicts or not name_mapping:
        st.error("딕셔너리 파일을 찾을 수 없습니다. create_hierarchy_dicts.py를 실행해주세요.")
        return
    
    # 사이드바 - 계층화된 입력 폼
    with st.sidebar:
        st.header("차량 정보 입력")
        
        # 1. 국산외제차 선택
        국산외제차_options = {"국산차": 0, "외제차": 1}
        국산외제차_label = st.radio("국산/외제차", options=list(국산외제차_options.keys()))
        국산외제차 = 국산외제차_options[국산외제차_label]
        
        # 2. 제조사 선택
        국산외제차_dict = hierarchy_dicts.get("국산외제차_dict", {})
        제조사코드_list = 국산외제차_dict.get(str(국산외제차), [])
        
        if not 제조사코드_list:
            st.warning("선택한 국산/외제차에 해당하는 제조사가 없습니다.")
            return
        
        제조사_이름_dict = name_mapping.get("제조사_이름_dict", {})
        제조사_options = [
            format_code_name("제조사", int(code), name_mapping)
            for code in 제조사코드_list
        ]
        
        제조사_selected = st.selectbox("제조사", options=제조사_options)
        제조사코드 = extract_code_from_formatted_string(제조사_selected)
        
        # 3. 대표차종명 선택
        제조사_dict = hierarchy_dicts.get("제조사_dict", {})
        대표차종명_list = 제조사_dict.get(str(제조사코드), [])
        
        if not 대표차종명_list:
            st.warning("선택한 제조사에 해당하는 대표차종명이 없습니다.")
            return
        
        대표차종명 = st.selectbox("대표차종명", options=대표차종명_list)
        
        # 4. 모델 선택
        대표차종명_dict = hierarchy_dicts.get("대표차종명_dict", {})
        key = (제조사코드, 대표차종명)
        모델코드_list = 대표차종명_dict.get(str(key), [])
        
        if not 모델코드_list:
            st.warning("선택한 대표차종명에 해당하는 모델이 없습니다.")
            return
        
        모델_이름_dict = name_mapping.get("모델_이름_dict", {})
        모델_options = [
            format_code_name("모델", int(code), name_mapping)
            for code in 모델코드_list
        ]
        
        모델_selected = st.selectbox("모델", options=모델_options)
        모델코드 = extract_code_from_formatted_string(모델_selected)
        
        # 5. 등급 선택
        모델_dict = hierarchy_dicts.get("모델_dict", {})
        등급코드_list = 모델_dict.get(str(모델코드), [])
        
        if not 등급코드_list:
            st.warning("선택한 모델에 해당하는 등급이 없습니다.")
            return
        
        등급_이름_dict = name_mapping.get("등급_이름_dict", {})
        등급_options = [
            format_code_name("등급", int(code), name_mapping)
            for code in 등급코드_list
        ]
        
        등급_selected = st.selectbox("등급", options=등급_options)
        등급코드 = extract_code_from_formatted_string(등급_selected)
        
        # 6. 세부등급 선택
        등급_dict = hierarchy_dicts.get("등급_dict", {})
        세부등급코드_list = 등급_dict.get(str(등급코드), [])
        
        세부등급_이름_dict = name_mapping.get("세부등급_이름_dict", {})
        세부등급_options = ["세부등급 없음"]
        if 세부등급코드_list:
            세부등급_options = [
                format_code_name("세부등급", int(code), name_mapping)
                if code is not None else "세부등급 없음"
                for code in 세부등급코드_list
            ]
        
        세부등급_selected = st.selectbox("세부등급", options=세부등급_options)
        if "(" in 세부등급_selected and ")" in 세부등급_selected:
            try:
                세부등급코드 = extract_code_from_formatted_string(세부등급_selected)
            except ValueError:
                세부등급코드 = None
        else:
            세부등급코드 = None
    
    # 메인 영역 - 추가 입력 필드
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("차량 상세 정보")
        신차가격 = st.number_input("신차가격 (만원)", min_value=0, value=3000, step=100)
        연월식 = st.text_input("연월식", value="2020-01", help="YYYY-MM 형식으로 입력하세요")
        주행거리 = st.number_input("주행거리 (km)", min_value=0, value=50000, step=1000)
        배기량 = st.number_input("배기량 (cc)", min_value=0, value=2000, step=100)
        
        색상_list = hierarchy_dicts.get("색상_list", [])
        색상 = st.selectbox("색상", options=색상_list)
        
        용도변경이력 = st.radio("용도변경이력", options=["N", "Y"], index=0)
    
    with col2:
        st.subheader("자동 계산 필드")
        
        # 배기량_구간 계산
        배기량_구간 = create_displacement_interval(배기량)
        st.info(f"**배기량 구간**: {배기량_구간}")
        
        # 신차가격_구간 계산 (만원 단위로 입력받았으므로 is_만원_unit=True)
        신차가격_구간 = create_price_interval(신차가격, is_만원_unit=True)
        st.info(f"**신차가격 구간**: {신차가격_구간}")
        
        # 차나이 계산
        current_datetime = datetime.now()
        차나이_월, 차나이_년 = calculate_car_age(연월식, current_datetime)
        if 차나이_월 is not None:
            st.info(f"**차나이**: {차나이_년}년 {차나이_월 % 12}개월 (총 {차나이_월}개월)")
        
        # 시간 피처 계산
        temporal_features = calculate_temporal_features(current_datetime)
        st.info(f"**월 피처**: sin={temporal_features['month_sin']}, cos={temporal_features['month_cos']}")
        st.info(f"**주 피처**: sin={temporal_features['week_sin']}, cos={temporal_features['week_cos']}")
        
        # 등급코드 기반 정보 매핑
        grade_info = map_grade_code_info(등급코드, mapping_dicts)
        st.info(f"**차종**: {grade_info.get('차종', 'N/A')}")
        st.info(f"**연료**: {grade_info.get('연료', 'N/A')}")
        
        # 클러스터 계산
        cluster = get_cluster_from_grade_displacement_price(
            등급코드, 배기량_구간, 신차가격_구간, mapping_dicts
        )
        if cluster is not None:
            st.success(f"**예상 클러스터**: {cluster}")
    
    # 예측 실행 버튼
    st.markdown("---")
    if st.button("🔮 예측하기", type="primary", use_container_width=True):
        try:
            # 입력 데이터프레임 생성
            input_data = {
                '제조사': get_name_from_code("제조사", 제조사코드, {'name_mapping': name_mapping}),
                '대표차종명': 대표차종명,
                '모델': get_name_from_code("모델", 모델코드, {'name_mapping': name_mapping}),
                '등급': get_name_from_code("등급", 등급코드, {'name_mapping': name_mapping}),
                '세부등급': get_name_from_code("세부등급", 세부등급코드, {'name_mapping': name_mapping}) if 세부등급코드 else "세부등급 없음",
                '제조사코드': 제조사코드,
                '모델코드': 모델코드,
                '등급코드': 등급코드,
                '세부등급코드': 세부등급코드 if 세부등급코드 else None,
                '색상': 색상,
                '연월식': 연월식,
                '신차가격(선택옵션포함)': 신차가격 * 10000,  # 만원 -> 원 변환
                '배기량': 배기량,
                '차종': grade_info.get('차종'),
                '연료': grade_info.get('연료'),
                '주행거리': 주행거리,
                '용도변경이력': 용도변경이력,
                '사고유무': 'N',
                '단순수리': 'N',
                '사고_심각도': 0,
                '국산외제차': 국산외제차,
                '판매기준금리': 2.5,
                'gen_norm': grade_info.get('gen_norm'),
                '판매속도지수_등급코드': grade_info.get('판매속도지수_등급코드'),
                '빈도_등급코드': grade_info.get('빈도_등급코드'),
                '판매속도지수_등급코드_구간': grade_info.get('판매속도지수_등급코드_구간'),
                '빈도_등급코드_구간': grade_info.get('빈도_등급코드_구간'),
                'is_ev_h2': grade_info.get('is_ev_h2'),
                '차나이_월': 차나이_월,
                '차나이_년': 차나이_년,
            }
            
            # km_per_month 계산
            if 차나이_월 and 차나이_월 > 0:
                input_data['km_per_month'] = calculate_km_per_month(주행거리, 차나이_월)
            
            # 시간 피처 추가
            input_data.update(temporal_features)
            
            df_input = pd.DataFrame([input_data])
            
            # 전처리
            # 1. 배기량_구간 생성
            df_input['배기량_구간'] = df_input['배기량'].apply(create_displacement_interval)
            df_input = categorize_displacement(df_input)
            
            # 2. 신차가격_구간 생성 (원 단위를 만원 단위로 변환한 후 전달)
            df_input['신차가격_구간'] = df_input['신차가격(선택옵션포함)'].apply(lambda x: create_price_interval(x / 10000, is_만원_unit=True))
            
            # 3. 구간을 순서형 카테고리로 변환
            df_input = convert_binned_features_to_ordinal(df_input)
            
            # 4. 카테고리 변환
            df_input = convert_to_categorical(df_input, verbose=False)
            
            # 5. 불필요한 컬럼 제거
            df_input = drop_columns_for_lgbm(df_input)
            
            # 6. NaN 행 제거
            df_input_before_remove = df_input.copy()
            df_input = remove_nan_rows_in_binned_features(df_input, verbose=False)
            
            if len(df_input) == 0:
                st.error("전처리 후 데이터가 없습니다. 입력값을 확인해주세요.")
                # 디버깅 정보 표시
                with st.expander("🔍 디버깅 정보"):
                    st.write("**NaN 행 제거 전 데이터프레임:**")
                    st.dataframe(df_input_before_remove)
                    st.write("**NaN이 있는 컬럼:**")
                    nan_cols = df_input_before_remove.columns[df_input_before_remove.isna().any()].tolist()
                    st.write(nan_cols)
                    if nan_cols:
                        st.write("**NaN 값 상세:**")
                        for col in nan_cols:
                            st.write(f"- {col}: {df_input_before_remove[col].isna().sum()}개")
                return
            
            # 모델 로드 및 예측
            models = load_lgbm_models()
            if cluster not in models:
                st.error(f"클러스터 {cluster}에 해당하는 모델을 찾을 수 없습니다.")
                return
            
            model = models[cluster]
            
            # 예측 전 데이터 확인 (디버깅)
            with st.expander("🔍 예측 전 데이터 확인", expanded=False):
                st.write(f"**데이터프레임 shape**: {df_input.shape}")
                st.write(f"**컬럼 수**: {len(df_input.columns)}")
                st.write("**컬럼 목록:**")
                st.write(list(df_input.columns))
                st.write("**예측에 사용할 데이터 (첫 번째 행):**")
                st.dataframe(df_input.iloc[[0]])
            
            # 예측 (모델은 감가율을 예측함)
            prediction = model.predict(df_input.iloc[[0]], num_iteration=model.best_iteration)
            예측_감가율 = prediction[0]
            
            # 감가율을 판매가로 변환: 판매가 = 신차가격 * (1 - 감가율)
            신차가격_원 = input_data['신차가격(선택옵션포함)']
            예측가격 = 신차가격_원 * (1 - 예측_감가율)
            
            # 예측값이 비정상적인 경우 경고
            if 예측가격 <= 0 or 예측_감가율 < 0 or 예측_감가율 > 1:
                st.warning(f"⚠️ 예측값이 비정상적입니다!")
                with st.expander("🔍 예측 디버깅 정보"):
                    st.write(f"**예측 감가율**: {예측_감가율:.4f}")
                    st.write(f"**예측 판매가**: {예측가격:,.0f}원")
                    st.write(f"**신차가격**: {신차가격_원:,.0f}원")
                    st.write(f"**모델 best_iteration**: {model.best_iteration}")
                    st.write(f"**클러스터**: {cluster}")
                    st.write("**입력 데이터:**")
                    st.dataframe(df_input.iloc[[0]])
            
            # 결과 표시
            st.success("✅ 예측 완료!")
            st.markdown("---")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("예측 판매가", f"{예측가격:,.0f}원", f"{예측가격/10000:,.0f}만원")
            with col2:
                st.metric("신차가격", f"{input_data['신차가격(선택옵션포함)']:,.0f}원", f"{신차가격:,.0f}만원")
            with col3:
                감가율_퍼센트 = 예측_감가율 * 100
                st.metric("예측 감가율", f"{감가율_퍼센트:.1f}%")
            
            # 상세 정보 표시
            with st.expander("📋 입력 정보 확인"):
                st.json(input_data)
            
        except Exception as e:
            st.error(f"예측 중 오류가 발생했습니다: {str(e)}")
            st.exception(e)


if __name__ == "__main__":
    main()
