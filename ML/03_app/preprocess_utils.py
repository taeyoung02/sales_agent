"""
전처리 유틸리티 함수 모듈
앱에서 사용할 전처리 함수들을 제공합니다.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Tuple
import json
from pathlib import Path


# 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(__file__).resolve().parent / "data"


def load_mapping_dicts():
    """
    매핑 딕셔너리들을 로드하는 함수
    
    Returns:
        dict: 매핑 딕셔너리들
    """
    name_mapping_path = DATA_DIR / "name_mapping.json"
    grade_code_mapping_path = DATA_DIR / "grade_code_mapping.json"
    grade_displacement_price_mapping_path = DATA_DIR / "grade_displacement_price_mapping.json"
    
    mapping_dicts = {}
    
    if name_mapping_path.exists():
        with open(name_mapping_path, 'r', encoding='utf-8') as f:
            mapping_dicts['name_mapping'] = json.load(f)
    
    if grade_code_mapping_path.exists():
        with open(grade_code_mapping_path, 'r', encoding='utf-8') as f:
            mapping_dicts['grade_code_mapping'] = json.load(f)
    
    if grade_displacement_price_mapping_path.exists():
        with open(grade_displacement_price_mapping_path, 'r', encoding='utf-8') as f:
            mapping_dicts['grade_displacement_price_mapping'] = json.load(f)
    
    return mapping_dicts


def parse_연월식(연월식: str) -> Optional[datetime]:
    """
    연월식 문자열을 datetime 객체로 변환
    
    Parameters:
        연월식: str, "YYYY-MM", "YYYYMM", "YYYY/MM" 형식
    
    Returns:
        datetime 객체 또는 None
    """
    if pd.isna(연월식) or not 연월식:
        return None
    
    연월식 = str(연월식).strip()
    
    # 다양한 형식 지원
    formats = [
        "%Y-%m",      # 2023-09
        "%Y%m",       # 202309
        "%Y/%m",      # 2023/09
        "%Y-%m-%d",   # 2023-09-15
        "%Y%m%d",     # 20230915
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(연월식, fmt)
        except ValueError:
            continue
    
    # 마지막 시도: 6자리 숫자만 있는 경우
    if 연월식.isdigit() and len(연월식) == 6:
        try:
            return datetime.strptime(연월식, "%Y%m")
        except ValueError:
            pass
    
    return None


def calculate_car_age(연월식: str, current_datetime: Optional[datetime] = None) -> Tuple[Optional[int], Optional[int]]:
    """
    연월식으로부터 차나이_월, 차나이_년 계산
    
    Parameters:
        연월식: str, "YYYY-MM" 형식의 연월식
        current_datetime: datetime, 현재 시점 (None이면 현재 시간 사용)
    
    Returns:
        tuple: (차나이_월, 차나이_년)
    """
    if current_datetime is None:
        current_datetime = datetime.now()
    
    연월식_dt = parse_연월식(연월식)
    if 연월식_dt is None:
        return None, None
    
    # 차이 계산
    delta = current_datetime - 연월식_dt
    
    # 월 단위 계산 (대략적으로)
    차나이_월 = delta.days // 30
    차나이_년 = delta.days // 365
    
    return 차나이_월, 차나이_년


def calculate_temporal_features(current_datetime: Optional[datetime] = None) -> Dict[str, float]:
    """
    current_datetime으로부터 month_sin, month_cos, week_sin, week_cos 계산
    
    Parameters:
        current_datetime: datetime, 현재 시점 (None이면 현재 시간 사용)
    
    Returns:
        dict: {"month_sin": float, "month_cos": float, "week_sin": float, "week_cos": float}
    """
    if current_datetime is None:
        current_datetime = datetime.now()
    
    # 월 피처
    sale_month = current_datetime.month
    month_sin = round(np.sin(2 * np.pi * sale_month / 12), 2)
    month_cos = round(np.cos(2 * np.pi * sale_month / 12), 2)
    
    # 주 피처 (ISO 주차)
    sale_week = current_datetime.isocalendar().week
    week_sin = round(np.sin(2 * np.pi * sale_week / 52), 2)
    week_cos = round(np.cos(2 * np.pi * sale_week / 52), 2)
    
    return {
        "month_sin": month_sin,
        "month_cos": month_cos,
        "week_sin": week_sin,
        "week_cos": week_cos
    }


def create_price_interval(신차가격: float, is_만원_unit: bool = True) -> Optional[str]:
    """
    신차가격을 구간으로 변환하는 함수
    
    Parameters:
        신차가격: float, 신차가격 값
        is_만원_unit: bool, True면 만원 단위, False면 원 단위 (기본값: True)
    
    Returns:
        str: 신차가격 구간 레이블
    """
    if pd.isna(신차가격) or 신차가격 is None:
        return None
    
    신차가격 = float(신차가격)
    
    # 만원 단위로 변환
    if is_만원_unit:
        # 이미 만원 단위
        price_in_만원 = 신차가격
    else:
        # 원 단위를 만원 단위로 변환
        price_in_만원 = 신차가격 / 10000
    
    if price_in_만원 < 2000:
        return "2천만원 미만"
    elif price_in_만원 < 3000:
        return "2천만원 이상 ~ 3천만원 미만"
    elif price_in_만원 < 4000:
        return "3천만원 이상 ~ 4천만원 미만"
    elif price_in_만원 < 5000:
        return "4천만원 이상 ~ 5천만원 미만"
    elif price_in_만원 < 7000:
        return "5천만원 이상 ~ 7천만원 미만"
    elif price_in_만원 < 10000:
        return "7천만원 이상 ~ 1억원 미만"
    else:
        return "1억원 이상"


def create_displacement_interval(배기량: float) -> Optional[str]:
    """
    배기량을 구간으로 변환하는 함수
    
    Parameters:
        배기량: float, 배기량 값 (cc 단위)
    
    Returns:
        str: 배기량 구간 레이블
    """
    if pd.isna(배기량) or 배기량 is None:
        return None
    
    배기량 = float(배기량)
    
    if 배기량 == 0:
        return "0cc(전기/수소)"
    elif 배기량 < 500:
        return "0cc(전기/수소)"  # 0~499cc는 0cc로 처리
    elif 배기량 < 1000:
        return "500~999cc"
    elif 배기량 < 1500:
        return "1000~1499cc"
    elif 배기량 < 2000:
        return "1500~1999cc"
    elif 배기량 < 2500:
        return "2000~2499cc"
    elif 배기량 < 3000:
        return "2500~2999cc"
    else:
        return "3000cc~"


def map_grade_code_info(등급코드: int, mapping_dicts: Optional[Dict] = None) -> Dict:
    """
    등급코드로부터 차종, 연료, gen_norm 등 정보 매핑
    
    Parameters:
        등급코드: int, 등급코드
        mapping_dicts: dict, 매핑 딕셔너리들 (None이면 자동 로드)
    
    Returns:
        dict: 등급코드에 해당하는 정보들
    """
    if mapping_dicts is None:
        mapping_dicts = load_mapping_dicts()
    
    grade_code_mapping = mapping_dicts.get('grade_code_mapping', {})
    
    등급코드_str = str(등급코드)
    grade_info = grade_code_mapping.get(등급코드_str, {})
    
    return {
        '차종': grade_info.get('차종'),
        '연료': grade_info.get('연료'),
        'gen_norm': grade_info.get('gen_norm'),
        '판매속도지수_등급코드': grade_info.get('판매속도지수_등급코드'),
        '빈도_등급코드': grade_info.get('빈도_등급코드'),
        '판매속도지수_등급코드_구간': grade_info.get('판매속도지수_등급코드_구간'),
        '빈도_등급코드_구간': grade_info.get('빈도_등급코드_구간'),
        'is_ev_h2': grade_info.get('is_ev_h2'),
        'most_common_cluster': grade_info.get('most_common_cluster'),
        'possible_clusters': grade_info.get('possible_clusters', [])
    }


def get_cluster_from_grade_displacement_price(
    등급코드: int,
    배기량_구간: str,
    신차가격_구간: str,
    mapping_dicts: Optional[Dict] = None
) -> Optional[int]:
    """
    등급코드 + 배기량_구간 + 신차가격_구간 조합으로 클러스터 값 가져오기
    
    Parameters:
        등급코드: int, 등급코드
        배기량_구간: str, 배기량 구간 레이블
        신차가격_구간: str, 신차가격 구간 레이블
        mapping_dicts: dict, 매핑 딕셔너리들 (None이면 자동 로드)
    
    Returns:
        int: 클러스터 값 또는 None
    """
    if mapping_dicts is None:
        mapping_dicts = load_mapping_dicts()
    
    grade_displacement_price_mapping = mapping_dicts.get('grade_displacement_price_mapping', {})
    
    # 키 형식: "등급코드|배기량_구간|신차가격_구간"
    key = f"{등급코드}|{배기량_구간}|{신차가격_구간}"
    
    cluster = grade_displacement_price_mapping.get(key)
    
    if cluster is not None:
        return int(cluster)
    
    # 매핑이 없으면 등급코드 기반 most_common_cluster 사용
    grade_info = map_grade_code_info(등급코드, mapping_dicts)
    return grade_info.get('most_common_cluster')


def get_name_from_code(code_type: str, code: int, mapping_dicts: Optional[Dict] = None) -> Optional[str]:
    """
    코드로부터 이름 가져오기
    
    Parameters:
        code_type: str, "제조사", "모델", "등급", "세부등급" 중 하나
        code: int, 코드 값
        mapping_dicts: dict, 매핑 딕셔너리들 (None이면 자동 로드)
    
    Returns:
        str: 이름 또는 None
    """
    if mapping_dicts is None:
        mapping_dicts = load_mapping_dicts()
    
    name_mapping = mapping_dicts.get('name_mapping', {})
    
    code_type_map = {
        '제조사': '제조사_이름_dict',
        '모델': '모델_이름_dict',
        '등급': '등급_이름_dict',
        '세부등급': '세부등급_이름_dict'
    }
    
    dict_key = code_type_map.get(code_type)
    if dict_key is None:
        return None
    
    name_dict = name_mapping.get(dict_key, {})
    return name_dict.get(str(code))


def calculate_km_per_month(주행거리: float, 차나이_월: int) -> Optional[float]:
    """
    월평균 주행거리 계산
    
    Parameters:
        주행거리: float, 주행거리 (km)
        차나이_월: int, 차나이 (월)
    
    Returns:
        float: 월평균 주행거리 또는 None
    """
    if pd.isna(주행거리) or pd.isna(차나이_월) or 차나이_월 == 0:
        return None
    
    return round(주행거리 / 차나이_월, 2)
