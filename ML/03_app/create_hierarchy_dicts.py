"""
계층화된 딕셔너리 구조 생성 스크립트
df_car_combined_gen.csv에서 계층화된 딕셔너리 및 등급코드 매핑 딕셔너리를 생성합니다.
"""
import pandas as pd
import json
import os
import numpy as np
from pathlib import Path
from collections import defaultdict

# 경로 설정
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "00_data" / "02_combined_data_gen"
CLUSTER_DIR = BASE_DIR / "00_data" / "03_result" / "kmodes_model"
OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)

def create_hierarchy_dicts(df, df_cluster):
    """
    계층화된 딕셔너리 구조 생성 (코드만 저장)
    
    Returns:
        dict: 계층화된 딕셔너리들 (코드 기반)
    """
    print("=" * 60)
    print("계층화된 딕셔너리 생성 시작 (코드 기반)...")
    print("=" * 60)
    
    # 1. 국산외제차 -> 제조사코드 딕셔너리 (df_cluster에서 가져오기)
    print("\n1. 국산외제차 -> 제조사코드 딕셔너리 생성 중...")
    국산외제차_dict = defaultdict(set)
    
    for _, row in df_cluster.iterrows():
        국산외제차 = row.get('국산외제차')
        제조사코드 = row.get('제조사코드')
        
        if pd.notna(국산외제차) and pd.notna(제조사코드):
            국산외제차_dict[int(국산외제차)].add(int(제조사코드))
    
    # 리스트로 변환 및 정렬
    국산외제차_dict = {
        k: sorted(list(v))
        for k, v in 국산외제차_dict.items()
    }
    
    print(f"  ✓ 국산차(0): {len(국산외제차_dict.get(0, []))}개 제조사코드")
    print(f"  ✓ 외제차(1): {len(국산외제차_dict.get(1, []))}개 제조사코드")
    
    # 2. 제조사코드 -> 대표차종명 딕셔너리 (코드는 저장하지 않고 이름만)
    print("\n2. 제조사코드 -> 대표차종명 딕셔너리 생성 중...")
    제조사_dict = defaultdict(set)
    
    for _, row in df.iterrows():
        제조사코드 = row.get('제조사코드')
        대표차종명 = row.get('대표차종명')
        
        if pd.notna(제조사코드) and pd.notna(대표차종명):
            제조사_dict[int(제조사코드)].add(대표차종명)
    
    # 리스트로 변환 및 정렬
    제조사_dict = {
        k: sorted(list(v))
        for k, v in 제조사_dict.items()
    }
    
    print(f"  ✓ {len(제조사_dict)}개 제조사코드에 대한 대표차종명 매핑 완료")
    
    # 3. (제조사코드, 대표차종명) -> 모델코드 딕셔너리
    print("\n3. (제조사코드, 대표차종명) -> 모델코드 딕셔너리 생성 중...")
    대표차종명_dict = defaultdict(set)
    
    for _, row in df.iterrows():
        제조사코드 = row.get('제조사코드')
        대표차종명 = row.get('대표차종명')
        모델코드 = row.get('모델코드')
        
        if all(pd.notna(x) for x in [제조사코드, 대표차종명, 모델코드]):
            key = (int(제조사코드), 대표차종명)
            대표차종명_dict[key].add(int(모델코드))
    
    # 리스트로 변환 및 정렬
    대표차종명_dict = {
        k: sorted(list(v))
        for k, v in 대표차종명_dict.items()
    }
    
    print(f"  ✓ {len(대표차종명_dict)}개 (제조사코드, 대표차종명) 조합에 대한 모델코드 매핑 완료")
    
    # 4. 모델코드 -> 등급코드 딕셔너리
    print("\n4. 모델코드 -> 등급코드 딕셔너리 생성 중...")
    모델_dict = defaultdict(set)
    
    for _, row in df.iterrows():
        모델코드 = row.get('모델코드')
        등급코드 = row.get('등급코드')
        
        if all(pd.notna(x) for x in [모델코드, 등급코드]):
            모델_dict[int(모델코드)].add(int(등급코드))
    
    # 리스트로 변환 및 정렬
    모델_dict = {
        k: sorted(list(v))
        for k, v in 모델_dict.items()
    }
    
    print(f"  ✓ {len(모델_dict)}개 모델코드에 대한 등급코드 매핑 완료")
    
    # 5. 등급코드 -> 세부등급코드 딕셔너리
    print("\n5. 등급코드 -> 세부등급코드 딕셔너리 생성 중...")
    등급_dict = defaultdict(set)
    
    for _, row in df.iterrows():
        등급코드 = row.get('등급코드')
        세부등급코드 = row.get('세부등급코드')
        
        if pd.notna(등급코드):
            if pd.notna(세부등급코드):
                등급_dict[int(등급코드)].add(int(세부등급코드))
            else:
                # 세부등급이 없는 경우 None으로 표시
                등급_dict[int(등급코드)].add(None)
    
    # 리스트로 변환 및 정렬 (None은 마지막에)
    등급_dict = {
        k: sorted([x for x in v if x is not None]) + ([None] if None in v else [])
        for k, v in 등급_dict.items()
    }
    
    print(f"  ✓ {len(등급_dict)}개 등급코드에 대한 세부등급코드 매핑 완료")
    
    # 6. 색상 유니크값 리스트 (빈도순으로 정렬)
    print("\n6. 색상 유니크값 리스트 생성 중...")
    색상_빈도 = df['색상'].value_counts()
    색상_list = 색상_빈도.index.tolist()  # 빈도가 높은 순서대로 정렬됨
    print(f"  ✓ {len(색상_list)}개 색상 (빈도순 정렬)")
    print(f"  - 상위 5개: {', '.join(색상_list[:5])}")
    
    return {
        '국산외제차_dict': 국산외제차_dict,
        '제조사_dict': 제조사_dict,
        '대표차종명_dict': 대표차종명_dict,
        '모델_dict': 모델_dict,
        '등급_dict': 등급_dict,
        '색상_list': 색상_list
    }


def create_name_mapping_dicts(df):
    """
    코드 -> 이름 매핑 딕셔너리 생성
    
    Returns:
        dict: 코드별 이름 매핑 딕셔너리들
    """
    print("\n" + "=" * 60)
    print("코드 -> 이름 매핑 딕셔너리 생성 시작...")
    print("=" * 60)
    
    # 1. 제조사코드 -> 제조사명 매핑
    print("\n1. 제조사코드 -> 제조사명 매핑 생성 중...")
    제조사_이름_dict = {}
    
    for _, row in df.iterrows():
        제조사코드 = row.get('제조사코드')
        제조사 = row.get('제조사')
        
        if pd.notna(제조사코드) and pd.notna(제조사):
            제조사코드 = int(제조사코드)
            if 제조사코드 not in 제조사_이름_dict:
                제조사_이름_dict[제조사코드] = 제조사
    
    print(f"  ✓ {len(제조사_이름_dict)}개 제조사코드 매핑 완료")
    
    # 2. 모델코드 -> 모델명 매핑
    print("\n2. 모델코드 -> 모델명 매핑 생성 중...")
    모델_이름_dict = {}
    
    for _, row in df.iterrows():
        모델코드 = row.get('모델코드')
        모델 = row.get('모델')
        
        if pd.notna(모델코드) and pd.notna(모델):
            모델코드 = int(모델코드)
            if 모델코드 not in 모델_이름_dict:
                모델_이름_dict[모델코드] = 모델
    
    print(f"  ✓ {len(모델_이름_dict)}개 모델코드 매핑 완료")
    
    # 3. 등급코드 -> 등급명 매핑
    print("\n3. 등급코드 -> 등급명 매핑 생성 중...")
    등급_이름_dict = {}
    
    for _, row in df.iterrows():
        등급코드 = row.get('등급코드')
        등급 = row.get('등급')
        
        if pd.notna(등급코드) and pd.notna(등급):
            등급코드 = int(등급코드)
            if 등급코드 not in 등급_이름_dict:
                등급_이름_dict[등급코드] = 등급
    
    print(f"  ✓ {len(등급_이름_dict)}개 등급코드 매핑 완료")
    
    # 4. 세부등급코드 -> 세부등급명 매핑
    print("\n4. 세부등급코드 -> 세부등급명 매핑 생성 중...")
    세부등급_이름_dict = {}
    
    for _, row in df.iterrows():
        세부등급코드 = row.get('세부등급코드')
        세부등급 = row.get('세부등급')
        
        if pd.notna(세부등급코드) and pd.notna(세부등급):
            세부등급코드 = int(세부등급코드)
            if 세부등급코드 not in 세부등급_이름_dict:
                세부등급_이름_dict[세부등급코드] = 세부등급
    
    print(f"  ✓ {len(세부등급_이름_dict)}개 세부등급코드 매핑 완료")
    
    return {
        '제조사_이름_dict': 제조사_이름_dict,
        '모델_이름_dict': 모델_이름_dict,
        '등급_이름_dict': 등급_이름_dict,
        '세부등급_이름_dict': 세부등급_이름_dict
    }


def create_grade_code_mapping(df, df_cluster):
    """
    등급코드 기반 매핑 딕셔너리 생성 (기존 방식 - 하위 호환성 유지)
    
    Returns:
        dict: 등급코드별 매핑 정보
    """
    print("\n" + "=" * 60)
    print("등급코드 기반 매핑 딕셔너리 생성 시작 (기존 방식)...")
    print("=" * 60)
    
    grade_mapping = {}
    
    # df_cluster에서 등급코드별 통계량 집계
    for 등급코드 in df_cluster['등급코드'].dropna().unique():
        등급코드 = int(등급코드)
        grade_data = df_cluster[df_cluster['등급코드'] == 등급코드]
        
        if len(grade_data) == 0:
            continue
        
        # 차종, 연료는 첫 번째 값 사용 (일반적으로 동일)
        차종 = grade_data['차종'].iloc[0] if '차종' in grade_data.columns else None
        연료 = grade_data['연료'].iloc[0] if '연료' in grade_data.columns else None
        
        # gen_norm은 평균값 사용
        gen_norm = grade_data['gen_norm'].mean() if 'gen_norm' in grade_data.columns else None
        
        # 판매속도지수_등급코드: 첫 번째 값
        판매속도지수 = grade_data['판매속도지수_등급코드'].iloc[0] if '판매속도지수_등급코드' in grade_data.columns else None
        
        # 빈도_등급코드: 첫 번째 값
        빈도 = grade_data['빈도_등급코드'].iloc[0] if '빈도_등급코드' in grade_data.columns else None
        
        # 판매속도지수_등급코드_구간: 최빈값
        판매속도지수_구간 = grade_data['판매속도지수_등급코드_구간'].mode().iloc[0] if '판매속도지수_등급코드_구간' in grade_data.columns and len(grade_data['판매속도지수_등급코드_구간'].mode()) > 0 else None
        
        # 빈도_등급코드_구간: 최빈값
        빈도_구간 = grade_data['빈도_등급코드_구간'].mode().iloc[0] if '빈도_등급코드_구간' in grade_data.columns and len(grade_data['빈도_등급코드_구간'].mode()) > 0 else None
        
        # is_ev_h2: 첫 번째 값 또는 배기량 기반 계산
        is_ev_h2 = None
        if 'is_ev_h2' in grade_data.columns:
            is_ev_h2 = int(grade_data['is_ev_h2'].iloc[0]) if pd.notna(grade_data['is_ev_h2'].iloc[0]) else None
        elif '배기량_구간' in grade_data.columns:
            # 배기량_구간에서 0cc인지 확인
            배기량_구간 = grade_data['배기량_구간'].iloc[0]
            if pd.notna(배기량_구간) and ('0cc' in str(배기량_구간) or str(배기량_구간) == '0cc(전기/수소)'):
                is_ev_h2 = 1
            else:
                is_ev_h2 = 0
        
        # 클러스터: 가장 많이 속한 클러스터
        cluster_counts = grade_data['cluster'].value_counts()
        most_common_cluster = int(cluster_counts.index[0]) if len(cluster_counts) > 0 else None
        possible_clusters = sorted(grade_data['cluster'].unique().tolist())
        
        grade_mapping[등급코드] = {
            '차종': str(차종) if 차종 is not None else None,
            '연료': str(연료) if 연료 is not None else None,
            'gen_norm': float(gen_norm) if gen_norm is not None else None,
            '판매속도지수_등급코드': float(판매속도지수) if 판매속도지수 is not None else None,
            '빈도_등급코드': int(빈도) if 빈도 is not None else None,
            '판매속도지수_등급코드_구간': str(판매속도지수_구간) if 판매속도지수_구간 is not None else None,
            '빈도_등급코드_구간': str(빈도_구간) if 빈도_구간 is not None else None,
            'is_ev_h2': is_ev_h2,
            'most_common_cluster': most_common_cluster,
            'possible_clusters': [int(c) for c in possible_clusters]
        }
    
    print(f"  ✓ {len(grade_mapping)}개 등급코드에 대한 매핑 정보 생성 완료")
    
    return grade_mapping


def get_displacement_interval(배기량):
    """
    배기량을 구간으로 변환하는 헬퍼 함수
    
    Parameters:
        배기량: int or float, 배기량 값 (cc 단위)
    
    Returns:
        str: 배기량 구간 레이블
    """
    배기량 = float(배기량) if pd.notna(배기량) else np.nan
    
    if pd.isna(배기량):
        return None
    
    if 배기량 == 0:
        return "0cc"
    elif 배기량 < 500:
        return "0cc"  # 0~499cc는 0cc로 처리
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


def get_price_interval(신차가격):
    """
    신차가격을 구간으로 변환하는 헬퍼 함수 (만원 단위)
    
    Parameters:
        신차가격: int or float, 신차가격 값 (원 단위 또는 만원 단위)
    
    Returns:
        str: 신차가격 구간 레이블
    """
    신차가격 = float(신차가격) if pd.notna(신차가격) else np.nan
    
    if pd.isna(신차가격):
        return None
    
    # 만원 단위로 변환 (1억 이상이면 만원 단위로 가정)
    if 신차가격 >= 10000:
        # 이미 만원 단위
        price_in_만원 = 신차가격
    else:
        # 원 단위로 가정하고 만원 단위로 변환
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


def create_grade_displacement_price_mapping(df_cluster):
    """
    등급코드 + 배기량_구간 + 신차가격_구간 조합 기반 클러스터 매핑 딕셔너리 생성
    
    이 함수는 등급코드만으로는 부족하고, 배기량과 가격구간도 고려해야 
    정확한 클러스터를 얻을 수 있도록 하는 새로운 매핑 구조를 생성합니다.
    
    Returns:
        dict: "등급코드|배기량_구간|신차가격_구간" 형식의 키로 클러스터 정보를 매핑
    """
    print("\n" + "=" * 60)
    print("등급코드 + 배기량_구간 + 신차가격_구간 조합 기반 매핑 딕셔너리 생성 시작...")
    print("=" * 60)
    
    # 필요한 컬럼 확인
    required_cols = ['등급코드', '배기량_구간', '신차가격_구간', 'cluster']
    missing_cols = [col for col in required_cols if col not in df_cluster.columns]
    if missing_cols:
        raise ValueError(f"필수 컬럼이 없습니다: {missing_cols}")
    
    # 데이터 준비
    mapping_data = df_cluster[required_cols].dropna()
    
    if len(mapping_data) == 0:
        print("  ⚠ 매핑할 데이터가 없습니다.")
        return {}
    
    print(f"  ✓ 총 {len(mapping_data):,}건의 데이터로 매핑 생성")
    
    # 조합별로 클러스터 집계 (각 조합은 단일 클러스터만 가짐)
    combo_cluster = mapping_data.groupby(['등급코드', '배기량_구간', '신차가격_구간'])['cluster'].first().reset_index()
    combo_cluster.columns = ['등급코드', '배기량_구간', '신차가격_구간', 'cluster']
    
    # 매핑 딕셔너리 생성 (최적화된 구조: 키에 정보가 있으므로 값은 cluster만 저장)
    mapping_dict = {}
    for _, row in combo_cluster.iterrows():
        # 키 형식: "등급코드|배기량_구간|신차가격_구간"
        key = f"{int(row['등급코드'])}|{row['배기량_구간']}|{row['신차가격_구간']}"
        
        # 값은 cluster만 저장 (키에 이미 등급코드, 배기량_구간, 신차가격_구간 정보가 포함됨)
        mapping_dict[key] = int(row['cluster']) if pd.notna(row['cluster']) else None
    
    print(f"  ✓ {len(mapping_dict):,}개 조합에 대한 매핑 정보 생성 완료")
    
    return mapping_dict


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("딕셔너리 생성 스크립트 시작")
    print("=" * 60)
    
    # 1. 데이터 로드
    print("\n데이터 로드 중...")
    df_path = DATA_DIR / "df_car_combined_gen.csv"
    cluster_path = CLUSTER_DIR / "df_train_cluster.csv"
    
    if not df_path.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {df_path}")
    if not cluster_path.exists():
        raise FileNotFoundError(f"클러스터 파일을 찾을 수 없습니다: {cluster_path}")
    
    print(f"  ✓ {df_path} 로드 중...")
    df = pd.read_csv(df_path, low_memory=False)
    print(f"    - 총 {len(df):,}건의 데이터")
    
    print(f"  ✓ {cluster_path} 로드 중...")
    df_cluster = pd.read_csv(cluster_path, low_memory=False)
    print(f"    - 총 {len(df_cluster):,}건의 클러스터 데이터")
    
    # 2. 계층화된 딕셔너리 생성 (코드 기반)
    hierarchy_dicts = create_hierarchy_dicts(df, df_cluster)
    
    # 3. 코드 -> 이름 매핑 딕셔너리 생성
    name_mapping_dicts = create_name_mapping_dicts(df)
    
    # 4. 등급코드 매핑 딕셔너리 생성 (기존 방식)
    grade_code_mapping = create_grade_code_mapping(df, df_cluster)
    
    # 5. 등급코드 + 배기량_구간 + 신차가격_구간 조합 기반 매핑 생성 (새로운 방식)
    grade_displacement_price_mapping = create_grade_displacement_price_mapping(df_cluster)
    
    # 6. 결과 저장
    print("\n" + "=" * 60)
    print("결과 저장 중...")
    print("=" * 60)
    
    # JSON으로 저장 (튜플은 문자열로 변환)
    def convert_for_json(obj):
        """JSON 직렬화를 위한 변환 함수"""
        if isinstance(obj, dict):
            return {str(k): convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_for_json(item) for item in obj]
        elif isinstance(obj, tuple):
            return list(obj)
        elif pd.isna(obj):
            return None
        elif obj is None:
            return None
        elif isinstance(obj, (int, float, str, bool)):
            return obj
        else:
            return str(obj)
    
    # 계층화된 딕셔너리 저장 (코드 기반)
    hierarchy_output = {}
    for key, value in hierarchy_dicts.items():
        if key == '색상_list':
            hierarchy_output[key] = value
        else:
            hierarchy_output[key] = convert_for_json(value)
    
    hierarchy_path = OUTPUT_DIR / "hierarchy_dicts.json"
    with open(hierarchy_path, 'w', encoding='utf-8') as f:
        json.dump(hierarchy_output, f, ensure_ascii=False, indent=2)
    print(f"  ✓ {hierarchy_path} 저장 완료")
    
    # 코드 -> 이름 매핑 딕셔너리 저장
    name_mapping_output = convert_for_json(name_mapping_dicts)
    name_mapping_path = OUTPUT_DIR / "name_mapping.json"
    with open(name_mapping_path, 'w', encoding='utf-8') as f:
        json.dump(name_mapping_output, f, ensure_ascii=False, indent=2)
    print(f"  ✓ {name_mapping_path} 저장 완료")
    
    # 등급코드 매핑 저장 (기존 방식)
    grade_mapping_path = OUTPUT_DIR / "grade_code_mapping.json"
    with open(grade_mapping_path, 'w', encoding='utf-8') as f:
        json.dump(convert_for_json(grade_code_mapping), f, ensure_ascii=False, indent=2)
    print(f"  ✓ {grade_mapping_path} 저장 완료")
    
    # 등급코드 + 배기량_구간 + 신차가격_구간 조합 매핑 저장 (새로운 방식)
    grade_displacement_price_mapping_path = OUTPUT_DIR / "grade_displacement_price_mapping.json"
    with open(grade_displacement_price_mapping_path, 'w', encoding='utf-8') as f:
        json.dump(convert_for_json(grade_displacement_price_mapping), f, ensure_ascii=False, indent=2)
    print(f"  ✓ {grade_displacement_price_mapping_path} 저장 완료")
    
    print("\n" + "=" * 60)
    print("딕셔너리 생성 완료!")
    print("=" * 60)
    print(f"\n생성된 파일:")
    print(f"  - {hierarchy_path} (계층화된 딕셔너리 - 코드 기반)")
    print(f"  - {name_mapping_path} (코드 -> 이름 매핑)")
    print(f"  - {grade_mapping_path} (등급코드 매핑 - 기존 방식)")
    print(f"  - {grade_displacement_price_mapping_path} (등급코드+배기량+가격 조합 매핑 - 새로운 방식)")


if __name__ == "__main__":
    main()
