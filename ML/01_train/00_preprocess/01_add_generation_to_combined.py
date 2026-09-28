import pandas as pd
import json
import numpy as np

print("=== Generation 매칭 스크립트 시작 ===\n")

# 경로 설정

BASE_DIR = "03_배포용/00_data"
DATA_PATH1 = f"{BASE_DIR}/01_combined_data/df_car_combined.csv"
DATA_PATH2 = f"{BASE_DIR}/00_raw_data/generation_map_live.csv"
OUTPUT_PATH = f"{BASE_DIR}/02_combined_data_gen/df_car_combined_gen.csv"

print("=== 1) 데이터 로드 ===")
df1 = pd.read_csv(DATA_PATH1, encoding="utf-8-sig", low_memory=False)
df2 = pd.read_csv(DATA_PATH2, encoding="utf-8-sig", low_memory=False)

print(f"df1 shape: {df1.shape}")
print(f"df2 shape: {df2.shape}")

print("\n=== 2) 데이터 전처리 - 연월 형식 통일 ===")

# df1: 연월식 -> year_month (YYYYMM 정수)
df1['연월'] = df1['연월식'].astype(str).str.zfill(6).str[:6]
df1['연월'] = pd.to_numeric(df1['연월'], errors='coerce')

# df2: gen_start_year/month -> gen_start_ym 함수로 처리
def create_year_month(year, month, default_month=1):
    """year와 month를 YYYYMM 형식의 정수로 변환"""
    if pd.isna(year):
        return pd.NA
    year_int = int(year)
    if pd.notna(month):
        month_int = int(month)
    else:
        month_int = default_month
    return year_int * 100 + month_int

# gen_start_ym 생성
df2['gen_start_ym'] = df2.apply(
    lambda row: create_year_month(row['gen_start_year'], row['gen_start_month'], default_month=1),
    axis=1
)

# gen_end_ym 생성
df2['gen_end_ym'] = df2.apply(
    lambda row: create_year_month(row['gen_end_year'], row['gen_end_month'], default_month=12),
    axis=1
)

print(f"df1 연월 범위: {df1['연월'].min()} ~ {df1['연월'].max()}")
print(f"df2 세대 수: {len(df2)}")
print(f"df2 status 분포:\n{df2['status'].value_counts()}")

print("\n=== 3) 세대 매칭 함수 정의 ===")

def extract_model_type(model_name):
    """모델명에서 타입 추출 (액티브 투어러, 쿠페, 그란쿠페)"""
    if pd.isna(model_name):
        return None
    model_str = str(model_name)
    
    # 순서 중요: 더 구체적인 것부터 확인
    if '그란쿠페' in model_str or 'F44' in model_str or 'F74' in model_str or 'F78' in model_str:
        return '그란쿠페'
    elif '액티브 투어러' in model_str or 'F45' in model_str or 'U06' in model_str:
        return '액티브 투어러'
    elif '쿠페' in model_str or 'F22' in model_str or 'F23' in model_str or 'G42' in model_str:
        return '쿠페'
    return None

def extract_gen_type(gen_label):
    """generation_label에서 타입 추출"""
    if pd.isna(gen_label):
        return None
    label_str = str(gen_label)
    
    if '그란쿠페' in label_str:
        return '그란쿠페'
    elif '액티브 투어러' in label_str:
        return '액티브 투어러'
    elif '쿠페' in label_str:
        return '쿠페'
    return None

def match_generation(row, df2_filtered):
    """
    개별 row의 연월에 맞는 세대를 찾는 함수
    BMW 2시리즈의 경우 모델명과 generation_label을 비교하여 정확히 매칭
    """
    if pd.isna(row['연월']) or len(df2_filtered) == 0:
        return None
    
    연월 = row['연월']
    
    # BMW 2시리즈인 경우 모델 타입 추출
    model_type = None
    if row['제조사'] == 'BMW' and row['대표차종명'] == '2시리즈':
        model_type = extract_model_type(row.get('모델', ''))
    
    # 후보 세대들 필터링
    candidates = []
    
    for idx, gen_row in df2_filtered.iterrows():
        # status가 planned인 경우 건너뛰기 (미래 세대는 현재 판매 데이터와 매칭 불가)
        if gen_row['status'] == 'planned':
            continue
        
        # 중국 전용 세대 제외 (대한민국 데이터에는 해당 없음)
        generation_label = str(gen_row.get('generation_label', ''))
        if '중국' in generation_label or 'China' in generation_label:
            continue
        
        # BMW 2시리즈인 경우 모델 타입과 generation_label 타입이 일치해야 함
        if model_type is not None:
            gen_type = extract_gen_type(generation_label)
            if gen_type != model_type:
                continue
            
        gen_start = gen_row['gen_start_ym']
        gen_end = gen_row['gen_end_ym']
        
        # 시작 시점이 없으면 건너뛰기
        if pd.isna(gen_start):
            continue
        
        # 연월이 세대 기간 내에 있는지 확인
        if 연월 >= gen_start:
            if pd.isna(gen_end):  # 현재까지 판매 중
                candidates.append(gen_row)
            elif 연월 <= gen_end:  # 종료 시점 이전
                candidates.append(gen_row)
    
    # 후보가 없으면 None
    if len(candidates) == 0:
        return None
    
    # 후보가 여러 개면 가장 최근 시작한 세대 선택
    if len(candidates) > 1:
        candidates_df = pd.DataFrame(candidates)
        best_match = candidates_df.loc[candidates_df['gen_start_ym'].idxmax()]
        return best_match
    
    return candidates[0]

print("세대 매칭 함수 정의 완료")

print("\n=== 4) 세대 매칭 실행 ===")

# 결과 저장용 리스트
results = []

# 제조사+차종 그룹별로 처리
grouped = df1.groupby(['제조사', '대표차종명'])

total_groups = len(grouped)
print(f"총 {total_groups}개의 제조사+차종 그룹 처리 시작...")

for idx, ((maker, car_name), group) in enumerate(grouped):
    if idx % 100 == 0:
        print(f"진행중: {idx}/{total_groups} 그룹 처리 완료...")
    
    # 해당 제조사+차종의 세대 정보 필터링
    df2_filtered = df2[(df2['제조사'] == maker) & (df2['대표차종명'] == car_name)]
    
    # 각 행에 대해 세대 매칭
    for _, row in group.iterrows():
        matched_gen = match_generation(row, df2_filtered)
        
        if matched_gen is not None:
            result = {
                'index': row.name,
                'gen_index': matched_gen['generation_index'],
                'generation_label': matched_gen['generation_label'],
                'platform_codes': matched_gen['platform_codes'],
                'gen_start_ym': matched_gen['gen_start_ym'],
                'gen_end_ym': matched_gen['gen_end_ym'],
                'status': matched_gen['status'],
                'phases_json': matched_gen['phases_json']
            }
        else:
            result = {
                'index': row.name,
                'gen_index': None,
                'generation_label': None,
                'platform_codes': None,
                'gen_start_ym': None,
                'gen_end_ym': None,
                'status': None,
                'phases_json': None
            }
        
        results.append(result)

# 결과를 DataFrame으로 변환
results_df = pd.DataFrame(results)
results_df.set_index('index', inplace=True)

# df1과 병합
df1_result = df1.join(results_df, how='left')

matched_count = df1_result['gen_index'].notna().sum()
unmatched_count = df1_result['gen_index'].isna().sum()

print(f"\n매칭 성공: {matched_count}개")
print(f"매칭 실패: {unmatched_count}개")

if unmatched_count > 0:
    print(f"\n매칭 실패한 제조사+차종별 분포:")
    unmatched_rows = df1_result[df1_result['gen_index'].isna()]
    print(unmatched_rows.groupby(['제조사', '대표차종명']).size().sort_values(ascending=False).head(20))

print("\n=== 5) status_current 생성 ===")

df1_result['status_current'] = df1_result['status'].map({
    'past': 'discontinued',
    'current': 'active',
    'planned': 'planned'
})

print(f"status_current 분포:\n{df1_result['status_current'].value_counts(dropna=False)}")

print("\n=== 6) detailed_status 생성 ===")

def parse_detailed_status(row):
    """
    phases_json을 파싱하여 해당 연월에 맞는 phase 이름을 반환
    """
    if pd.isna(row['phases_json']) or row['phases_json'] == '[]':
        return 'base'
    
    if pd.isna(row['연월']):
        return 'unknown'
    
    try:
        phases = json.loads(row['phases_json'])
    except:
        return 'unknown'
    
    if not isinstance(phases, list) or len(phases) == 0:
        return 'base'
    
    연월 = row['연월']
    
    # 각 phase를 검사하여 연월이 속하는지 확인
    matched_phases = []
    
    for phase in phases:
        year_from = phase.get('year_from')
        month_from = phase.get('month_from')
        year_to = phase.get('year_to')
        month_to = phase.get('month_to')
        
        # 시작 연월 계산
        if year_from is not None and month_from is not None:
            phase_start = int(year_from) * 100 + int(month_from)
        elif year_from is not None:
            phase_start = int(year_from) * 100 + 1  # 1월로 가정
        else:
            phase_start = None
        
        # 종료 연월 계산
        if year_to is not None and month_to is not None:
            phase_end = int(year_to) * 100 + int(month_to)
        elif year_to is not None:
            phase_end = int(year_to) * 100 + 12  # 12월로 가정
        else:
            phase_end = None
        
        # 연월이 phase 기간 내에 있는지 확인
        if phase_start is not None:
            if 연월 >= phase_start:
                if phase_end is None or 연월 <= phase_end:
                    matched_phases.append(phase)
    
    # 매칭된 phase가 없으면 base
    if len(matched_phases) == 0:
        return 'base'
    
    # 가장 최근에 시작한 phase 선택
    if len(matched_phases) > 1:
        matched_phases.sort(key=lambda p: (
            p.get('year_from', 0) or 0,
            p.get('month_from', 0) or 0
        ), reverse=True)
    
    best_phase = matched_phases[0]
    
    # phase_type 우선, 없으면 phase_label의 일부 추출
    phase_type = best_phase.get('phase_type')
    if phase_type:
        return phase_type
    
    # phase_label에서 타입 추출
    phase_label = best_phase.get('phase_label', '')
    if '초기형' in phase_label or 'early' in phase_label.lower():
        return 'early'
    elif '페이스리프트' in phase_label or 'facelift' in phase_label.lower():
        return 'facelift'
    elif '후기형' in phase_label or 'late' in phase_label.lower():
        return 'late'
    elif '중기형' in phase_label:
        return 'mid'
    else:
        return 'other'

print("phases_json 파싱 중...")
df1_result['detailed_status'] = df1_result.apply(parse_detailed_status, axis=1)

print(f"\ndetailed_status 분포:")
print(df1_result['detailed_status'].value_counts())

print("\n=== 7) gen_max 계산 ===")

# gen_max 계산: 같은 제조사+차종의 최대 세대 번호
gen_max_dict = {}
for (maker, car_name), group in df2.groupby(['제조사', '대표차종명']):
    # planned가 아닌 세대 중 최대값
    valid_gens = group[group['status'] != 'planned']['generation_index']
    if len(valid_gens) > 0:
        gen_max_dict[(maker, car_name)] = valid_gens.max()
    else:
        gen_max_dict[(maker, car_name)] = None

# df1_result에 gen_max 추가
df1_result['gen_max'] = df1_result.apply(
    lambda row: gen_max_dict.get((row['제조사'], row['대표차종명']), None),
    axis=1
)

print(f"gen_max 계산 완료")

print("\n=== 8) 매핑 실패한 행 제거 ===")

# 매핑 실패한 행 (gen_index가 nan인 행) 제거
before_count = len(df1_result)
df1_result = df1_result[df1_result['gen_index'].notna()].copy()
after_count = len(df1_result)
removed_count = before_count - after_count

print(f"제거 전: {before_count}건")
print(f"제거 후: {after_count}건")
print(f"제거된 행: {removed_count}건 ({removed_count/before_count*100:.2f}%)")

if removed_count > 0:
    print(f"\n제거된 행의 제조사+차종별 분포:")
    removed_rows = df1_result[df1_result['gen_index'].isna()] if removed_count > 0 else pd.DataFrame()
    # 제거된 행 정보는 이미 필터링 전에 확인해야 하므로, 별도로 확인
    print("(제거된 행 정보는 위의 매칭 실패 통계 참조)")

print("\n=== 9) 필요한 컬럼만 선택하여 저장 ===")

# 원본 컬럼 + generation 관련 컬럼만 선택
generation_columns = [
    'gen_index',
    'generation_label',
    'platform_codes',
    'gen_start_ym',
    'gen_end_ym',
    'gen_max',
    'status_current',
    'detailed_status'
]

# 원본 df1의 모든 컬럼 + generation 컬럼
output_columns = list(df1.columns) + generation_columns

# 존재하는 컬럼만 선택
available_columns = [col for col in output_columns if col in df1_result.columns]
df_final = df1_result[available_columns]

print(f"최종 데이터 shape: {df_final.shape}")
print(f"포함된 generation 컬럼: {generation_columns}")

print(f"\n=== 10) 저장 ===")
df_final.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
print(f"✅ 저장 완료: {OUTPUT_PATH}")
print(f"   총 {len(df_final)} 행, {len(df_final.columns)} 열")

