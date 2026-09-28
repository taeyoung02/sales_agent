"""
LGBM 모델링을 위한 전처리 함수 모듈
클러스터별로 나누어 처리하는 용도
"""
import numpy as np
import pandas as pd


def drop_columns_for_lgbm(df: pd.DataFrame) -> pd.DataFrame:
    """
    LGBM 모델링에 불필요한 컬럼들을 제거하는 함수
    
    Parameters
    ----------
    df : pd.DataFrame
        입력 데이터프레임
    
    Returns
    -------
    pd.DataFrame
        불필요한 컬럼이 제거된 데이터프레임
    """
    df = df.copy()
    
    # 제거할 컬럼 리스트
    # 주의: '판매신고가'는 감가율 계산에 필요하므로 여기서는 제거하지 않음
    # 감가율 계산 후 별도로 제거됨 (데이터 누출 방지)
    columns_to_drop = [
        '제조사', 
        '모델', 
        '등급', 
        '세부등급', 
        '연월식',
        '배기량',
        '교환부위',
        '판금부위',
        '최초 광고 등록일', 
        '최초 광고가', 
        '판매신고일', 
        'has_planned_next_gen', 
        'log_km_per_month', 
        'mileage_ratio', 
        'sale_month',
        'sale_week',
        'sale_day',
        'day_normalized',
        'day_sin',
        'day_cos'

    ]
    
    # 존재하는 컬럼만 제거
    existing_columns = [col for col in columns_to_drop if col in df.columns]
    if existing_columns:
        df = df.drop(columns=existing_columns, errors='ignore')
        print(f"✓ 제거된 컬럼: {len(existing_columns)}개")
        print(f"  - {', '.join(existing_columns)}")
    else:
        print("✓ 제거할 컬럼이 없습니다.")
    
    return df


def categorize_displacement(df: pd.DataFrame, 
                            disp_col: str = "배기량") -> pd.DataFrame:
    """
    배기량을 순서형(ordinal) 카테고리로 변환하는 함수
    - 배기량 0인 경우는 전기/수소차로 별도 카테고리로 유지
    - LightGBM은 NaN을 처리할 수 있으므로 0인 경우도 포함
    
    Parameters
    ----------
    df : pd.DataFrame
        입력 데이터프레임
    disp_col : str, default="배기량"
        배기량 컬럼명
    
    Returns
    -------
    pd.DataFrame
        배기량이 순서형 카테고리로 변환된 데이터프레임
    """
    df = df.copy()
    
    # 배기량 컬럼이 없으면 에러
    if disp_col not in df.columns:
        print(f"⚠ '{disp_col}' 컬럼이 없습니다. 건너뜁니다.")
        return df
    
    # 배기량을 숫자형으로 변환
    df[disp_col] = pd.to_numeric(df[disp_col], errors='coerce')
    
    # 배기량 0인 경우 확인 (전기/수소차)
    zero_count = (df[disp_col] == 0).sum()
    if zero_count > 0:
        print(f"✓ 배기량 0인 차량(전기/수소차): {zero_count:,}건")
    
    # 배기량 구간 설정 (0 포함, 순서형)
    # 0은 별도 카테고리로 유지
    # ENGINE_BINS의 순서대로 순서형 카테고리 생성: 0 < 500 < 1000 < 1500 < ...
    ENGINE_BINS = [0, 500, 1000, 1500, 2000, 2500, 3000, np.inf]
    ENGINE_LABELS = [
        "0cc(전기/수소)",
        "500~999cc",
        "1000~1499cc",
        "1500~1999cc",
        "2000~2499cc",
        "2500~2999cc",
        "3000cc~"
    ]
    # ENGINE_LABELS의 순서는 ENGINE_BINS의 순서와 일치해야 함 (ordered=True 기준)
    
    # 배기량을 구간으로 나누고 순서형 카테고리로 변환
    # ordered=True: ENGINE_BINS의 순서(0 < 500 < 1000 < ...)를 기준으로 순서형 카테고리 생성
    # LightGBM에서 순서 정보를 활용하려면 categorical_feature로 지정 필요
    df[disp_col] = pd.cut(
        df[disp_col],
        bins=ENGINE_BINS,
        labels=ENGINE_LABELS,
        include_lowest=True,
        right=False,
        ordered=True  # 순서형 카테고리 (ENGINE_BINS 순서 기준)
    )
    
    print(f"✓ 배기량을 순서형 카테고리로 변환 완료")
    print(f"  - 배기량 분포:")
    print(df[disp_col].value_counts().sort_index())
    
    return df


def _convert_to_ordinal_categorical(series: pd.Series, categories: list) -> pd.Series:
    """
    시리즈를 순서형 카테고리로 변환하는 헬퍼 함수
    
    Parameters
    ----------
    series : pd.Series
        변환할 시리즈
    categories : list
        카테고리 순서 리스트
    
    Returns
    -------
    pd.Series
        순서형 카테고리로 변환된 시리즈
    """
    if hasattr(series, 'cat'):
        # 이미 카테고리 타입인 경우
        existing_cats = [cat for cat in categories if cat in series.cat.categories]
        return series.cat.reorder_categories(existing_cats, ordered=True)
    else:
        # 문자열 타입인 경우 카테고리로 변환
        # 주의: dropna()는 unique 값 찾기용이며, 원본 시리즈는 변경하지 않음
        # NaN이 포함된 경우에도 원본 데이터는 그대로 유지됨
        unique_vals = series.dropna().unique()  # NaN 제외하고 unique 값만 확인
        existing_cats = [cat for cat in categories if cat in unique_vals]
        # 원본 시리즈를 그대로 사용하여 카테고리 변환 (NaN 포함)
        return pd.Categorical(series, categories=existing_cats, ordered=True)


def _get_interval_categories(series: pd.Series) -> list:
    """
    구간 레이블(구간1, 구간2, ...)을 숫자 순서로 정렬하는 헬퍼 함수
    
    Parameters
    ----------
    series : pd.Series
        구간 레이블이 있는 시리즈
    
    Returns
    -------
    list
        정렬된 카테고리 리스트
    """
    def get_interval_num(x):
        if pd.isna(x):
            return 0
        if isinstance(x, str) and '구간' in x:
            try:
                return int(x.replace('구간', ''))
            except:
                return 0
        return 0
    
    if hasattr(series, 'cat'):
        categories = series.cat.categories.tolist()
    else:
        categories = series.dropna().unique().tolist()
    
    return sorted(categories, key=get_interval_num)


def convert_binned_features_to_ordinal(df: pd.DataFrame) -> pd.DataFrame:
    """
    이미 생성된 구간 컬럼들을 순서형(ordinal) 카테고리로 변환하는 함수
    - 배기량_구간: 0cc를 제외하고 순서형으로 변환 (0cc는 전기/수소차로 별도 플래그 생성)
    - 신차가격_구간: 가격 순서대로 순서형 변환
    - 빈도_등급코드_구간: 구간1 < 구간2 < ... 순서로 순서형 변환
    - 판매속도지수_등급코드_구간: 구간1 < 구간2 < ... 순서로 순서형 변환
    
    Parameters
    ----------
    df : pd.DataFrame
        입력 데이터프레임
    
    Returns
    -------
    pd.DataFrame
        구간 컬럼들이 순서형 카테고리로 변환된 데이터프레임
    """
    df = df.copy()
    
    # 배기량_구간 처리: 전기/수소차 플래그 생성 + 0cc 제외하고 순서형으로 변환
    if '배기량_구간' in df.columns:
        # 1) 전기/수소차 플래그 생성 (0cc인 경우)
        # 주의: 이 플래그는 전기차 행을 제거하는 것이 아니라, 별도 컬럼으로 정보를 보존
        배기량_구간_str = df['배기량_구간'].astype(str)
        is_ev_h2 = (배기량_구간_str == '0cc') | 배기량_구간_str.str.contains('0cc', na=False)
        df['is_ev_h2'] = is_ev_h2.astype(int)  # 전기차=1, 일반차=0
        
        # 2) 배기량_구간에서 0cc를 제외한 나머지 구간만 순서형으로 변환
        # 0cc는 전기/수소차로 순서 관계가 없으므로 제외
        engine_order = [
            "500~999cc",
            "1000~1499cc",
            "1500~1999cc",
            "2000~2499cc",
            "2500~2999cc",
            "3000cc~"
        ]
        
        # 3) 배기량_구간 복사본 생성 (원본 데이터는 유지)
        배기량_구간_copy = df['배기량_구간'].copy()
        
        # 4) 전기/수소차(0cc)인 경우 배기량_구간을 NaN으로 설정
        # 주의: 이는 행을 제거하는 것이 아니라, 해당 컬럼만 NaN으로 설정
        # 전기차 행은 그대로 유지되며, is_ev_h2=1로 정보가 보존됨
        배기량_구간_copy.loc[is_ev_h2] = pd.NA
        
        # 5) 0cc가 아닌 경우만 순서형으로 변환
        # dropna()는 unique 값 찾기용이며, 원본 데이터는 변경하지 않음
        배기량_구간_ordinal = _convert_to_ordinal_categorical(배기량_구간_copy, engine_order)
        df['배기량_구간'] = 배기량_구간_ordinal
        
        # 결과 요약
        ev_h2_count = is_ev_h2.sum()
        total_rows = len(df)
        print("✓ 배기량_구간 처리 완료:")
        print(f"  - 전기/수소차 플래그(is_ev_h2) 생성: {ev_h2_count:,}건 (전체 {total_rows:,}건 중)")
        print(f"  - 배기량_구간: 0cc 제외, 나머지 {len(engine_order)}개 구간을 순서형으로 변환")
        print(f"  - 전기차 행은 유지됨: is_ev_h2=1, 배기량_구간=NaN")
    
    # 신차가격_구간을 순서형으로 변환
    if '신차가격_구간' in df.columns:
        price_order = [
            "2천만원 미만",
            "2천만원 이상 ~ 3천만원 미만",
            "3천만원 이상 ~ 4천만원 미만",
            "4천만원 이상 ~ 5천만원 미만",
            "5천만원 이상 ~ 7천만원 미만",
            "7천만원 이상 ~ 1억원 미만",
            "1억원 이상"
        ]
        df['신차가격_구간'] = _convert_to_ordinal_categorical(df['신차가격_구간'], price_order)
        print("✓ 신차가격_구간을 순서형 카테고리로 변환 완료")
    
    # 빈도_등급코드_구간을 순서형으로 변환
    if '빈도_등급코드_구간' in df.columns:
        categories_sorted = _get_interval_categories(df['빈도_등급코드_구간'])
        df['빈도_등급코드_구간'] = _convert_to_ordinal_categorical(df['빈도_등급코드_구간'], categories_sorted)
        print("✓ 빈도_등급코드_구간을 순서형 카테고리로 변환 완료")
    
    # 판매속도지수_등급코드_구간을 순서형으로 변환
    if '판매속도지수_등급코드_구간' in df.columns:
        categories_sorted = _get_interval_categories(df['판매속도지수_등급코드_구간'])
        df['판매속도지수_등급코드_구간'] = _convert_to_ordinal_categorical(df['판매속도지수_등급코드_구간'], categories_sorted)
        print("✓ 판매속도지수_등급코드_구간을 순서형 카테고리로 변환 완료")
    
    return df


def remove_nan_rows_in_binned_features(df: pd.DataFrame,
                                       check_cols: list = None,
                                       verbose: bool = True) -> pd.DataFrame:
    """
    구간 피처에서 NaN이 있는 행을 제거하는 함수
    (Test set의 unseen 등급코드로 인한 NaN 처리용)
    
    Parameters
    ----------
    df : pd.DataFrame
        입력 데이터프레임
    check_cols : list, optional
        NaN을 확인할 컬럼 리스트
        None이면 ['판매속도지수_등급코드_구간', '빈도_등급코드_구간'] 확인
    verbose : bool, default=True
        제거 과정을 출력할지 여부
    
    Returns
    -------
    pd.DataFrame
        NaN이 제거된 데이터프레임
    """
    df = df.copy()
    
    if check_cols is None:
        check_cols = ['판매속도지수_등급코드_구간', '빈도_등급코드_구간']
    
    # 실제 존재하는 컬럼만 확인
    check_cols = [col for col in check_cols if col in df.columns]
    
    if not check_cols:
        if verbose:
            print("  ✓ 확인할 구간 컬럼이 없습니다.")
        return df
    
    # NaN 확인
    before_len = len(df)
    nan_mask = df[check_cols].isna().any(axis=1)
    nan_count = nan_mask.sum()
    
    if nan_count > 0:
        if verbose:
            # 각 컬럼별 NaN 개수 출력
            print(f"  ⚠ 구간 피처 NaN 발견:")
            for col in check_cols:
                col_nan = df[col].isna().sum()
                if col_nan > 0:
                    print(f"    - {col}: {col_nan}건")
        
        # NaN이 있는 행 제거
        df = df[~nan_mask].copy()
        removed_count = before_len - len(df)
        
        if verbose:
            print(f"  → NaN 행 제거: {removed_count}건 ({removed_count/before_len*100:.2f}%)")
            print(f"  → 제거 후: {len(df):,}건")
    else:
        if verbose:
            print("  ✓ 구간 피처 NaN 없음")
    
    return df


def convert_to_categorical(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    LightGBM 호환성을 위해 명목형 컬럼들을 category 타입으로 변환하는 함수
    - object 타입 컬럼들을 category로 변환
    - 국산외제차 (0: 국산차, 1: 외제차)를 category로 변환
    - 코드/식별자 컬럼들(제조사코드, 모델코드, 등급코드, 세부등급코드)을 category로 변환
    
    Parameters
    ----------
    df : pd.DataFrame
        입력 데이터프레임
    verbose : bool, default=True
        변환 과정을 출력할지 여부
    
    Returns
    -------
    pd.DataFrame
        카테고리로 변환된 데이터프레임
    """
    df = df.copy()
    converted_cols = []
    
    # 1) object 타입 컬럼들을 category로 변환
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    if object_cols:
        for col in object_cols:
            df[col] = df[col].astype('category')
            converted_cols.append(col)
        if verbose:
            print(f"  ✓ {len(object_cols)}개 object 컬럼 변환: {', '.join(object_cols)}")
    
    # 2) 국산외제차를 category로 변환 (0: 국산차, 1: 외제차)
    if '국산외제차' in df.columns:
        df['국산외제차'] = df['국산외제차'].astype('category')
        converted_cols.append('국산외제차')
        if verbose:
            print("  ✓ 국산외제차를 category로 변환 완료")
    
    # 3) 코드/식별자 컬럼들도 category로 변환 (숫자지만 명목형 카테고리)
    code_cols = ['제조사코드', '모델코드', '등급코드', '세부등급코드']
    code_cols_found = [col for col in code_cols if col in df.columns]
    if code_cols_found:
        for col in code_cols_found:
            df[col] = df[col].astype('category')
            converted_cols.append(col)
        if verbose:
            print(f"  ✓ {len(code_cols_found)}개 코드 컬럼을 category로 변환: {', '.join(code_cols_found)}")
    
    # 4) is_ev_h2를 category로 변환 (convert_binned_features_to_ordinal에서 생성됨)
    if 'is_ev_h2' in df.columns:
        df['is_ev_h2'] = df['is_ev_h2'].astype('category')
        if verbose:
            print("  ✓ is_ev_h2를 category로 변환 완료")
    
    if verbose and not converted_cols:
        print("  ✓ 변환할 컬럼이 없습니다.")
    
    return df
