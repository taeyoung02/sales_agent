"""
세그먼테이션을 위한 전처리 함수 모듈
"""
import numpy as np
import pandas as pd


def clean_num_space_comma(x):
    """
    숫자 문자열에서 공백과 쉼표를 제거하고 정수로 변환
    
    Parameters
    ----------
    x : str, int, float
        변환할 값
    
    Returns
    -------
    int or np.nan
        변환된 정수값 또는 nan
    """
    if pd.isna(x):
        return np.nan
    x = str(x).strip().replace(",", "")
    return int(x) if x.isdigit() else np.nan


def apply_optimal_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    데이터프레임의 컬럼들을 최적의 데이터 타입으로 변환
    
    Parameters
    ----------
    df : pd.DataFrame
        입력 데이터프레임
    
    Returns
    -------
    pd.DataFrame
        데이터 타입이 최적화된 데이터프레임
    """
    df = df.copy()

    def to_int_series(s: pd.Series, dtype: str):
        # 쉼표/원/km/cc 등 숫자 아닌 문자 제거 -> nullable int로
        out = (
            s.astype(str)
             .str.replace(r"[^\d]", "", regex=True)
             .replace("", pd.NA)
        )
        return out.astype(dtype)

    # -----------------------------
    # 0) 컬럼 삭제
    # -----------------------------
    df = df.drop(['선택 추가 옵션1','선택 추가 옵션2','선택 추가 옵션3','선택 추가 옵션4',
                      '선택 추가 옵션5','선택 추가 옵션6','선택 추가 옵션7','선택 추가 옵션8',
                      '선택 추가 옵션9','선택 추가 옵션10','선택 추가 옵션11','선택 추가 옵션12',
                      '선택 추가 옵션13','차량번호', '등록번호'], axis=1, errors='ignore')
    
    # -----------------------------
    # 1) 숫자형 변환
    # -----------------------------
    df["신차가격(선택옵션포함)"] = to_int_series(df["신차가격(선택옵션포함)"], "Int64")//10000
    df["주행거리"] = to_int_series(df["주행거리"], "Int32")
    df["배기량"] = to_int_series(df["배기량"], "Int32")

    # 광고가/판매신고가: 예시상 "만원 단위" 숫자처럼 보임 (4150, 4050 등)
    df["최초 광고가"] = to_int_series(df["최초 광고가"], "Int32")
    df["판매신고가"] = to_int_series(df["판매신고가"], "Int32")

    # -----------------------------
    # 2) 날짜 변환
    # -----------------------------
    df["최초 광고 등록일"] = pd.to_datetime(df["최초 광고 등록일"], errors="coerce")
    df["판매신고일"] = pd.to_datetime(df["판매신고일"], errors="coerce")

    # -----------------------------
    # 3) YYYYMM 계열 정리
    # 연월식: YYYYMM 형식 (예: 202101) -> Period 타입으로 변환 (날짜 연산 용이)
    # 연월: 연월식과 중복이므로 제거
    # -----------------------------
    # 연월식을 Period 타입으로 변환 (YYYYMM 형식 -> Period)
    # NaN이 없으므로 간단하게 변환
    연월식_num = pd.to_numeric(df["연월식"], errors="coerce")
    # 정수로 변환 후 문자열로 변환하고 6자리로 맞춤
    연월식_str = 연월식_num.astype(int).astype(str).str.zfill(6)
    df["연월식"] = pd.to_datetime(연월식_str, format='%Y%m', errors='coerce').dt.to_period('M')
    
    # 연월은 연월식과 중복이므로 제거
    if "연월" in df.columns:
        df = df.drop(columns=["연월"])

    # gen_start_ym / gen_end_ym: 202101.0 같은 float -> Int32
    for col in ["gen_start_ym", "gen_end_ym"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(0).astype("Int32")

    # 세부등급코드 NaN 마스크
    mask = df['세부등급코드'].isna()

    # float → int → str 변환 (.0 제거)
    df['세부등급코드'] = df['세부등급코드'].fillna(-1).astype(int).astype(str)
    df.loc[mask, '세부등급코드'] = '_missing_'
    if '세부등급' in df.columns:
        df.loc[mask, '세부등급'] = '_missing_'

    # -----------------------------
    # 4) 명목형 범주형(nominal category)로 묶기
    # 순서형 범주형(ordinal)은 create_binned_feature에서 처리됨
    # -----------------------------
    nominal_category_cols = [
        "제조사", "대표차종명", "모델", "등급", "세부등급",
        "제조사코드", "모델코드", "등급코드", "세부등급코드",
        "색상", "차종", "연료",
        "용도변경이력", "사고유무", "단순수리",
    #    "교환부위", "판금부위",
        "generation_label", "platform_codes",
        "status_current", "detailed_status",
    ]
    for c in nominal_category_cols:
        if c in df.columns:
            df[c] = df[c].astype("category")  # 명목형 (ordered=False가 기본값)

    # -----------------------------
    # 5) 나머지 수치형
    # -----------------------------
    if "gen_index" in df.columns:
        df["gen_index"] = pd.to_numeric(df["gen_index"], errors="coerce").astype("int8")
    if "gen_max" in df.columns:
        df["gen_max"] = pd.to_numeric(df["gen_max"], errors="coerce").fillna(0).astype("int8")
    if "if_planned" in df.columns:
        df["if_planned"] = pd.to_numeric(df["if_planned"], errors="coerce").fillna(0).astype("int8")
    if "if_planned_year" in df.columns:
        df["if_planned_year"] = pd.to_numeric(df["if_planned_year"], errors="coerce").astype("Int16")

    return df


def create_binned_feature(df, col, bins, labels, new_col_name, right=False, ordered=True):
    """
    범용 구간화 함수
    
    Parameters
    ----------
    df : pd.DataFrame
        입력 데이터프레임
    col : str
        구간화할 컬럼명
    bins : list
        구간 경계값 리스트
    labels : list
        구간 레이블 리스트
    new_col_name : str
        생성할 새 컬럼명
    right : bool, default=False
        구간이 오른쪽 경계를 포함하는지 여부
    ordered : bool, default=True
        순서형 범주형으로 만들지 여부 (True: ordinal, False: nominal)
    
    Returns
    -------
    df : pd.DataFrame
        구간화된 컬럼이 추가된 데이터프레임
    dist_df : pd.DataFrame
        구간별 분포 통계 (count, pct)
    """
    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors="coerce")

    df[new_col_name] = pd.cut(
        df[col],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=right,
        ordered=ordered,
    )

    dist = df[new_col_name].value_counts().sort_index()
    total = dist.sum()
    pct = (dist / total * 100).round(2)

    dist_df = pd.DataFrame({"count": dist, "pct": pct})
    return df, dist_df


def create_price_binned_feature(df, 
                                 col="신차가격(선택옵션포함)",
                                 new_col_name="신차가격_구간",
                                 right=False):
    """
    신차가격을 구간으로 나누는 함수
    
    주의: apply_optimal_dtypes를 사용한 경우 신차가격이 만원 단위로 변환되므로,
    이 함수의 bins도 만원 단위로 조정해야 할 수 있습니다.
    
    Parameters
    ----------
    df : pd.DataFrame
        입력 데이터프레임
    col : str, default="신차가격(선택옵션포함)"
        구간화할 컬럼명
    new_col_name : str, default="신차가격_구간"
        생성할 새 컬럼명
    right : bool, default=False
        구간이 오른쪽 경계를 포함하는지 여부
    
    Returns
    -------
    df : pd.DataFrame
        구간화된 컬럼이 추가된 데이터프레임
    dist_df : pd.DataFrame
        구간별 분포 통계 (count, pct)
    """
    # 신차가격 구간 설정 (만원 단위)
    # apply_optimal_dtypes 사용 시 신차가격이 만원 단위로 변환되므로 bins도 만원 단위로 설정
    PRICE_BINS = [0, 2000, 3000, 4000, 5000, 7000, 10000, np.inf]
    
    PRICE_LABELS = [
        "2천만원 미만",
        "2천만원 이상 ~ 3천만원 미만",
        "3천만원 이상 ~ 4천만원 미만",
        "4천만원 이상 ~ 5천만원 미만",
        "5천만원 이상 ~ 7천만원 미만",
        "7천만원 이상 ~ 1억원 미만",
        "1억원 이상"
    ]
    
    # 신차가격 구간은 순서형(ordinal)
    return create_binned_feature(
        df, col, PRICE_BINS, PRICE_LABELS, new_col_name, right, ordered=True
    )


def create_engine_binned_feature(df,
                                 disp_col="배기량",
                                 new_col_name="배기량_구간"):
    """
    배기량을 구간으로 나누는 함수
    
    Parameters
    ----------
    df : pd.DataFrame
        입력 데이터프레임
    disp_col : str, default="배기량"
        배기량 컬럼명
    new_col_name : str, default="배기량_구간"
        생성할 새 컬럼명
    
    Returns
    -------
    df : pd.DataFrame
        구간화된 컬럼이 추가된 데이터프레임
    dist_df : pd.DataFrame
        구간별 분포 통계 (count, pct)
    """
    # 배기량 구간 설정 (0~499cc 포함)
    ENGINE_BINS = [0, 500, 1000, 1500, 2000, 2500, 3000, np.inf]
    ENGINE_LABELS = [
        "0cc",
        "500~999cc",
        "1000~1499cc",
        "1500~1999cc",
        "2000~2499cc",
        "2500~2999cc",
        "3000cc~"
    ]
    
    return create_binned_feature(
        df, disp_col, ENGINE_BINS, ENGINE_LABELS, new_col_name, right=False, ordered=False
    )


def add_generation_features_optimal(df: pd.DataFrame) -> pd.DataFrame:
    """
    세대 관련 피처 생성
    gen_index와 gen_max는 모델의 고유 속성이므로 Train/Test 분리 전에 계산해도 문제없음
    """
    df = df.copy()
    
    df["gen_norm"] = np.where(
        df["gen_max"] == 1,
        1.0,
        (df["gen_index"] - 1) / (df["gen_max"] - 1)
    )
    
    # if_planned 컬럼이 있으면 사용하고, 없으면 0으로 설정
    if "if_planned" in df.columns:
        df["has_planned_next_gen"] = df["if_planned"].fillna(0).astype(int)
    else:
        df["has_planned_next_gen"] = 0
    
    # 불필요한 컬럼 제거
    remove_cols = [
        "gen_index", "gen_max", "status_current", "if_planned",
        "if_planned_year", "generation_label", "platform_codes",
        "gen_start_ym", "gen_end_ym", "detailed_status",
    ]
    exist_cols = [c for c in remove_cols if c in df.columns]
    if exist_cols:
        df = df.drop(columns=exist_cols)
    
    return df

def remove_outliers_by_new_car_price(
    df,
    new_car_price_col="신차가격(선택옵션포함)",
    sale_price_col="판매신고가"):

    df = df.copy()
    
    # apply_optimal_dtypes 적용 후 둘 다 만원 단위이므로 직접 비교
    # 이상치 마스크: 판매신고가(만원)가 신차가격(만원)보다 높은 경우
    outlier_mask = df[sale_price_col] > df[new_car_price_col]
    
    # 이상치 인덱스
    outlier_idx = df[outlier_mask].index.tolist()
    
    # 이상치 제거
    df_clean = df[~outlier_mask].copy()
    
    print("=" * 50)
    print("신차가격 대비 판매신고가 이상치 제거")
    print("=" * 50)
    print("전체 행 수:", len(df))
    print("제거된 outlier 수:", len(outlier_idx))
    print("제거 후 행 수:", len(df_clean))
    print(f"제거 비율: {len(outlier_idx)/len(df)*100:.2f}%")
    print("=" * 50)
    
    return df_clean

def remove_outliers_by_band(
    df,
    ad_price_col="최초 광고가",
    sale_price_col="판매신고가",
    threshold=0.40):
    """광고가와 판매신고가 차이 비율이 threshold보다 큰 경우 이상치로 제거"""
    df = df.copy()
    
    # 광고가 차이 비율 계산
    price_diff_ratio = (
        (df[ad_price_col] - df[sale_price_col]).abs()
        / df[ad_price_col]
    )
    
    # 이상치 마스크: 비율이 threshold보다 큰 경우
    outlier_mask = price_diff_ratio > threshold
    
    # 이상치 인덱스
    outlier_idx = df[outlier_mask].index.tolist()
    
    # 이상치 제거
    df_clean = df[~outlier_mask].copy()
    
    print("전체 행 수:", len(df))
    print("제거된 outlier 수:", len(outlier_idx))
    print("제거 후 행 수:", len(df_clean))
    print(f"제거 비율: {len(outlier_idx)/len(df)*100:.2f}%")
    
    return df_clean


# ============================================================
# Train/Test 분리 후 사용하는 통계량 계산 및 매핑 함수들
# ============================================================

def calculate_판매속도지수_stats(df_train, group_col='등급코드', 
                                ad_date_col='최초 광고 등록일', 
                                sale_date_col='판매신고일', 
                                multiplier=100):
    """
    Train set에서 판매속도지수 통계량만 계산
    
    Parameters
    ----------
    df_train : pd.DataFrame
        Train 데이터프레임
    group_col : str, default='대표차종명'
        그룹화할 컬럼명 (등급코드, 대표차종명 등)
    ad_date_col : str, default='최초 광고 등록일'
        광고 등록일 컬럼명
    sale_date_col : str, default='판매신고일'
        판매신고일 컬럼명
    multiplier : int, default=100
        판매속도지수 계산 시 곱하는 배수
    
    Returns
    -------
    pd.Series
        {group_value: 판매속도지수} 형태의 Series
    """
    df = df_train.copy()
    
    # DOM 계산
    if "DOM" not in df.columns:
        df["DOM"] = (df[sale_date_col] - df[ad_date_col]).dt.days
    
    # 그룹별 DOM 평균 계산
    dom_by_group = df.groupby(group_col, observed=True)['DOM'].mean()
    
    # 판매속도지수 계산
    판매속도지수 = np.where(
        (dom_by_group > 0) & (dom_by_group.notna()),
        1 / dom_by_group * multiplier,
        np.nan
    )
    
    return pd.Series(판매속도지수, index=dom_by_group.index)


def apply_판매속도지수(df, stats_dict, group_col='등급코드'):
    """
    계산된 판매속도지수 통계량을 데이터에 매핑
    
    Parameters
    ----------
    df : pd.DataFrame
        매핑할 데이터프레임 (train 또는 test)
    stats_dict : pd.Series or dict
        calculate_판매속도지수_stats로 계산된 통계량
    group_col : str, default='대표차종명'
        그룹화할 컬럼명
    
    Returns
    -------
    pd.DataFrame
        판매속도지수가 추가된 데이터프레임
    """
    df = df.copy()
    col_name = f'판매속도지수_{group_col}'
    df[col_name] = df[group_col].map(stats_dict).round(2)
    return df


def calculate_빈도_stats(df_train, group_col='등급코드'):
    """
    Train set에서 빈도 통계량만 계산
    
    Parameters
    ----------
    df_train : pd.DataFrame
        Train 데이터프레임
    group_col : str, default='대표차종명'
        그룹화할 컬럼명 (등급코드, 대표차종명 등)
    
    Returns
    -------
    pd.Series
        {group_value: 빈도} 형태의 Series
    """
    freq_by_group = df_train.groupby(group_col, observed=True).size()
    return freq_by_group


def apply_빈도(df, stats_dict, group_col='등급코드'):
    """
    계산된 빈도 통계량을 데이터에 매핑
    
    Parameters
    ----------
    df : pd.DataFrame
        매핑할 데이터프레임 (train 또는 test)
    stats_dict : pd.Series or dict
        calculate_빈도_stats로 계산된 통계량
    group_col : str, default='대표차종명'
        그룹화할 컬럼명
    
    Returns
    -------
    pd.DataFrame
        빈도가 추가된 데이터프레임
    """
    df = df.copy()
    col_name = f'빈도_{group_col}'
    df[col_name] = df[group_col].map(stats_dict)
    return df


def create_판매속도지수_binned_feature(df, 
                                       col='판매속도지수_등급코드',
                                       new_col_name='판매속도지수_등급코드_구간',
                                       n_bins=5,
                                       method='quantile',
                                       bins=None):
    """
    판매속도지수를 구간으로 나누는 함수 (KModes용)
    
    Parameters
    ----------
    df : pd.DataFrame
        입력 데이터프레임
    col : str, default='판매속도지수_등급코드'
        구간화할 컬럼명
    new_col_name : str, default='판매속도지수_등급코드_구간'
        생성할 새 컬럼명
    n_bins : int, default=5
        구간 개수
    method : str, default='quantile'
        구간 나누는 방법 
        - 'quantile': 분위수 기반 (각 구간에 데이터 균등 분포)
        - 'uniform': 균등 구간 (값 범위를 균등하게 나눔)
        - 'kmeans': K-means 클러스터링 기반 (분포의 자연스러운 경계 찾기)
    bins : array-like, optional
        구간 경계값 (지정하면 이 값 사용, None이면 계산)
    
    Returns
    -------
    df : pd.DataFrame
        구간화된 컬럼이 추가된 데이터프레임
    dist_df : pd.DataFrame
        구간별 분포 통계 (count, pct)
    bins : array
        사용된 구간 경계값 (Test set에 적용 시 사용)
    """
    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # bins가 지정되지 않았으면 계산
    if bins is None:
        # NaN 제외한 값으로 구간 경계 계산
        valid_values = df[col].dropna()
        
        if len(valid_values) == 0:
            df[new_col_name] = pd.NA
            return df, pd.DataFrame(), None
        
        if method == 'quantile':
            # 분위수 기반 구간
            bins = np.quantile(valid_values, np.linspace(0, 1, n_bins + 1))
            bins = np.unique(bins)  # 중복 제거
            if len(bins) < 2:
                bins = [valid_values.min(), valid_values.max()]
        elif method == 'kmeans':
            # K-means 클러스터링 기반 구간 (분포의 자연스러운 경계 찾기)
            from sklearn.cluster import KMeans
            # 1차원 데이터를 2차원으로 변환 (KMeans는 2D 이상 필요)
            values_2d = valid_values.values.reshape(-1, 1)
            kmeans = KMeans(n_clusters=n_bins, random_state=42, n_init=10)
            kmeans.fit(values_2d)
            # 클러스터 중심점을 정렬
            centers = np.sort(kmeans.cluster_centers_.flatten())
            
            # 클러스터 중심점 사이의 중간점을 구간 경계로 사용
            # n_bins=5이면 4개의 경계점이 필요 (5개 구간을 만들기 위해)
            if len(centers) > 1:
                # 인접한 중심점 사이의 중간점 계산
                boundaries = []
                for i in range(len(centers) - 1):
                    mid_point = (centers[i] + centers[i + 1]) / 2
                    boundaries.append(mid_point)
                # 최소값과 첫 번째 경계, 마지막 경계와 최대값 포함
                bins = np.concatenate([[valid_values.min()], boundaries, [valid_values.max()]])
            else:
                # 클러스터가 1개인 경우 (거의 발생하지 않음)
                bins = [valid_values.min(), valid_values.max()]
            bins = np.unique(bins)  # 중복 제거
        else:  # uniform
            # 균등 구간
            bins = np.linspace(valid_values.min(), valid_values.max(), n_bins + 1)
    
    # 레이블 생성
    labels = [f'구간{i+1}' for i in range(len(bins) - 1)]
    
    # 구간화
    df[new_col_name] = pd.cut(
        df[col],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
        ordered=False,  # KModes용 명목형
    )
    
    # 통계 계산
    dist = df[new_col_name].value_counts().sort_index()
    total = dist.sum()
    pct = (dist / total * 100).round(2)
    dist_df = pd.DataFrame({"count": dist, "pct": pct})
    
    return df, dist_df, bins


def create_빈도_binned_feature(df,
                               col='빈도_등급코드',
                               new_col_name='빈도_등급코드_구간',
                               n_bins=5,
                               method='quantile',
                               bins=None):
    """
    빈도를 구간으로 나누는 함수 (KModes용)
    
    Parameters
    ----------
    df : pd.DataFrame
        입력 데이터프레임
    col : str, default='빈도_등급코드'
        구간화할 컬럼명
    new_col_name : str, default='빈도_등급코드_구간'
        생성할 새 컬럼명
    n_bins : int, default=5
        구간 개수
    method : str, default='quantile'
        구간 나누는 방법 
        - 'quantile': 분위수 기반 (각 구간에 데이터 균등 분포)
        - 'uniform': 균등 구간 (값 범위를 균등하게 나눔)
        - 'kmeans': K-means 클러스터링 기반 (분포의 자연스러운 경계 찾기)
    bins : array-like, optional
        구간 경계값 (지정하면 이 값 사용, None이면 계산)
    
    Returns
    -------
    df : pd.DataFrame
        구간화된 컬럼이 추가된 데이터프레임
    dist_df : pd.DataFrame
        구간별 분포 통계 (count, pct)
    bins : array
        사용된 구간 경계값 (Test set에 적용 시 사용)
    """
    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # bins가 지정되지 않았으면 계산
    if bins is None:
        # NaN 제외한 값으로 구간 경계 계산
        valid_values = df[col].dropna()
        
        if len(valid_values) == 0:
            df[new_col_name] = pd.NA
            return df, pd.DataFrame(), None
        
        if method == 'quantile':
            # 분위수 기반 구간
            bins = np.quantile(valid_values, np.linspace(0, 1, n_bins + 1))
            bins = np.unique(bins)  # 중복 제거
            if len(bins) < 2:
                bins = [valid_values.min(), valid_values.max()]
        elif method == 'kmeans':
            # K-means 클러스터링 기반 구간 (분포의 자연스러운 경계 찾기)
            from sklearn.cluster import KMeans
            # 1차원 데이터를 2차원으로 변환 (KMeans는 2D 이상 필요)
            values_2d = valid_values.values.reshape(-1, 1)
            kmeans = KMeans(n_clusters=n_bins, random_state=42, n_init=10)
            kmeans.fit(values_2d)
            # 클러스터 중심점을 정렬
            centers = np.sort(kmeans.cluster_centers_.flatten())
            
            # 클러스터 중심점 사이의 중간점을 구간 경계로 사용
            # n_bins=5이면 4개의 경계점이 필요 (5개 구간을 만들기 위해)
            if len(centers) > 1:
                # 인접한 중심점 사이의 중간점 계산
                boundaries = []
                for i in range(len(centers) - 1):
                    mid_point = (centers[i] + centers[i + 1]) / 2
                    boundaries.append(mid_point)
                # 최소값과 첫 번째 경계, 마지막 경계와 최대값 포함
                bins = np.concatenate([[valid_values.min()], boundaries, [valid_values.max()]])
            else:
                # 클러스터가 1개인 경우 (거의 발생하지 않음)
                bins = [valid_values.min(), valid_values.max()]
            bins = np.unique(bins)  # 중복 제거
        else:  # uniform
            # 균등 구간
            bins = np.linspace(valid_values.min(), valid_values.max(), n_bins + 1)
    
    # 레이블 생성
    labels = [f'구간{i+1}' for i in range(len(bins) - 1)]
    
    # 구간화
    df[new_col_name] = pd.cut(
        df[col],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
        ordered=False,  # KModes용 명목형
    )
    
    # 통계 계산
    dist = df[new_col_name].value_counts().sort_index()
    total = dist.sum()
    pct = (dist / total * 100).round(2)
    dist_df = pd.DataFrame({"count": dist, "pct": pct})
    
    return df, dist_df, bins



# ============================================================
# 국산차/외제차 구분 함수
# ============================================================

def add_국산외제차_구분(df, 제조사코드_col='제조사코드', new_col_name='국산외제차'):
    """
    제조사코드를 기반으로 국산차/외제차 구분 컬럼 추가
    - 국산차: 0
    - 외제차: 1
    
    Parameters
    ----------
    df : pd.DataFrame
        입력 데이터프레임
    제조사코드_col : str, default='제조사코드'
        제조사코드 컬럼명
    new_col_name : str, default='국산외제차'
        생성할 컬럼명
    
    Returns
    -------
    pd.DataFrame
        국산외제차 컬럼이 추가된 데이터프레임 (국산차=0, 외제차=1)
    """
    df = df.copy()
    
    # 국산차 제조사코드
    # 1: 현대, 2: 기아, 3: 쉐보레, 4: 쌍용, 5: 르노(르노삼성), 7: 제네시스
    국산차_코드 = {1, 2, 3, 4, 5, 7}
    
    # 제조사코드가 없으면 에러
    if 제조사코드_col not in df.columns:
        raise ValueError(f"'{제조사코드_col}' 컬럼이 필요합니다.")
    
    # 제조사코드를 숫자로 변환
    제조사코드_num = pd.to_numeric(df[제조사코드_col], errors='coerce')
    
    # 국산차면 0, 외제차면 1
    # 국산차 코드에 있으면 False → 0, 없으면 True → 1
    df[new_col_name] = (~제조사코드_num.isin(국산차_코드)).astype(int)
    
    return df


# ============================================================
# 추가 피처 생성 함수들
# ============================================================

def assign_interest_rate_by_sale_date(
    df,
    sale_col="판매신고일",
    new_col="판매기준금리"
):
    """
    판매신고일 기준으로 금리를 매핑하는 함수.
    금리 구간:
    - 2024-12-31 ~ 2025-01-24 : 3.00%
    - 2025-01-25 ~ 2025-02-25 : 2.75%
    - 2025-02-26 ~ 현재까지   : 2.50%
    
    Parameters
    ----------
    df : pd.DataFrame
        입력 데이터프레임
    sale_col : str, default='판매신고일'
        판매신고일 컬럼명
    new_col : str, default='판매기준금리'
        생성할 컬럼명
    
    Returns
    -------
    pd.DataFrame
        판매기준금리 컬럼이 추가된 데이터프레임
    """
    df = df.copy()
    # 날짜형 변환
    df[sale_col] = pd.to_datetime(df[sale_col], errors="coerce")
    # 초기값
    df[new_col] = np.nan
    # 조건 정의
    cond1 = (df[sale_col] >= pd.Timestamp("2024-12-31")) & (df[sale_col] <= pd.Timestamp("2025-01-24"))
    cond2 = (df[sale_col] >= pd.Timestamp("2025-01-25")) & (df[sale_col] <= pd.Timestamp("2025-02-25"))
    cond3 = (df[sale_col] >= pd.Timestamp("2025-02-26"))
    # 매핑
    df.loc[cond1, new_col] = 3.00
    df.loc[cond2, new_col] = 2.75
    df.loc[cond3, new_col] = 2.50
    return df


def add_accident_rank_full(
    df,
    col_acc_flag="사고유무",
    col_replace="교환부위",
    col_panel="판금부위",
    new_col_severity="사고_심각도"
):
    """
    사진 기준(외판부위·주요골격) 랭크 정보를 바탕으로 사고 랭크 컬럼 생성
    
    Parameters
    ----------
    df : pd.DataFrame
        입력 데이터프레임
    col_acc_flag : str, default='사고유무'
        사고유무 컬럼명
    col_replace : str, default='교환부위'
        교환부위 컬럼명
    col_panel : str, default='판금부위'
        판금부위 컬럼명
    new_col_severity : str, default='사고_심각도'
        생성할 컬럼명
    
    Returns
    -------
    pd.DataFrame
        사고_심각도 컬럼이 추가된 데이터프레임
    """
    df = df.copy()
    
    # --- 키워드 사전 정의 (사진 기반) ---
    rank_A = [
        "프론트패널", "크로스멤버", "인사이드패널",
        "트렁크플로어", "리어패널"
    ]
    rank_B = [
        "사이드멤버", "휠하우스",
        "필러패널", "필러 패널", "A필러", "B필러", "C필러",
        "패키지트레이"
    ]
    rank_C = [
        "대쉬패널", "플로어패널"
    ]
    rank_outer_2 = [
        "쿼터패널", "쿼터 패널",
        "루프패널", "루프 패널",
        "사이드실패널", "사이드 실패널"
    ]
    rank_outer_1 = [
        "후드", "본넷",
        "프론트휀더", "프론트 휀더",
        "도어",
        "트렁크리드", "트렁크 리드",
        "라디에이터서포트", "라디에이터 서포트"
    ]
    
    # 우선순위 맵핑
    priority = {
        "A": 3,
        "B": 4,
        "C": 5,
        "외판2": 2,
        "외판1": 1,
        "NO_ACC": 0
    }
    
    def classify_row(row):
        acc_flag = str(row[col_acc_flag]).upper().strip()
        parts = []
        for c in [col_replace, col_panel]:
            val = row.get(c)
            if isinstance(val, str):
                parts.append(val)
        text = " ".join(parts)
        
        # 무사고
        if acc_flag == "N" and text.strip() == "":
            return priority["NO_ACC"]
        
        # A 랭크
        severity = 0
        for kw in rank_A:
            if kw in text:
                severity += priority["A"]
        # B 랭크
        for kw in rank_B:
            if kw in text:
                severity += priority["B"]
        # C 랭크
        for kw in rank_C:
            if kw in text:
                severity += priority["C"]
        # 외판 2랭크
        for kw in rank_outer_2:
            if kw in text:
                severity += priority["외판2"]
        # 외판 1랭크
        for kw in rank_outer_1:
            if kw in text:
                severity += priority["외판1"]
        
        # 사고유무=Y인데 키워드에 안 걸린 경우 → 가장 경미한 외판 1
        if severity == 0:
            severity = priority["외판1"]
        
        return severity
    
    # 적용
    df[new_col_severity] = df.apply(classify_row, axis=1)
    
    return df


def add_car_age_features(
    df: pd.DataFrame,
    prod_ym_col: str = "연월식",
    sale_date_col: str = "판매신고일",
    out_year_col: str = "차나이_년",
    out_month_col: str = "차나이_월",
    max_year_cap: int | None = 20
) -> pd.DataFrame:
    """
    연월식(YYYYMM) → 판매신고일 기준 차량 나이 계산
    - 차나이_월 : 월 단위 정수
    - 차나이_년 : 월/12 후 반올림 (연 단위)
    
    Parameters
    ----------
    df : pd.DataFrame
        입력 데이터프레임
    prod_ym_col : str, default='연월식'
        연월식 컬럼명 (Period 타입 또는 YYYYMM 형식)
    sale_date_col : str, default='판매신고일'
        판매신고일 컬럼명
    out_year_col : str, default='차나이_년'
        생성할 연 단위 나이 컬럼명
    out_month_col : str, default='차나이_월'
        생성할 월 단위 나이 컬럼명
    max_year_cap : int or None, default=20
        최대 나이 제한 (년 단위), None이면 제한 없음
    
    Returns
    -------
    pd.DataFrame
        차나이_년, 차나이_월 컬럼이 추가된 데이터프레임
    """
    df_out = df.copy()

    # 1️⃣ 연월식 → datetime (해당 월의 1일)
    # Period 타입인 경우 문자열로 변환
    if df_out[prod_ym_col].dtype.name == 'period[M]':
        prod_ym_str = df_out[prod_ym_col].astype(str).str.replace('-', '').str.zfill(6)
    else:
        prod_ym_str = df_out[prod_ym_col].astype(str).str.zfill(6)
    
    prod_date = pd.to_datetime(
        prod_ym_str + "01",
        format="%Y%m%d",
        errors="coerce"
    )

    # 2️⃣ 판매신고일
    sale_date = pd.to_datetime(df_out[sale_date_col], errors="coerce")

    # 3️⃣ 개월 수 차이
    gap_months = (
        (sale_date.dt.year - prod_date.dt.year) * 12
        + (sale_date.dt.month - prod_date.dt.month)
    ).astype("Int16")

    # 4️⃣ 연 단위 변환 (반올림)
    gap_years = (gap_months / 12).round(0)

    if max_year_cap is not None:
        gap_years = gap_years.clip(upper=max_year_cap)

    # 5️⃣ 컬럼 저장
    df_out[out_month_col] = gap_months.astype("Int16")
    df_out[out_year_col] = gap_years.astype("Int8")

    return df_out


def add_mileage_features(df_car: pd.DataFrame,
                       mileage_col: str = "주행거리",
                       prod_ym_col: str = "연월식",
                       sale_date_col: str = "판매신고일",
                       month_smoothing: int = 1,
                       avg_km_per_year: int = 20_000) -> pd.DataFrame:
    """
    주행거리(사용량) 핵심 파생피처 생성 함수 (최소 세트)
    생성 컬럼:
    - 차나이_월 (이미 있으면 재계산하지 않음)
    - km_per_month        : 연식 대비 월 사용량
    - mileage_ratio       : 평균(연 2만 km) 대비 사용 비율
    
    Parameters
    ----------
    df_car : pd.DataFrame
        입력 데이터프레임
    mileage_col : str, default='주행거리'
        주행거리 컬럼명
    prod_ym_col : str, default='연월식'
        연월식 컬럼명
    sale_date_col : str, default='판매신고일'
        판매신고일 컬럼명
    month_smoothing : int, default=1
        월 단위 스무딩 파라미터
    avg_km_per_year : int, default=20000
        연평균 주행거리 (km)
    
    Returns
    -------
    pd.DataFrame
        주행거리 관련 피처가 추가된 데이터프레임
    """
    df = df_car.copy()
    
    # 차나이_월이 없으면 계산
    if "차나이_월" not in df.columns:
        df = add_car_age_features(df, prod_ym_col=prod_ym_col, sale_date_col=sale_date_col)
    
    # Period 타입인 경우 처리
    if df[prod_ym_col].dtype.name == 'period[M]':
        prod_ym_str = df[prod_ym_col].astype(str).str.replace('-', '').str.zfill(6)
    else:
        prod_ym_str = df[prod_ym_col].astype("Int64").astype(str).str.zfill(6)
    
    prod_dt = pd.to_datetime(
        prod_ym_str,
        format="%Y%m",
        errors="coerce"
    )
    sale_dt = pd.to_datetime(df[sale_date_col], errors="coerce")
    
    age_months = (
        (sale_dt.dt.year - prod_dt.dt.year) * 12 +
        (sale_dt.dt.month - prod_dt.dt.month)
    )
    age_months = age_months.clip(lower=1).astype("Int32")
    df["차나이_월"] = age_months
    
    mileage = pd.to_numeric(df[mileage_col], errors="coerce").astype("Float32")
    df[mileage_col] = mileage
    
    denom_month = df["차나이_월"].astype("float32") + float(month_smoothing)
    df["km_per_month"] = (mileage / denom_month).round(2).astype("float32")
    df["log_km_per_month"] = np.log1p(df["km_per_month"])
    
    effective_years = (df["차나이_월"].astype("float32") / 12.0) + (month_smoothing / 12.0)
    expected_km = avg_km_per_year * effective_years
    df["mileage_ratio"] = (mileage / (expected_km + 1.0)).astype("float32")
    
    return df


def remove_nan_rows_in_binned_features(df: pd.DataFrame,
                                       check_cols: list = None) -> pd.DataFrame:
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
        print("  ✓ 확인할 구간 컬럼이 없습니다.")
        return df
    
    # NaN 확인
    before_len = len(df)
    nan_mask = df[check_cols].isna().any(axis=1)
    nan_count = nan_mask.sum()
    
    if nan_count > 0:
        # 각 컬럼별 NaN 개수 출력
        print(f"  ⚠ 구간 피처 NaN 발견:")
        for col in check_cols:
            col_nan = df[col].isna().sum()
            if col_nan > 0:
                print(f"    - {col}: {col_nan}건")
        
        # NaN이 있는 행 제거
        df = df[~nan_mask].copy()
        removed_count = before_len - len(df)
        
        print(f"  → NaN 행 제거: {removed_count}건 ({removed_count/before_len*100:.2f}%)")
        print(f"  → 제거 후: {len(df):,}건")
    else:
        print("  ✓ 구간 피처 NaN 없음")
    
    return df


def add_temporal_features_normalized(df: pd.DataFrame,
                                    sale_date_col: str = "판매신고일") -> pd.DataFrame:
    """
    시간 관련 피처 생성 (정규화된 순환 피처)
    생성 컬럼:
    - sale_month, sale_week, sale_day
    - month_sin, month_cos
    - week_sin, week_cos
    - day_sin, day_cos
    - day_normalized
    
    Parameters
    ----------
    df : pd.DataFrame
        입력 데이터프레임
    sale_date_col : str, default='판매신고일'
        판매신고일 컬럼명
    
    Returns
    -------
    pd.DataFrame
        시간 관련 피처가 추가된 데이터프레임
    """
    df_out = df.copy()
    
    sale_date = pd.to_datetime(df_out[sale_date_col], errors="coerce")
    
    # 월 피처
    df_out["sale_month"] = sale_date.dt.month
    df_out["month_sin"] = np.sin(2 * np.pi * df_out["sale_month"] / 12).round(2)
    df_out["month_cos"] = np.cos(2 * np.pi * df_out["sale_month"] / 12).round(2)
    
    # 주 피처
    df_out["sale_week"] = sale_date.dt.isocalendar().week
    df_out["week_sin"] = np.sin(2 * np.pi * df_out["sale_week"] / 52).round(2)
    df_out["week_cos"] = np.cos(2 * np.pi * df_out["sale_week"] / 52).round(2)
    
    # # 일 피처
    # df_out["sale_day"] = sale_date.dt.day
    # days_in_month = sale_date.dt.days_in_month  # 각 월의 실제 일수 (28, 29, 30, 31)
    # df_out["day_normalized"] = df_out["sale_day"] / days_in_month  # 0~1 정규화 (각 월의 실제 일수로)
    # df_out["day_sin"] = np.sin(2 * np.pi * df_out["sale_day"] / days_in_month)
    # df_out["day_cos"] = np.cos(2 * np.pi * df_out["sale_day"] / days_in_month)
    
    return df_out
