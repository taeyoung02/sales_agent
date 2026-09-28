"""
숫자를 한국어로 변환하는 유틸리티 함수
TTS에서 자연스럽게 읽히도록 숫자를 한국어로 변환합니다.
"""

import re


def number_to_korean(num: int) -> str:
    """
    숫자를 한국어로 변환

    Args:
        num: 변환할 숫자

    Returns:
        한국어로 변환된 숫자 문자열

    Examples:
        number_to_korean(49000000) -> "사천구백만"
        number_to_korean(50000) -> "오만"
        number_to_korean(12345) -> "만이천삼백사십오"
    """
    if num == 0:
        return "영"

    # 한자리 숫자 변환
    units = ["", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"]
    # 십, 백, 천 단위
    tens = ["", "십", "백", "천"]
    # 만, 억, 조 단위
    large_units = ["", "만", "억", "조"]

    # 음수 처리
    is_negative = num < 0
    num = abs(num)

    # 0 처리
    if num == 0:
        return "영"

    # 숫자를 문자열로 변환하여 역순으로 처리
    num_str = str(num)
    result_parts = []

    # 4자리씩 묶어서 처리 (만, 억, 조 단위)
    num_groups = []
    while num > 0:
        num_groups.append(num % 10000)
        num //= 10000

    # 각 그룹을 한국어로 변환
    for group_idx, group_num in enumerate(num_groups):
        if group_num == 0:
            continue

        group_str = ""
        group_num_str = str(group_num).zfill(4)

        # 천, 백, 십, 일 자리 처리
        for digit_idx, digit in enumerate(group_num_str):
            digit_int = int(digit)
            if digit_int == 0:
                continue

            # 천의 자리는 "일천"이 아닌 "천"으로 표기
            if digit_idx == 0 and digit_int == 1:
                group_str += "천"
            # 백의 자리는 "일백"이 아닌 "백"으로 표기
            elif digit_idx == 1 and digit_int == 1:
                group_str += "백"
            # 십의 자리는 "일십"이 아닌 "십"으로 표기
            elif digit_idx == 2 and digit_int == 1:
                group_str += "십"
            else:
                group_str += units[digit_int] + tens[3 - digit_idx]

        # 만, 억, 조 단위 추가
        if group_idx > 0 and group_str:
            group_str += large_units[group_idx]
        # 첫 번째 그룹(일의 자리 그룹)은 단위 없음

        result_parts.insert(0, group_str)

    result = "".join(result_parts)

    # "일만", "일천", "일백", "일십"을 "만", "천", "백", "십"으로 변환
    # 단, 맨 앞에 오는 경우만 (예: "일만이천" -> "만이천", "일천" -> "천")
    # 또는 단독으로 오는 경우 (예: "일천" -> "천")
    if result == "일만":
        result = "만"
    elif result == "일천":
        result = "천"
    elif result == "일백":
        result = "백"
    elif result == "일십":
        result = "십"
    elif result.startswith("일만"):
        result = "만" + result[2:]
    elif result.startswith("일천"):
        result = "천" + result[2:]
    elif result.startswith("일백"):
        result = "백" + result[2:]
    elif result.startswith("일십"):
        result = "십" + result[2:]

    # 음수 처리
    if is_negative:
        result = "마이너스 " + result

    return result


def format_price_korean(price: int) -> str:
    """
    가격을 한국어로 변환 (원 단위 포함)

    Args:
        price: 가격 (원)

    Returns:
        한국어로 변환된 가격 문자열

    Examples:
        format_price_korean(49000000) -> "약 사천구백만원"
        format_price_korean(50000000) -> "약 오천만원"
        format_price_korean(12345678) -> "약 천이백삼십사만오천육백칠십팔원"
    """
    if price < 10000:
        return f"{number_to_korean(price)}원"
    elif price < 100000000:  # 1억 미만
        return f"약 {number_to_korean(price)}원"
    else:  # 1억 이상
        # 억 단위와 나머지로 분리
        eok = price // 100000000
        remainder = price % 100000000

        if remainder == 0:
            return f"약 {number_to_korean(eok)}억원"
        else:
            return f"약 {number_to_korean(eok)}억 {number_to_korean(remainder)}원"


def format_mileage_korean(mileage: int) -> str:
    """
    주행거리를 한국어로 변환 (km 단위 포함)

    Args:
        mileage: 주행거리 (km)

    Returns:
        한국어로 변환된 주행거리 문자열

    Examples:
        format_mileage_korean(50000) -> "오만 킬로미터"
        format_mileage_korean(12345) -> "만이천삼백사십오 킬로미터"
        format_mileage_korean(1000) -> "천 킬로미터"
    """
    korean = number_to_korean(mileage)
    # "일천"을 "천"으로 변환
    if korean == "일천":
        korean = "천"
    elif korean.startswith("일천"):
        korean = "천" + korean[2:]

    return f"{korean} 킬로미터"


def convert_numbers_in_text(text: str) -> str:
    """
    텍스트 내의 숫자를 한국어로 변환
    가격(원), 주행거리(km) 패턴을 자동으로 감지하여 적절한 형식으로 변환

    Args:
        text: 변환할 텍스트

    Returns:
        숫자가 한국어로 변환된 텍스트

    Examples:
        convert_numbers_in_text("가격은 49000000원입니다") -> "가격은 약 사천구백만원입니다"
        convert_numbers_in_text("주행거리는 50000km입니다") -> "주행거리는 오만 킬로미터입니다"
    """
    # 가격 패턴: 숫자 + "원" (콤마 포함 가능, 큰 숫자도 처리)
    # 콤마가 있는 경우: \d{1,3}(?:,\d{3})+
    # 콤마가 없는 경우: \d{4,}
    price_pattern = r"(\d{1,3}(?:,\d{3})+|\d{4,})\s*원"

    def replace_price(match):
        price_str = match.group(1).replace(",", "")
        try:
            price = int(price_str)
            return format_price_korean(price)
        except ValueError:
            return match.group(0)

    text = re.sub(price_pattern, replace_price, text)

    # 주행거리 패턴: 숫자 + "km" 또는 "킬로미터" (콤마 포함 가능, 큰 숫자도 처리)
    mileage_pattern = r"(\d{1,3}(?:,\d{3})+|\d{4,})\s*(?:km|킬로미터)"

    def replace_mileage(match):
        mileage_str = match.group(1).replace(",", "")
        try:
            mileage = int(mileage_str)
            return format_mileage_korean(mileage)
        except ValueError:
            return match.group(0)

    text = re.sub(mileage_pattern, replace_mileage, text)

    # 일반 숫자 패턴 (가격, 주행거리가 아닌 경우)
    # 큰 숫자만 변환 (1000 이상), 이미 변환된 부분은 제외
    # 가격/주행거리 패턴과 겹치지 않도록 주의
    general_number_pattern = r"(?<!\d)(\d{4,})(?!\s*(?:원|km|킬로미터))"

    def replace_general_number(match):
        num_str = match.group(1).replace(",", "")
        try:
            num = int(num_str)
            # 1000 이상인 경우만 변환
            if num >= 1000:
                korean = number_to_korean(num)
                # "일천"을 "천"으로 변환
                if korean == "일천":
                    korean = "천"
                elif korean.startswith("일천"):
                    korean = "천" + korean[2:]
                return korean
            else:
                return match.group(0)
        except ValueError:
            return match.group(0)

    text = re.sub(general_number_pattern, replace_general_number, text)

    return text
