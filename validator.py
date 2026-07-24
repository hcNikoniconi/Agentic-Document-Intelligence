import os
import re
from datetime import date, datetime


# =========================
# 可配置规则
# =========================

MIN_AGE = 18

PASSPORT_EXPIRY_WARNING_DAYS = 365

CHECK_DATE = date.today()

NORMAL_PASSPORT_TYPES = {
    "PASSPORT",
    "ORDINARY PASSPORT",
    "REGULAR PASSPORT",
    "NORMAL PASSPORT",
}

DEFAULT_DOC_TYPE_ORDER = [
    "passport",
    "application_form",
    "transcript",
    "diploma_certificate",
    "english_language",
]

# 暂时不对 diploma_certificate 做校验，只展示原始抽取结果
SKIP_CROSS_CHECK_DOCS = {
    "diploma_certificate",
}


# =========================
# 基础工具
# =========================

def html_escape(value):
    text = "" if value is None else str(value)
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
    )


def is_empty_or_unknown(value):
    if value is None:
        return True

    v = str(value).strip().lower()

    if not v:
        return True

    return v in {
        "unknown",
        "unkown",
        "unclear",
        "n/a",
        "na",
        "none",
        "null",
        "-",
        "not available",
        "not provided",
        "not found",
    }


def get_chat_data(result):
    if not result:
        return {}
    return result.get("chat_data", {}) or {}


def normalize_key(key):
    return str(key or "").strip().lower()


def get_field_with_key(chat_data, aliases):
    """
    按多个可能字段名取值，同时返回实际命中的字段名。
    兼容大小写。
    """
    if not chat_data:
        return "", ""

    for key in aliases:
        if key in chat_data and not is_empty_or_unknown(chat_data.get(key)):
            return key, str(chat_data.get(key, "")).strip()

    lower_map = {
        normalize_key(k): k
        for k in chat_data.keys()
    }

    for alias in aliases:
        real_key = lower_map.get(normalize_key(alias))
        if real_key is not None:
            value = chat_data.get(real_key)
            if not is_empty_or_unknown(value):
                return real_key, str(value).strip()

    return "", ""


def get_field(chat_data, aliases):
    _, value = get_field_with_key(chat_data, aliases)
    return value


def yes_no_unknown(value):
    """
    统一 Yes / No / Unknown。
    """
    if is_empty_or_unknown(value):
        return "UNKNOWN"

    v = str(value).strip().upper()

    if v in {"YES", "Y", "TRUE"}:
        return "YES"

    if v in {"NO", "N", "FALSE"}:
        return "NO"

    if "NO" in v and "YES" not in v:
        return "NO"

    if "YES" in v:
        return "YES"

    if "UNKNOWN" in v or "UNCLEAR" in v:
        return "UNKNOWN"

    return "UNKNOWN"


# =========================
# 标准化
# =========================

def normalize_name(value):
    return re.sub(r"[^A-Z]", "", str(value or "").upper())


def normalize_passport_number(value):
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def normalize_gender(value):
    v = str(value or "").strip().upper()

    if v in {"F", "FEMALE", "WOMAN"}:
        return "FEMALE"

    if v in {"M", "MALE", "MAN"}:
        return "MALE"

    return v


def normalize_nationality(value):
    v = str(value or "").strip().upper()

    mapping = {
        "INDONESIA": "INDONESIAN",
        "INDONESIAN": "INDONESIAN",
        "CHINA": "CHINESE",
        "CHINESE": "CHINESE",
        "PRC": "CHINESE",
        "MAINLAND CHINA": "CHINESE",
    }

    return mapping.get(v, v)


def parse_date_flexible(value):
    if not value:
        return None

    raw = str(value).strip()
    raw = raw.replace(",", " ")
    raw = re.sub(r"\s+", " ", raw)

    formats = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%d %b %Y",
        "%d%b%Y",
        "%d %B %Y",
        "%d%B%Y",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(raw.upper(), fmt).date()
        except Exception:
            pass

    return None


def normalize_date(value):
    d = parse_date_flexible(value)
    return d.isoformat() if d else str(value or "").strip()


def calculate_age(dob, check_date=None):
    if not dob:
        return None

    check_date = check_date or CHECK_DATE

    return check_date.year - dob.year - (
        (check_date.month, check_date.day) < (dob.month, dob.day)
    )


# =========================
# 异常记录
# =========================

def make_issue(doc_type, field, reason, aliases=None):
    return {
        "doc_type": doc_type,
        "field": field,
        "reason": reason,
        "aliases": aliases or [field],
    }


def field_matches_issue(field_name, issue):
    field_norm = normalize_key(field_name)

    candidates = [issue.get("field", "")]
    candidates.extend(issue.get("aliases", []))

    return field_norm in {normalize_key(x) for x in candidates}


# =========================
# Passport 校验
# =========================

def validate_passport(results_by_type):
    issues = []

    passport_result = results_by_type.get("passport")
    if not passport_result:
        issues.append(
            make_issue(
                "passport",
                "passport",
                "未找到 passport，无法进行身份标准校验。"
            )
        )
        return issues

    data = get_chat_data(passport_result)

    passport_type_key, passport_type = get_field_with_key(
        data,
        ["passport type", "type"]
    )

    if not passport_type:
        issues.append(
            make_issue(
                "passport",
                "passport type",
                "未抽取到护照类型，无法判断是否为普通护照。",
                ["passport type", "type"]
            )
        )
    else:
        passport_type_norm = passport_type.strip().upper()
        if passport_type_norm not in NORMAL_PASSPORT_TYPES:
            issues.append(
                make_issue(
                    "passport",
                    passport_type_key or "passport type",
                    f"护照类型为 {passport_type}，不是普通护照，需要人工确认。",
                    ["passport type", "type"]
                )
            )

    expiry_key, expiry_raw = get_field_with_key(
        data,
        ["date of expiry", "expiry date", "passport expiry"]
    )

    expiry_date = parse_date_flexible(expiry_raw)

    if not expiry_raw:
        issues.append(
            make_issue(
                "passport",
                "date of expiry",
                "未抽取到护照有效期。",
                ["date of expiry", "expiry date", "passport expiry"]
            )
        )
    elif not expiry_date:
        issues.append(
            make_issue(
                "passport",
                expiry_key or "date of expiry",
                "护照有效期日期格式无法解析，需要人工确认。",
                ["date of expiry", "expiry date", "passport expiry"]
            )
        )
    else:
        days_left = (expiry_date - CHECK_DATE).days

        if days_left < 0:
            issues.append(
                make_issue(
                    "passport",
                    expiry_key or "date of expiry",
                    "护照已过期。",
                    ["date of expiry", "expiry date", "passport expiry"]
                )
            )
        elif days_left < PASSPORT_EXPIRY_WARNING_DAYS:
            issues.append(
                make_issue(
                    "passport",
                    expiry_key or "date of expiry",
                    f"护照有效期不足 {PASSPORT_EXPIRY_WARNING_DAYS} 天，剩余约 {days_left} 天。",
                    ["date of expiry", "expiry date", "passport expiry"]
                )
            )

    dob_key, dob_raw = get_field_with_key(
        data,
        ["date of birth", "dob", "birth date"]
    )

    dob = parse_date_flexible(dob_raw)

    if dob:
        age = calculate_age(dob)
        if age is not None and age < MIN_AGE:
            issues.append(
                make_issue(
                    "passport",
                    dob_key or "date of birth",
                    f"申请人年龄为 {age}，低于当前年龄阈值 {MIN_AGE}。",
                    ["date of birth", "dob", "birth date"]
                )
            )

    return issues


# =========================
# 跨文件身份一致性校验
# =========================

COMPARE_FIELDS = {
    "name": {
        "passport_aliases": [
            "name",
            "full name",
            "passport name",
        ],
        "doc_aliases": [
            "name",
            "student name",
            "applicant name",
            "full name",
        ],
        "normalizer": normalize_name,
    },
    "date of birth": {
        "passport_aliases": [
            "date of birth",
            "dob",
            "birth date",
        ],
        "doc_aliases": [
            "date of birth",
            "dob",
            "birth date",
            "applicant date of birth",
            "applicant dob",
        ],
        "normalizer": normalize_date,
    },
    "gender": {
        "passport_aliases": [
            "gender",
            "sex",
        ],
        "doc_aliases": [
            "gender",
            "sex",
        ],
        "normalizer": normalize_gender,
    },
    "nationality": {
        "passport_aliases": [
            "nationality",
            "citizenship",
            "country of nationality",
        ],
        "doc_aliases": [
            "nationality",
            "citizenship",
            "country of nationality",
        ],
        "normalizer": normalize_nationality,
    },
    "passport number": {
        "passport_aliases": [
            "passport number",
            "passport no",
            "passport no.",
            "document number",
            "passport id",
        ],
        "doc_aliases": [
            "passport number",
            "passport no",
            "passport no.",
            "document number",
            "passport id",
        ],
        "normalizer": normalize_passport_number,
    },
}


def compare_with_passport(results_by_type):
    issues = []

    passport_result = results_by_type.get("passport")
    if not passport_result:
        return issues

    passport_data = get_chat_data(passport_result)

    for doc_type, result in results_by_type.items():
        if doc_type == "passport":
            continue

        if doc_type in SKIP_CROSS_CHECK_DOCS:
            continue

        doc_data = get_chat_data(result)

        for field_name, cfg in COMPARE_FIELDS.items():
            passport_key, passport_value = get_field_with_key(
                passport_data,
                cfg["passport_aliases"]
            )

            doc_key, doc_value = get_field_with_key(
                doc_data,
                cfg["doc_aliases"]
            )

            # 两边都有值才判断不一致。
            # 缺失不直接标红，避免成绩单/英语成绩单没有某些字段时误报。
            if not passport_value or not doc_value:
                continue

            passport_norm = cfg["normalizer"](passport_value)
            doc_norm = cfg["normalizer"](doc_value)

            if passport_norm != doc_norm:
                issues.append(
                    make_issue(
                        doc_type,
                        doc_key or field_name,
                        (
                            f"与 passport 的 {field_name} 不一致；"
                            f"passport = {passport_value}，当前文件 = {doc_value}。"
                        ),
                        cfg["doc_aliases"]
                    )
                )

    return issues


# =========================
# Transcript 专项校验
# =========================

def validate_transcript(results_by_type):
    issues = []

    result = results_by_type.get("transcript")
    if not result:
        return issues

    data = get_chat_data(result)

    # 是否英语成绩单
    language_key, language = get_field_with_key(
        data,
        ["transcript language", "language"]
    )

    if not language:
        issues.append(
            make_issue(
                "transcript",
                "transcript language",
                "未抽取到成绩单语言，无法确认是否为英语成绩单。",
                ["transcript language", "language"]
            )
        )
    else:
        lang_upper = language.upper()
        if "NOT ENGLISH" in lang_upper or "ENGLISH" not in lang_upper:
            issues.append(
                make_issue(
                    "transcript",
                    language_key or "transcript language",
                    f"成绩单语言为 {language}，可能不是英语成绩单。",
                    ["transcript language", "language"]
                )
            )

    # 是否正式文件
    official_key, official = get_field_with_key(
        data,
        ["official document", "validity check"]
    )

    official_status = yes_no_unknown(official)

    if official_status != "YES":
        issues.append(
            make_issue(
                "transcript",
                official_key or "official document",
                "无法确认成绩单是正式文件，或判断为非正式文件。",
                ["official document", "validity check"]
            )
        )

    # 是否有学校盖章
    stamp_key, stamp = get_field_with_key(
        data,
        ["school stamp present", "stamp present"]
    )

    stamp_status = yes_no_unknown(stamp)

    if stamp_status != "YES":
        issues.append(
            make_issue(
                "transcript",
                stamp_key or "school stamp present",
                "未确认成绩单有学校盖章。",
                ["school stamp present", "stamp present"]
            )
        )

    # 印章文字与学校名是否一致
    seal_match_key, seal_match = get_field_with_key(
        data,
        ["seal school name match"]
    )

    seal_match_status = yes_no_unknown(seal_match)

    if seal_match_status != "YES":
        seal_reason = get_field(
            data,
            ["seal school name match reason"]
        )

        issues.append(
            make_issue(
                "transcript",
                seal_match_key or "seal school name match",
                "无法确认印章文字与成绩单学校名一致。"
                + (f" {seal_reason}" if seal_reason else ""),
                ["seal school name match"]
            )
        )
    # 是否有签名
    signature_key, signature = get_field_with_key(
        data,
        ["signature present"]
    )

    signature_status = yes_no_unknown(signature)

    if signature_status != "YES":
        issues.append(
            make_issue(
                "transcript",
                signature_key or "signature present",
                "未确认成绩单有签名。",
                ["signature present"]
            )
        )

    # 总平均分
    overall_key, overall_average = get_field_with_key(
        data,
        ["overall average", "average grade", "average score"]
    )

    if not overall_average:
        issues.append(
            make_issue(
                "transcript",
                "overall average",
                "未抽取到总平均成绩。",
                ["overall average", "average grade", "average score"]
            )
        )

    # 每一年平均成绩
    year_avg_key, each_year_average = get_field_with_key(
        data,
        [
            "each year average",
            "yearly average",
            "year average",
            "grade 10 average",
            "year 10 average",
            "g10 average",
        ]
    )

    if not each_year_average:
        issues.append(
            make_issue(
                "transcript",
                "each year average",
                "未抽取到每一年的平均成绩。",
                [
                    "each year average",
                    "yearly average",
                    "year average",
                    "grade 10 average",
                    "year 10 average",
                    "g10 average",
                ]
            )
        )

    # 是否全部及格
    pass_key, pass_status = get_field_with_key(
        data,
        ["pass status"]
    )

    pass_status_norm = yes_no_unknown(pass_status)

    if pass_status_norm != "YES":
        issues.append(
            make_issue(
                "transcript",
                pass_key or "pass status",
                "无法确认所有科目都及格，或检测到可能存在未及格科目。",
                ["pass status"]
            )
        )

    # 数学成绩
    math_key, math_score = get_field_with_key(
        data,
        ["math score", "mathematics score", "math"]
    )

    if not math_score:
        issues.append(
            make_issue(
                "transcript",
                "math score",
                "未抽取到数学成绩。",
                ["math score", "mathematics score", "math"]
            )
        )

    # 物理成绩
    physics_key, physics_score = get_field_with_key(
        data,
        ["physics score", "physics"]
    )

    if not physics_score:
        issues.append(
            make_issue(
                "transcript",
                "physics score",
                "未抽取到物理成绩。",
                ["physics score", "physics"]
            )
        )

    return issues


# =========================
# 总异常构建
# =========================

def build_validation_issues(results_by_type):
    issues = []

    issues.extend(validate_passport(results_by_type))
    issues.extend(compare_with_passport(results_by_type))
    issues.extend(validate_transcript(results_by_type))

    return issues


# =========================
# 输出：原始 txt 风格 + HTML 异常行标红
# =========================

def format_normal_line(key, value):
    return f"{html_escape(key)}: {html_escape(value)}"


def format_red_line(key, value, reason):
    value_text = value

    if is_empty_or_unknown(value_text):
        value_text = "[missing]"

    return (
        '<span class="issue">'
        f"{html_escape(key)}: {html_escape(value_text)}"
        f"  ← {html_escape(reason)}"
        "</span>"
    )


def collect_issues_for_field(issues, doc_type, field_name):
    matched = []

    for issue in issues:
        if issue.get("doc_type") != doc_type:
            continue

        if field_matches_issue(field_name, issue):
            matched.append(issue)

    return matched


def save_validation_report_to_html(
    results_by_type,
    output_file,
    doc_type_order=None,
):
    """
    生成 HTML 校验报告。

    输出效果：
    1. 保留原始 txt 风格。
    2. 异常字段所在行标红。
    3. 输出文件后缀为 .html。
    4. diploma_certificate 暂时不做专项检测，只展示原始抽取结果。
    """
    doc_type_order = doc_type_order or DEFAULT_DOC_TYPE_ORDER

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    issues = build_validation_issues(results_by_type)

    printed_doc_types = set()

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Document Validation Report</title>
<style>
body {
    font-family: Arial, Helvetica, sans-serif;
    margin: 24px;
    background: #ffffff;
    color: #111827;
}
pre {
    white-space: pre-wrap;
    word-wrap: break-word;
    font-family: Consolas, Monaco, "Courier New", monospace;
    font-size: 14px;
    line-height: 1.6;
}
.issue {
    color: red;
    font-weight: bold;
    background: #fff1f2;
}
.doc-title {
    font-weight: bold;
    color: #111827;
}
</style>
</head>
<body>
<pre>
""")

        for doc_type in doc_type_order:
            printed_doc_types.add(doc_type)

            f.write(f'<span class="doc-title">#{html_escape(doc_type)}</span>\n')

            result = results_by_type.get(doc_type)

            if not result:
                f.write(
                    format_red_line(
                        doc_type,
                        "[missing]",
                        "未提供或未识别到该类文件。"
                    )
                    + "\n\n"
                )
                continue

            chat_data = get_chat_data(result)
            field_meta = result.get("field_meta", []) or []

            printed_fields = set()
            printed_issue_ids = set()

            # 1. 按模板字段顺序输出
            for item in field_meta:
                key = item.get("name", "")
                if not key:
                    continue

                value = chat_data.get(key, "")
                printed_fields.add(normalize_key(key))

                matched_issues = collect_issues_for_field(
                    issues,
                    doc_type,
                    key
                )

                if matched_issues:
                    reason = "；".join(
                        issue.get("reason", "")
                        for issue in matched_issues
                    )

                    f.write(format_red_line(key, value, reason) + "\n")

                    for issue in matched_issues:
                        printed_issue_ids.add(id(issue))
                else:
                    f.write(format_normal_line(key, value) + "\n")

            # 2. 输出 chat_data 中存在，但模板字段里没有的额外字段
            for key, value in chat_data.items():
                if normalize_key(key) in printed_fields:
                    continue

                matched_issues = collect_issues_for_field(
                    issues,
                    doc_type,
                    key
                )

                if matched_issues:
                    reason = "；".join(
                        issue.get("reason", "")
                        for issue in matched_issues
                    )

                    f.write(format_red_line(key, value, reason) + "\n")

                    for issue in matched_issues:
                        printed_issue_ids.add(id(issue))
                else:
                    f.write(format_normal_line(key, value) + "\n")

            # 3. 如果异常字段原始结果里没有这一行，就补一行 missing
            for issue in issues:
                if issue.get("doc_type") != doc_type:
                    continue

                if id(issue) in printed_issue_ids:
                    continue

                field = issue.get("field", "")
                reason = issue.get("reason", "")

                f.write(format_red_line(field, "[missing]", reason) + "\n")

            f.write("\n")

        # 4. 防止有额外 doc_type 没被输出
        for doc_type, result in results_by_type.items():
            if doc_type in printed_doc_types:
                continue

            f.write(f'<span class="doc-title">#{html_escape(doc_type)}</span>\n')

            chat_data = get_chat_data(result)

            for key, value in chat_data.items():
                f.write(format_normal_line(key, value) + "\n")

            f.write("\n")

        f.write("""</pre>
</body>
</html>
""")

    return output_file

