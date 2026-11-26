import re

EN_CHARS = "`qwertyuiop[]asdfghjkl;'zxcvbnm,./~QWERTYUIOP{}ASDFGHJKL:\"ZXCVBNM<>?"
RU_CHARS = "ёйцукенгшщзхъфывапролджэячсмитьбю.ЁЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮ,"
EN_TO_RU = str.maketrans(EN_CHARS, RU_CHARS)

SQL_AND_TECH_WORDS = {
    "SQL", "DB", "CPU", "GPU", "PDF", "MD", "RRF", "FTS", "ETL", "API", "LLM", "RAG", "CSV",
    "SELECT", "FROM", "WHERE", "GROUP", "ORDER", "BY", "HAVING", "JOIN", "LEFT", "RIGHT",
    "INNER", "OUTER", "FULL", "ON", "IN", "AS", "OR", "AND", "NOT", "NULL", "IS", "DISTINCT",
    "COUNT", "SUM", "AVG", "MIN", "MAX", "LIMIT", "OFFSET", "UNION", "WITH", "OVER", "PARTITION",
    "INDEX", "TABLE", "CREATE", "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "PRIMARY", "KEY",
    "PYTHON", "DOCKER", "POSTGRES", "POSTGRESQL", "ONNX", "JSON", "JSONB", "VECTOR"
}

STOP_PHRASES = [
    r"\bподскажи(те)?\b", r"\bпожалуйста\b", r"\bскажи(те)?\b",
    r"\bможешь\b", r"\bобъясни(те)?\b", r"\bрасскажи(те)?\b",
    r"\bпривет\b", r"\bздравствуй(те)?\b"
]

def fix_keyboard_layout(text: str) -> str:
    words = text.split()
    fixed_words = []
    for w in words:
        clean_token = w.strip(".,!?:;\"'()").upper()
        if clean_token in SQL_AND_TECH_WORDS:
            fixed_words.append(w)
            continue

        if re.search(r"[a-zA-Z]", w) and not re.search(r"[а-яА-Я]", w):
            has_en_vowel = bool(re.search(r"[aeiouAEIOU]", w))
            has_punct_inside = bool(re.search(r"[a-zA-Z][,;'\[\]\{\}][a-zA-Z]?", w))
            # Характерные биграммы русского текста в английской раскладке
            has_ru_bigram = bool(re.search(r"(jy|hj|vf|wb|bz|lj|ty|nf|ae|rw|bb|gh|cj|ct|yj|gk|bl|hf|gj)", w.lower()))
            if not has_en_vowel or has_punct_inside or has_ru_bigram:
                fixed_words.append(w.translate(EN_TO_RU))
                continue
        fixed_words.append(w)
    return " ".join(fixed_words)

def clean_search_query(text: str) -> str:
    cleaned = fix_keyboard_layout(text)
    for pattern in STOP_PHRASES:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.?!-")
    return cleaned if len(cleaned) >= 2 else fix_keyboard_layout(text)
