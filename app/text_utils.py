import re

EN_CHARS = "qwertyuiop[]asdfghjkl;'zxcvbnm,./QWERTYUIOP{}ASDFGHJKL:\"ZXCVBNM<>?"
RU_CHARS = "йцукенгшщзхъфывапролджэячсмитьбю.ЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮ,"
EN_TO_RU = str.maketrans(EN_CHARS, RU_CHARS)

STOP_PHRASES = [
    r"\bподскажи(те)?\b", r"\bпожалуйста\b", r"\bскажи(те)?\b",
    r"\bможешь\b", r"\bобъясни(те)?\b", r"\bрасскажи(те)?\b",
    r"\bкратко\b", r"\bподробно\b", r"\bпривет\b", r"\bздравствуй(те)?\b"
]

def fix_keyboard_layout(text: str) -> str:
    """Переводит слова в сбитой латинской раскладке (rfr -> как), не трогая реальные англ. термины."""
    words = text.split()
    fixed_words = []
    for w in words:
        # Если в слове есть латиница и при этом нет ни одной английской гласной (a,e,i,o,u,y)
        # или внутри слова затесались запятая/точка с запятой/кавычка — это русское слово в EN раскладке
        if re.search(r"[a-zA-Z]", w) and not re.search(r"[а-яА-Я]", w):
            has_en_vowel = bool(re.search(r"[aeiouyAEIOUY]", w))
            has_punct_inside = bool(re.search(r"[a-zA-Z][,;'\[\]\{\}][a-zA-Z]?", w))
            # Исключаем короткие известные аббревиатуры (SQL, DB, CPU, GPU, PDF, MD, RRF, FTS)
            is_abbr = w.upper().strip(".,!?") in {"SQL", "DB", "CPU", "GPU", "PDF", "MD", "RRF", "FTS", "ETL", "API", "LLM", "RAG", "CSV"}
            if (not has_en_vowel or has_punct_inside) and not is_abbr:
                fixed_words.append(w.translate(EN_TO_RU))
                continue
        fixed_words.append(w)
    return " ".join(fixed_words)

def clean_search_query(text: str) -> str:
    """Исправляет раскладку и удаляет вежливые слова-паразиты для точного поиска."""
    cleaned = fix_keyboard_layout(text)
    for pattern in STOP_PHRASES:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.?!-")
    return cleaned if len(cleaned) >= 2 else fix_keyboard_layout(text)
