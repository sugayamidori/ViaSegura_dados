import re
import pandas as pd

def clean_address_part(x):
    if pd.isna(x):
        return ""
    x = str(x).strip().lower()
    if x in ["", "nan", "none", "unknown", "desconhecido"]:
        return ""
    abbreviations = {
        r'\bav\b': 'avenida', r'\br\b': 'rua', r'\bestr\b': 'estrada',
        r'\best\b': 'estrada', r'\brod\b': 'rodovia', r'\btrav\b': 'travessa',
        r'\bal\b': 'alameda', r'\bpça\b': 'praça'
    }
    for abbr, full in abbreviations.items():
        x = re.sub(abbr, full, x)
    x = re.sub(r'\s+', ' ', x)
    return x

def safe_mode(x):
    mode_result = x.mode()
    return mode_result.iloc[0] if len(mode_result) > 0 else (x.iloc[0] if len(x) > 0 else 'unknown')
