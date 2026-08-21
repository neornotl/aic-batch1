"""Query expansion and language variants without requiring an LLM."""
from __future__ import annotations
import re

VI_EN = {"người": "person people", "xe": "car vehicle", "đỏ": "red", "xanh": "blue",
         "đứng": "standing", "ngồi": "sitting", "đi": "walking", "nói": "speaking talk",
         "nhà": "house building", "biển": "sea sign", "màn hình": "screen display"}

def expand_query(query: str, remote: list[str] | None = None) -> list[str]:
    q = " ".join(query.strip().split())
    variants = [q]
    translated = q.lower()
    for vi, en in sorted(VI_EN.items(), key=lambda x: -len(x[0])):
        translated = translated.replace(vi, en)
    if translated != q.lower(): variants.append(translated)
    tokens = re.findall(r"[\wÀ-ỹ]+", q.lower())
    if len(tokens) > 1: variants.append(" ".join(tokens))
    for value in remote or []:
        if value and value not in variants: variants.append(value)
    return variants
