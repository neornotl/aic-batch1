"""Temporal clustering, neighbor context, duplicate suppression, and event scoring."""
from __future__ import annotations
import re

def temporal_cluster(rows, gap=8.0):
    groups=[]
    for row in sorted(rows, key=lambda x: (x.get("video_id", ""), x.get("timestamp") or 0)):
        if not groups or row.get("video_id") != groups[-1][-1].get("video_id") or (row.get("timestamp") or 0) - (groups[-1][-1].get("timestamp") or 0) > gap: groups.append([])
        groups[-1].append(row)
    return groups

def suppress_duplicates(rows, per_video=3, min_gap=2):
    out=[]; counts={}; last={}
    for row in rows:
        vid=row.get("video_id"); ts=row.get("timestamp")
        if counts.get(vid, 0) >= per_video: continue
        if ts is not None and last.get(vid) is not None and abs(ts-last[vid]) < min_gap: continue
        out.append(row); counts[vid]=counts.get(vid,0)+1; last[vid]=ts
    return out

def event_terms(query):
    return [x.strip() for x in re.split(r"\b(?:and then|trước khi|sau khi|before|after|sau đó|then|rồi)\b", query, flags=re.I) if x.strip()]

def temporal_score(row, query):
    terms=event_terms(query)
    if len(terms) < 2: return 0.0
    text=(row.get("text") or "").lower()
    return sum(1 for term in terms if any(token in text for token in re.findall(r"[\wÀ-ỹ]+", term.lower()))) / len(terms)
