import re
from typing import Dict, List

from medical_dictionary import MEDICAL_PATTERNS, MEDICAL_SYNONYMS


# =========================
# basic cleaning
# =========================

def clean_query(query: str) -> str:

    query = query.lower().strip()
    query = re.sub(r"[^a-z0-9\s\-]", " ", query)
    query = re.sub(r"\s+", " ", query).strip()

    return query


# =========================
# entity extraction
# =========================

def extract_entities(query: str) -> Dict[str, List[str]]:

    entities = {}

    for entity_type, pattern in MEDICAL_PATTERNS.items():

        matches = re.findall(pattern, query)

        if matches:
            entities[entity_type] = sorted(set(matches))

    return entities


# =========================
# synonym expansion
# =========================

def expand_terms(query: str) -> List[str]:

    expanded = []

    for key, values in MEDICAL_SYNONYMS.items():

        if re.search(rf"\b{re.escape(key)}\b", query):
            expanded.extend(values)

    expanded = sorted(set(expanded))

    return [term for term in expanded if term != query]


# =========================
# query builders
# =========================

def build_vector_query(query: str) -> str:

    return f"Represent this question for searching relevant passages: {query}"



def build_keyword_query(query: str, expanded_terms: List[str]) -> str:

    parts = [query]

    if expanded_terms:
        parts.extend(expanded_terms)

    return " ".join(parts)


# =========================
# filter extraction
# =========================

def extract_filters(query: str) -> Dict[str, object]:

    filters = {}

    years = re.findall(r"\b(19\d{2}|20\d{2})\b", query)

    if years:
        filters["years"] = [int(y) for y in years]

    m = re.search(r"after\s+(19\d{2}|20\d{2})", query)
    if m:
        filters["year_gte"] = int(m.group(1))

    m = re.search(r"before\s+(19\d{2}|20\d{2})", query)
    if m:
        filters["year_lte"] = int(m.group(1))

    m = re.search(r"last\s+(\d+)\s+years", query)
    if m:
        filters["last_n_years"] = int(m.group(1))

    if "recent" in query:
        filters["recent"] = True

    return filters


# =========================
# main processor
# =========================

def process_query(query: str) -> Dict[str, object]:

    clean = clean_query(query)
    entities = extract_entities(clean)
    expanded_terms = expand_terms(clean)
    vector_query = build_vector_query(clean)
    keyword_query = build_keyword_query(clean, expanded_terms)
    filters = extract_filters(clean)

    return {
        "original_query": query,
        "cleaned_query": clean,
        "entities": entities,
        "expanded_terms": expanded_terms,
        "vector_query": vector_query,
        "keyword_query": keyword_query,
        "filters": filters,
    }
