from typing import Dict, List

def concept_matches_query(query_lc, rec, languages):
    """Return True if the query matches keyword, label, or any alias."""
    # Keyword match
    kw = rec.get("keyword", "").strip().lower()
    if query_lc == kw:
        return True

    langs = rec.get("languages", {})

    # Label match in ANY language
    for lang in languages:
        label = langs.get(lang, {}).get("label", "")
        if label and query_lc == label.strip().lower():
            return True

    # Aliases match in ANY language
    for lang in languages:
        aliases = langs.get(lang, {}).get("also_known_as", [])
        for alias in aliases:
            if alias and query_lc == alias.strip().lower():
                return True

    return False


def expand_query_with_vectors(
    query: str,
    kb: List[dict],
    encoder,
    languages: List[str] = ["en", "es", "ca", "it"]
) -> Dict:

    query_lc = query.strip().lower()

    # Collect outputs
    aliases_per_lang = {lang: [] for lang in languages}
    definition_vectors = []

    # Iterate over KB
    for rec in kb:
        # ---- MATCHING LOGIC FIXED HERE ----
        if not concept_matches_query(query_lc, rec, languages):
            continue

        langs = rec.get("languages", {})

        # Gather aliases & definitions
        for lang in languages:
            lang_data = langs.get(lang, {})

            # Aliases + label (sparse expansion)
            aliases = lang_data.get("also_known_as", [])
            label = lang_data.get("label")
            if label:
                aliases.append(label)
            aliases_per_lang[lang].extend(aliases)

            # Definition → dense vector
            definition = lang_data.get("description")
            if definition:
                enriched_text = f"{query}. {definition}"
                vec = encoder.encode(enriched_text).tolist()
                definition_vectors.append({
                    "language": lang,
                    "definition": definition,
                    "vector": vec
                })

    # Base query vector
    query_vector = encoder.encode(query).tolist()

    return {
        "query_vector": query_vector,
        "definition_vectors": definition_vectors,
        "aliases": aliases_per_lang
    }