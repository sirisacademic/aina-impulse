"""
Query expansion using Wikidata KB with distance-based levels and clustering.

Version 3.0 changes:
- Centroid-based clustering (compare to cluster centroid, not seed)
- Different thresholds for aliases vs parents
- Language-aware representative selection when query language is known

Author: IMPULSE Project
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple
import numpy as np


@dataclass
class ExpansionConfig:
    """Configuration for query expansion behavior."""
    use_aliases: bool = True
    use_parents: bool = True
    use_definitions: bool = False
    only_parents_in_kb: bool = True
    max_parents: int = 3
    languages: List[str] = field(default_factory=lambda: ["en", "es", "ca"])
    
    # Different thresholds for aliases vs parents
    alias_level_boundaries: List[float] = field(default_factory=lambda: [0.85, 0.50, 0.30])
    parent_level_boundaries: List[float] = field(default_factory=lambda: [0.75, 0.50, 0.30])
    
    # Clustering threshold
    cluster_threshold: float = 0.70
    
    # Maximum clusters per level
    max_clusters_per_level: int = 5
    
    # Language preference tolerance (for representative selection)
    language_preference_tolerance: float = 0.05


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def build_kb_index(kb: List[dict]) -> Dict[str, dict]:
    """Build an index of wikidata_id -> KB record for fast parent lookup."""
    index = {}
    for rec in kb:
        wid = rec.get("wikidata_id", "")
        if wid and wid not in index:
            index[wid] = rec
    return index


def concept_matches_query(query_lc: str, rec: dict, languages: List[str]) -> bool:
    """Check if query exactly matches a KB concept's keyword, label, or alias."""
    kw = rec.get("keyword", "").strip().lower()
    if query_lc == kw:
        return True

    langs = rec.get("languages", {})

    for lang in languages:
        label = langs.get(lang, {}).get("label", "")
        if label and query_lc == label.strip().lower():
            return True

    for lang in languages:
        aliases = langs.get(lang, {}).get("also_known_as", [])
        for alias in aliases:
            if alias and query_lc == alias.strip().lower():
                return True

    return False


def extract_terms_from_concept(
    rec: dict, 
    languages: List[str],
    source: str,
    from_concept: str,
    from_parent: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Extract all terms (labels + aliases) from a KB concept with language info."""
    terms = []
    langs = rec.get("languages", {})
    
    for lang in languages:
        lang_data = langs.get(lang, {})
        
        label = lang_data.get("label")
        if label:
            terms.append({
                "term": label,
                "language": lang,
                "source": source,
                "source_type": "label",
                "from_concept": from_concept,
                "from_parent": from_parent
            })
        
        for alias in lang_data.get("also_known_as", []):
            if alias:
                terms.append({
                    "term": alias,
                    "language": lang,
                    "source": source,
                    "source_type": "alias",
                    "from_concept": from_concept,
                    "from_parent": from_parent
                })
    
    return terms


def get_parent_concepts(
    rec: dict,
    kb_index: Dict[str, dict],
    config: ExpansionConfig
) -> List[dict]:
    """Get parent concepts (via subclass_of) that exist in the KB."""
    parents = []
    subclass_of = rec.get("subclass_of", [])
    
    for parent_ref in subclass_of:
        if len(parents) >= config.max_parents:
            break
            
        parent_id = parent_ref.get("id", "")
        
        if config.only_parents_in_kb:
            if parent_id in kb_index:
                parents.append(kb_index[parent_id])
        else:
            parents.append({
                "wikidata_id": parent_id,
                "keyword": parent_ref.get("label", ""),
                "languages": {}
            })
    
    return parents


def get_distance_level(similarity: float, boundaries: List[float]) -> int:
    """Determine distance level based on similarity and boundaries."""
    for i, threshold in enumerate(boundaries):
        if similarity >= threshold:
            return i + 1
    return len(boundaries) + 1


def compute_centroid(vectors: List[np.ndarray]) -> np.ndarray:
    """Compute normalized centroid of vectors."""
    if not vectors:
        return np.array([])
    stacked = np.stack(vectors)
    centroid = np.mean(stacked, axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    return centroid


def cluster_terms_centroid_based(
    terms_with_vectors: List[Tuple[Dict, np.ndarray]],
    threshold: float = 0.70
) -> List[List[Tuple[Dict, np.ndarray]]]:
    """
    Cluster terms by comparing to cluster CENTROID (not just seed).
    
    Algorithm:
    1. Sort terms by some criteria (we use original order, highest query sim first)
    2. For each term, compare to existing cluster centroids
    3. Join cluster if similarity >= threshold, otherwise start new cluster
    4. Update centroid after each addition
    """
    if not terms_with_vectors:
        return []
    
    clusters: List[Dict] = []  # Each: {"terms": [...], "vectors": [...], "centroid": ndarray}
    
    for term_info, vec in terms_with_vectors:
        best_cluster_idx = -1
        best_similarity = 0.0
        
        # Compare to each existing cluster's centroid
        for idx, cluster in enumerate(clusters):
            sim = cosine_similarity(vec, cluster["centroid"])
            if sim >= threshold and sim > best_similarity:
                best_similarity = sim
                best_cluster_idx = idx
        
        if best_cluster_idx >= 0:
            # Add to existing cluster
            cluster = clusters[best_cluster_idx]
            cluster["terms"].append((term_info, vec))
            cluster["vectors"].append(vec)
            # Update centroid
            cluster["centroid"] = compute_centroid(cluster["vectors"])
        else:
            # Start new cluster
            clusters.append({
                "terms": [(term_info, vec)],
                "vectors": [vec],
                "centroid": vec.copy()
            })
    
    # Return just the terms lists
    return [c["terms"] for c in clusters]


def select_representative(
    cluster: List[Tuple[Dict, np.ndarray]],
    query_language: Optional[str],
    tolerance: float = 0.05
) -> Tuple[Dict, np.ndarray]:
    """
    Select cluster representative, preferring query language when known.
    
    Args:
        cluster: List of (term_info, vector) tuples, sorted by query similarity
        query_language: Language of the query (from parser), or None
        tolerance: How much similarity difference to tolerate for language preference
    
    Returns:
        The selected (term_info, vector) tuple
    """
    if not cluster:
        raise ValueError("Empty cluster")
    
    if len(cluster) == 1 or query_language is None:
        # No choice or no language preference
        return cluster[0]
    
    # cluster is sorted by query_similarity (highest first)
    best_term, best_vec = cluster[0]
    best_sim = best_term.get("query_similarity", 0)
    
    # Look for same-language term within tolerance
    for term_info, vec in cluster:
        term_sim = term_info.get("query_similarity", 0)
        term_lang = term_info.get("language", "")
        
        # If within tolerance and matches query language, prefer it
        if term_lang == query_language and (best_sim - term_sim) <= tolerance:
            return (term_info, vec)
    
    # No same-language term within tolerance, use highest similarity
    return (best_term, best_vec)


def process_terms_into_levels(
    terms: List[Dict],
    query_vec: np.ndarray,
    encoder,
    level_boundaries: List[float],
    config: ExpansionConfig,
    query_language: Optional[str] = None
) -> Dict[str, List[Dict]]:
    """
    Process terms: compute similarities, cluster with centroids, organize by level.
    
    Args:
        terms: List of term info dicts
        query_vec: Query embedding vector
        encoder: Embedder instance
        level_boundaries: Thresholds for this term type (alias or parent)
        config: Expansion configuration
        query_language: Query language for representative selection (optional)
    
    Returns:
        Dict with level keys containing cluster info
    """
    if not terms:
        return {"level_1": [], "level_2": [], "level_3": []}
    
    # Deduplicate terms by text
    unique_terms = {}
    for t in terms:
        term_text = t["term"]
        if term_text not in unique_terms:
            unique_terms[term_text] = t
    
    terms_list = list(unique_terms.values())
    
    # Embed all terms
    term_texts = [t["term"] for t in terms_list]
    term_vectors = encoder.encode(term_texts)
    
    # Compute query similarity and assign level
    terms_with_info = []
    for i, term_info in enumerate(terms_list):
        sim = cosine_similarity(query_vec, term_vectors[i])
        term_info["query_similarity"] = sim
        term_info["level"] = get_distance_level(sim, level_boundaries)
        terms_with_info.append((term_info, term_vectors[i]))
    
    # Sort by query similarity (highest first) for clustering
    terms_with_info.sort(key=lambda x: x[0]["query_similarity"], reverse=True)
    
    # Filter out terms below minimum threshold
    min_threshold = level_boundaries[-1] if level_boundaries else 0.3
    terms_with_info = [(t, v) for t, v in terms_with_info if t["query_similarity"] >= min_threshold]
    
    # Group by level, then cluster within each level
    levels = {"level_1": [], "level_2": [], "level_3": []}
    
    for level_num in [1, 2, 3]:
        level_key = f"level_{level_num}"
        level_terms = [(t, v) for t, v in terms_with_info if t["level"] == level_num]
        
        if not level_terms:
            continue
        
        # Cluster using centroid-based method
        clusters = cluster_terms_centroid_based(level_terms, config.cluster_threshold)
        
        cluster_list = []
        for cluster in clusters:
            # Sort cluster by query similarity
            cluster.sort(key=lambda x: x[0]["query_similarity"], reverse=True)
            
            # Select representative (language-aware if query_language known)
            rep_info, rep_vec = select_representative(
                cluster, 
                query_language, 
                config.language_preference_tolerance
            )
            
            # Collect all terms and vectors
            all_terms = [t["term"] for t, _ in cluster]
            all_vectors = [v for _, v in cluster]
            
            # Compute final centroid
            centroid = compute_centroid(all_vectors)
            
            cluster_list.append({
                "representative": rep_info["term"],
                "representative_language": rep_info.get("language", ""),
                "query_similarity": rep_info["query_similarity"],
                "terms": all_terms,
                "centroid": centroid.tolist(),
                "source_concept": rep_info.get("from_concept", "")
            })
        
        # Sort clusters by representative similarity
        cluster_list.sort(key=lambda x: x["query_similarity"], reverse=True)
        
        # Apply limit
        levels[level_key] = cluster_list[:config.max_clusters_per_level]
    
    return levels


def expand_query_with_vectors(
    query: str,
    kb: List[dict],
    encoder,
    config: Optional[ExpansionConfig] = None,
    kb_index: Optional[Dict[str, dict]] = None,
    query_language: Optional[str] = None
) -> Dict[str, Any]:
    """
    Expand query using Wikidata KB with distance-based levels and clustering.
    
    Args:
        query: User's search query
        kb: List of KB records
        encoder: Embedder instance
        config: Expansion configuration
        kb_index: Pre-built wikidata_id -> record index
        query_language: Query language (from parser) for representative selection
    
    Returns:
        Dictionary with expansion results
    """
    if config is None:
        config = ExpansionConfig()
    
    if kb_index is None:
        kb_index = build_kb_index(kb)
    
    query_lc = query.strip().lower()
    seen_wikidata_ids: Set[str] = set()
    
    alias_terms: List[Dict] = []
    parent_terms: List[Dict] = []
    matched_concepts: List[Dict] = []
    parent_concepts: List[Dict] = []
    
    # Embed query
    query_vec = encoder.encode([query])[0]
    
    # Find matching concepts
    for rec in kb:
        if not concept_matches_query(query_lc, rec, config.languages):
            continue
        
        wikidata_id = rec.get("wikidata_id", "")
        if wikidata_id in seen_wikidata_ids:
            continue
        
        seen_wikidata_ids.add(wikidata_id)
        
        matched_concepts.append({
            "wikidata_id": wikidata_id,
            "keyword": rec.get("keyword", ""),
            "label_en": rec.get("languages", {}).get("en", {}).get("label", "")
        })
        
        # Extract aliases
        if config.use_aliases:
            terms = extract_terms_from_concept(
                rec, config.languages, 
                source="alias", 
                from_concept=wikidata_id
            )
            alias_terms.extend(terms)
        
        # Extract parent concepts
        if config.use_parents:
            parents = get_parent_concepts(rec, kb_index, config)
            
            for parent_rec in parents:
                parent_id = parent_rec.get("wikidata_id", "")
                
                if parent_id in seen_wikidata_ids:
                    continue
                
                seen_wikidata_ids.add(parent_id)
                
                parent_concepts.append({
                    "wikidata_id": parent_id,
                    "keyword": parent_rec.get("keyword", ""),
                    "label_en": parent_rec.get("languages", {}).get("en", {}).get("label", ""),
                    "child_concept": wikidata_id
                })
                
                terms = extract_terms_from_concept(
                    parent_rec, config.languages,
                    source="parent",
                    from_concept=parent_id,
                    from_parent=wikidata_id
                )
                parent_terms.extend(terms)
    
    # Process with DIFFERENT thresholds
    alias_expansion = process_terms_into_levels(
        alias_terms, query_vec, encoder,
        level_boundaries=config.alias_level_boundaries,
        config=config,
        query_language=query_language
    )
    
    parent_expansion = process_terms_into_levels(
        parent_terms, query_vec, encoder,
        level_boundaries=config.parent_level_boundaries,
        config=config,
        query_language=query_language
    )
    
    return {
        "query": query,
        "query_language": query_language,
        "query_vector": query_vec.tolist(),
        "alias_expansion": alias_expansion,
        "parent_expansion": parent_expansion,
        "matched_concepts": matched_concepts,
        "parent_concepts": parent_concepts,
        "config": {
            "use_aliases": config.use_aliases,
            "use_parents": config.use_parents,
            "alias_level_boundaries": config.alias_level_boundaries,
            "parent_level_boundaries": config.parent_level_boundaries,
            "cluster_threshold": config.cluster_threshold
        }
    }


# === Helper functions for search integration ===

def get_active_centroids(
    expansion_result: Dict,
    alias_levels: List[int] = [1, 2],
    parent_levels: List[int] = []
) -> List[Dict]:
    """Get centroid vectors for active clusters based on selected levels."""
    centroids = []
    
    for level in alias_levels:
        level_key = f"level_{level}"
        for cluster in expansion_result.get("alias_expansion", {}).get(level_key, []):
            centroids.append({
                "representative": cluster["representative"],
                "centroid": cluster["centroid"],
                "source": "alias",
                "level": level,
                "query_similarity": cluster["query_similarity"]
            })
    
    for level in parent_levels:
        level_key = f"level_{level}"
        for cluster in expansion_result.get("parent_expansion", {}).get(level_key, []):
            centroids.append({
                "representative": cluster["representative"],
                "centroid": cluster["centroid"],
                "source": "parent",
                "level": level,
                "query_similarity": cluster["query_similarity"]
            })
    
    return centroids


def get_expansion_summary(expansion_result: Dict) -> Dict:
    """Get a summary of expansion for display/debugging."""
    summary = {
        "query": expansion_result.get("query", ""),
        "query_language": expansion_result.get("query_language"),
        "matched_concepts": len(expansion_result.get("matched_concepts", [])),
        "parent_concepts": len(expansion_result.get("parent_concepts", [])),
        "alias_levels": {},
        "parent_levels": {}
    }
    
    for level in ["level_1", "level_2", "level_3"]:
        alias_clusters = expansion_result.get("alias_expansion", {}).get(level, [])
        parent_clusters = expansion_result.get("parent_expansion", {}).get(level, [])
        
        if alias_clusters:
            summary["alias_levels"][level] = {
                "clusters": len(alias_clusters),
                "representatives": [c["representative"] for c in alias_clusters]
            }
        
        if parent_clusters:
            summary["parent_levels"][level] = {
                "clusters": len(parent_clusters),
                "representatives": [c["representative"] for c in parent_clusters]
            }
    
    return summary
