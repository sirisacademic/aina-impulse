
#!/usr/bin/env python
"""CLI helper that queries the running IMPULSE API (/search) with semantic + metadata filters."""
import argparse
import json
import sys
import requests

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://127.0.0.1", help="API host (scheme+host), e.g., http://127.0.0.1")
    ap.add_argument("--port", type=int, default=8000, help="API port (default: 8000)")
    ap.add_argument("--query", required=True, help="Semantic query text")
    ap.add_argument("--k", type=int, default=5, help="Top-k results to return")
    ap.add_argument("--k-factor", type=int, default=5, help="Candidate multiplier for post-filtering")
    ap.add_argument("--framework", nargs="*", default=None, help="Framework filter (one or more, case-insensitive exact)")
    ap.add_argument("--ris3cat-ambit", nargs="*", default=None, help="RIS3CAT Ambit filter (one or more)")
    ap.add_argument("--ris3cat-tft", nargs="*", default=None, help="RIS3CAT TFT filter (one or more)")
    ap.add_argument("--year-from", type=int, default=None, help="Start year (inclusive)")
    ap.add_argument("--year-to", type=int, default=None, help="End year (inclusive)")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON response")
    args = ap.parse_args()

    url = f"{args.host}:{args.port}/search"
    payload = {"query": args.query, "k": args.k, "k_factor": args.k_factor}

    filters = {}
    if args.framework:       filters["framework"] = args.framework
    if args.ris3cat_ambit:   filters["ris3cat_ambit"] = args.ris3cat_ambit
    if args.ris3cat_tft:     filters["ris3cat_tft"] = args.ris3cat_tft
    if args.year_from is not None: filters["year_from"] = args.year_from
    if args.year_to   is not None: filters["year_to"]   = args.year_to
    if filters: payload["filters"] = filters

    try:
        r = requests.post(url, json=payload, timeout=60)
    except requests.RequestException as e:
        print(f"ERROR: request failed: {e}", file=sys.stderr)
        sys.exit(2)

    if r.status_code != 200:
        print(f"ERROR: HTTP {r.status_code}: {r.text}", file=sys.stderr)
        sys.exit(1)

    data = r.json()
    if args.pretty:
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return

    print(f"Query: {data.get('query')}  |  returned: {data.get('returned')} / k={data.get('k')}")
    if data.get("filters"):
        print(f"Filters: {data['filters']}")
    print("Results:")
    for i, item in enumerate(data.get("results", []), 1):
        mid = item.get("id")
        score = item.get("score")
        meta = item.get("metadata", {})
        fw = meta.get("frameworkName") or meta.get("framework_norm") or ""
        yr = meta.get("year_norm") or meta.get("startingYear") or ""
        ambit = meta.get("RIS3CAT_Ambit") or ""
        tft = meta.get("RIS3CAT_TFT") or ""
        print(f" {i:>2}. {mid}  score={score:.4f}  framework={fw}  year={yr}  ambit={ambit}  tft={tft}")

if __name__ == "__main__":
    main()
