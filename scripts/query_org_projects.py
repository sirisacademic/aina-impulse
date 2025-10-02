
#!/usr/bin/env python
"""Answer: In how many H2020 projects related to sustainable energies did org X participate in Y1-Y2?"""
import argparse
from pathlib import Path
import pandas as pd

def norm(s):
    return (str(s) if s is not None else "").strip().lower()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", required=True, help="Organization name (exact, case-insensitive).")
    ap.add_argument("--framework", default="H2020", help="Normalized framework label (e.g., H2020).")
    ap.add_argument("--topic", default="sustainable_energy", choices=["sustainable_energy"])
    ap.add_argument("--start", type=int, required=True, help="Start year inclusive (e.g., 2016)")
    ap.add_argument("--end", type=int, required=True, help="End year inclusive (e.g., 2020)")
    ap.add_argument("--meta-dir", default="data/meta", help="Directory for projects.parquet and project_orgs.parquet")
    args = ap.parse_args()

    meta_dir = Path(args.meta_dir)
    projects = pd.read_parquet(meta_dir / "projects.parquet")
    project_orgs = pd.read_parquet(meta_dir / "project_orgs.parquet")

    projects["projectId"] = projects["projectId"].astype(str)
    project_orgs["projectId"] = project_orgs["projectId"].astype(str)

    fw_mask = projects["framework_norm"].map(norm) == norm(args.framework)
    projects["year_norm"] = pd.to_numeric(projects["year_norm"], errors="coerce")
    yr_mask = (projects["year_norm"] >= args.start) & (projects["year_norm"] <= args.end)

    topic_mask = True  # kept for compatibility; WP1 does not pre-tag topics
    proj_filtered = projects[fw_mask & yr_mask & topic_mask][["projectId", "framework_norm", "year_norm"]]

    project_orgs["organizationName_norm"] = project_orgs["organizationName"].map(norm)
    org_hits = project_orgs[project_orgs["organizationName_norm"] == norm(args.org)]

    joined = org_hits.merge(proj_filtered, on="projectId", how="inner")
    n_unique_projects = joined["projectId"].nunique()

    print(f"Organization: {args.org}")
    print(f"Framework: {args.framework} | Topic: {args.topic} | Years: {args.start}-{args.end}")
    print(f"Unique projects: {n_unique_projects}")
    if n_unique_projects > 0:
        print(joined[["projectId", "framework_norm", "year_norm", "organizationName"]]
              .drop_duplicates()
              .head(20)
              .to_string(index=False))

if __name__ == "__main__":
    main()
