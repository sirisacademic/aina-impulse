#!/usr/bin/env python3
"""
Comprehensive analysis of RIS3CAT data (NEW deduplicated parquet structure)

This script analyzes the deduplicated RIS3CAT dataset consisting of:
- project_db.parquet: Project-level information (deduplicated, no instrumentId)
- participant_db.parquet: Organization participation records

Column structure:
- Projects: project_id, title, abstract, instrument_name, framework_name, 
            year, total_investment, total_grant
- Participants: project_id, organization_id, organization_name, 
                organization_type, country, region, province, role, city

Author: IMPULS Project
Date: 2024
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
from pathlib import Path
import argparse


def load_ris3cat_data(data_dir='data/ris3cat'):
    """
    Load RIS3CAT data from normalized parquet files
    
    Args:
        data_dir: Directory containing the parquet files
        
    Returns:
        projects: DataFrame with project-level data
        participants: DataFrame with organization participation data
    """
    data_path = Path(data_dir)
    
    print("Loading RIS3CAT data from parquet files...")
    print(f"Data directory: {data_path.resolve()}")
    
    # Load parquet files
    projects_file = data_path / "project_db.parquet"
    participants_file = data_path / "participant_db.parquet"
    
    if not projects_file.exists():
        raise FileNotFoundError(f"Project file not found: {projects_file}")
    if not participants_file.exists():
        raise FileNotFoundError(f"Participants file not found: {participants_file}")
    
    projects = pd.read_parquet(projects_file)
    participants = pd.read_parquet(participants_file)
    
    print(f"✓ Loaded {len(projects)} projects")
    print(f"✓ Loaded {len(participants)} participation records")
    
    return projects, participants


def analyze_ris3cat_data(data_dir='data/ris3cat', output_file='ris3cat_analysis.json'):
    """
    Comprehensive analysis of RIS3CAT data to inform query schema design
    
    Args:
        data_dir: Directory containing the parquet files
        output_file: Path for JSON output file
    """
    
    # Load data
    projects, participants = load_ris3cat_data(data_dir)
    
    # Basic statistics
    print("\n" + "="*80)
    print("DATASET OVERVIEW")
    print("="*80)
    print(f"Projects shape: {projects.shape}")
    print(f"Participants shape: {participants.shape}")
    print(f"Unique projects: {projects['project_id'].nunique()}")
    print(f"Unique organizations: {participants['organization_id'].nunique()}")
    print("\n" + "="*80 + "\n")
    
    # Store results
    analysis = {
        'metadata': {
            'analysis_date': datetime.now().isoformat(),
            'data_source': str(data_dir),
            'projects_count': len(projects),
            'participants_count': len(participants),
            'unique_projects': projects['project_id'].nunique(),
            'unique_organizations': participants['organization_id'].nunique()
        }
    }
    
    # ========================================================================
    # 1. PROGRAMME/FRAMEWORK ANALYSIS
    # ========================================================================
    print("1. PROGRAMME/FRAMEWORK ANALYSIS")
    print("-" * 40)
    
    analysis['programmes'] = {
        'framework_name': {
            'unique_values': int(projects['framework_name'].nunique()),
            'values': projects['framework_name'].value_counts().to_dict(),
            'nulls': int(projects['framework_name'].isna().sum())
        },
        'instrument_name': {
            'unique_values': int(projects['instrument_name'].nunique()),
            'values': projects['instrument_name'].value_counts().to_dict(),
            'nulls': int(projects['instrument_name'].isna().sum())
        }
    }
    
    print(f"Framework names ({projects['framework_name'].nunique()} unique):")
    print(projects['framework_name'].value_counts())
    print(f"\nInstrument names ({projects['instrument_name'].nunique()} unique):")
    print(projects['instrument_name'].value_counts())
    print("\n" + "="*80 + "\n")
    
    # ========================================================================
    # 2. ORGANIZATION TYPE ANALYSIS
    # ========================================================================
    print("2. ORGANIZATION TYPE ANALYSIS")
    print("-" * 40)
    
    analysis['entity_types'] = {
        'organization_type': {
            'unique_values': int(participants['organization_type'].nunique()),
            'values': participants['organization_type'].value_counts().to_dict(),
            'nulls': int(participants['organization_type'].isna().sum())
        }
    }
    
    print(f"Organization types ({participants['organization_type'].nunique()} unique):")
    print(participants['organization_type'].value_counts())
    
    # Participation by org type
    print(f"\nParticipation records by organization type:")
    print(participants.groupby('organization_type')['project_id'].count().sort_values(ascending=False))
    print("\n" + "="*80 + "\n")
    
    # ========================================================================
    # 3. GEOGRAPHIC ANALYSIS
    # ========================================================================
    print("3. GEOGRAPHIC ANALYSIS")
    print("-" * 40)
    
    # Countries
    analysis['geography'] = {
        'countries': {
            'unique_values': int(participants['country'].nunique()),
            'values': participants['country'].value_counts().to_dict(),
            'nulls': int(participants['country'].isna().sum())
        },
        'regions': {
            'unique_values': int(participants['region'].nunique()),
            'spain_regions': participants[participants['country'] == 'Spain']['region'].value_counts().to_dict(),
            'nulls': int(participants['region'].isna().sum())
        },
        'provinces': {
            'unique_values': int(participants['province'].nunique()),
            'spain_provinces': participants[participants['country'] == 'Spain']['province'].value_counts().to_dict(),
            'nulls': int(participants['province'].isna().sum())
        },
        'cities': {
            'unique_values': int(participants['city'].nunique()),
            'nulls': int(participants['city'].isna().sum()),
            'top_cities': participants['city'].value_counts().head(20).to_dict() if participants['city'].notna().any() else {}
        }
    }
    
    print(f"Countries ({participants['country'].nunique()} unique):")
    print(participants['country'].value_counts())
    
    print(f"\nSpanish regions (NUTS2):")
    spanish_regions = participants[participants['country'] == 'Spain']['region'].value_counts()
    print(spanish_regions)
    
    print(f"\nSpanish provinces (NUTS3):")
    spanish_provinces = participants[participants['country'] == 'Spain']['province'].value_counts()
    print(spanish_provinces)
    
    if participants['city'].notna().any():
        print(f"\nTop 20 cities:")
        print(participants['city'].value_counts().head(20))
    else:
        print(f"\nNo city data available")
    
    print("\n" + "="*80 + "\n")
    
    # ========================================================================
    # 4. ROLE ANALYSIS (NEW FIELD)
    # ========================================================================
    print("4. ROLE ANALYSIS")
    print("-" * 40)
    
    analysis['roles'] = {
        'unique_values': int(participants['role'].nunique()) if 'role' in participants.columns else 0,
        'values': participants['role'].value_counts().to_dict() if 'role' in participants.columns and participants['role'].notna().any() else {},
        'nulls': int(participants['role'].isna().sum()) if 'role' in participants.columns else len(participants)
    }
    
    if 'role' in participants.columns and participants['role'].notna().any():
        print(f"Organization roles ({participants['role'].nunique()} unique):")
        print(participants['role'].value_counts())
        print(f"Role coverage: {(participants['role'].notna().sum() / len(participants)) * 100:.1f}%")
    else:
        print("No role data available or all null")
    
    print("\n" + "="*80 + "\n")
    
    # ========================================================================
    # 5. TEMPORAL ANALYSIS
    # ========================================================================
    print("5. TEMPORAL ANALYSIS")
    print("-" * 40)
    
    # Convert year to numeric if needed
    projects['year_numeric'] = pd.to_numeric(projects['year'], errors='coerce')
    
    analysis['temporal'] = {
        'year_range': {
            'min': int(projects['year_numeric'].min()) if pd.notna(projects['year_numeric'].min()) else None,
            'max': int(projects['year_numeric'].max()) if pd.notna(projects['year_numeric'].max()) else None
        },
        'year_distribution': projects['year_numeric'].value_counts().sort_index().to_dict(),
        'nulls': int(projects['year_numeric'].isna().sum())
    }
    
    print(f"Year range: {projects['year_numeric'].min():.0f} - {projects['year_numeric'].max():.0f}")
    print(f"Projects by year:")
    print(projects['year_numeric'].value_counts().sort_index())
    
    print("\n" + "="*80 + "\n")
    
    # ========================================================================
    # 6. COLLABORATION ANALYSIS
    # ========================================================================
    print("6. COLLABORATION ANALYSIS")
    print("-" * 40)
    
    # Partners per project
    partners_per_project = participants.groupby('project_id')['organization_id'].nunique()
    
    analysis['collaboration'] = {
        'avg_partners': float(partners_per_project.mean()),
        'max_partners': int(partners_per_project.max()),
        'min_partners': int(partners_per_project.min()),
        'median_partners': float(partners_per_project.median()),
        'single_partner_projects': int((partners_per_project == 1).sum()),
        'multi_partner_projects': int((partners_per_project > 1).sum())
    }
    
    print(f"Average partners per project: {partners_per_project.mean():.2f}")
    print(f"Median partners per project: {partners_per_project.median():.0f}")
    print(f"Max partners in a project: {partners_per_project.max()}")
    print(f"Min partners in a project: {partners_per_project.min()}")
    print(f"Single-partner projects: {(partners_per_project == 1).sum()}")
    print(f"Multi-partner projects: {(partners_per_project > 1).sum()}")
    
    # International collaborations
    countries_per_project = participants.groupby('project_id')['country'].nunique()
    analysis['collaboration']['international_projects'] = int((countries_per_project > 1).sum())
    
    print(f"International projects (>1 country): {(countries_per_project > 1).sum()}")
    
    # Top organizations by participation
    print(f"\nTop 100 organizations by participation:")
    top_orgs = participants.groupby('organization_name')['project_id'].nunique().sort_values(ascending=False).head(100)
    print(top_orgs)
    
    analysis['collaboration']['top_organizations'] = top_orgs.to_dict()
    
    print("\n" + "="*80 + "\n")
    
    # ========================================================================
    # 7. FINANCIAL ANALYSIS
    # ========================================================================
    print("7. FINANCIAL ANALYSIS")
    print("-" * 40)
    
    # Convert to numeric
    projects['total_investment_numeric'] = pd.to_numeric(projects['total_investment'], errors='coerce')
    projects['total_grant_numeric'] = pd.to_numeric(projects['total_grant'], errors='coerce')
    
    analysis['financial'] = {
        'total_investment': {
            'min': float(projects['total_investment_numeric'].min()) if pd.notna(projects['total_investment_numeric'].min()) else None,
            'max': float(projects['total_investment_numeric'].max()) if pd.notna(projects['total_investment_numeric'].max()) else None,
            'mean': float(projects['total_investment_numeric'].mean()) if pd.notna(projects['total_investment_numeric'].mean()) else None,
            'median': float(projects['total_investment_numeric'].median()) if pd.notna(projects['total_investment_numeric'].median()) else None,
            'sum': float(projects['total_investment_numeric'].sum()) if pd.notna(projects['total_investment_numeric'].sum()) else None,
            'nulls': int(projects['total_investment_numeric'].isna().sum())
        },
        'total_grant': {
            'min': float(projects['total_grant_numeric'].min()) if pd.notna(projects['total_grant_numeric'].min()) else None,
            'max': float(projects['total_grant_numeric'].max()) if pd.notna(projects['total_grant_numeric'].max()) else None,
            'mean': float(projects['total_grant_numeric'].mean()) if pd.notna(projects['total_grant_numeric'].mean()) else None,
            'median': float(projects['total_grant_numeric'].median()) if pd.notna(projects['total_grant_numeric'].median()) else None,
            'sum': float(projects['total_grant_numeric'].sum()) if pd.notna(projects['total_grant_numeric'].sum()) else None,
            'nulls': int(projects['total_grant_numeric'].isna().sum())
        }
    }
    
    print(f"Total Investment:")
    print(f"  Range: €{projects['total_investment_numeric'].min():,.0f} - €{projects['total_investment_numeric'].max():,.0f}")
    print(f"  Mean: €{projects['total_investment_numeric'].mean():,.0f}")
    print(f"  Median: €{projects['total_investment_numeric'].median():,.0f}")
    print(f"  Sum: €{projects['total_investment_numeric'].sum():,.0f}")
    print(f"  Nulls: {projects['total_investment_numeric'].isna().sum()}")
    
    print(f"\nTotal Grant:")
    print(f"  Range: €{projects['total_grant_numeric'].min():,.0f} - €{projects['total_grant_numeric'].max():,.0f}")
    print(f"  Mean: €{projects['total_grant_numeric'].mean():,.0f}")
    print(f"  Median: €{projects['total_grant_numeric'].median():,.0f}")
    print(f"  Sum: €{projects['total_grant_numeric'].sum():,.0f}")
    print(f"  Nulls: {projects['total_grant_numeric'].isna().sum()}")
    
    print("\n" + "="*80 + "\n")
    
    # ========================================================================
    # 8. TEXT FIELDS ANALYSIS
    # ========================================================================
    print("8. TEXT FIELDS ANALYSIS")
    print("-" * 40)
    
    # Sample abstracts for language detection
    abstracts_available = projects['abstract'].notna()
    sample_size = min(100, abstracts_available.sum())
    
    if sample_size > 0:
        sample_abstracts = projects[abstracts_available]['abstract'].sample(sample_size, random_state=42)
        
        # Simple language detection based on common words
        lang_indicators = {
            'en': ['the', 'and', 'of', 'to', 'in', 'for', 'with', 'will', 'project'],
            'es': ['el', 'la', 'de', 'en', 'por', 'para', 'con', 'los', 'las'],
            'ca': ['el', 'la', 'de', 'per', 'amb', 'dels', 'les', 'aquest', 'projecte']
        }
        
        lang_counts = {'en': 0, 'es': 0, 'ca': 0, 'other': 0}
        for abstract in sample_abstracts:
            abstract_lower = str(abstract).lower()
            scores = {}
            for lang, words in lang_indicators.items():
                scores[lang] = sum(1 for word in words if f' {word} ' in abstract_lower)
            
            if max(scores.values()) > 0:
                detected_lang = max(scores, key=scores.get)
                lang_counts[detected_lang] += 1
            else:
                lang_counts['other'] += 1
        
        avg_abstract_length = projects[abstracts_available]['abstract'].str.len().mean()
    else:
        lang_counts = {'en': 0, 'es': 0, 'ca': 0, 'other': 0}
        avg_abstract_length = 0
    
    analysis['text_fields'] = {
        'abstracts_available': int(abstracts_available.sum()),
        'abstracts_null': int((~abstracts_available).sum()),
        'abstracts_coverage': float((abstracts_available.sum() / len(projects)) * 100),
        'titles_available': int(projects['title'].notna().sum()),
        'titles_null': int(projects['title'].isna().sum()),
        'titles_coverage': float((projects['title'].notna().sum() / len(projects)) * 100),
        'language_sample': lang_counts,
        'avg_abstract_length': float(avg_abstract_length) if avg_abstract_length > 0 else None
    }
    
    print(f"Projects with abstracts: {abstracts_available.sum()} / {len(projects)} ({(abstracts_available.sum()/len(projects)*100):.1f}%)")
    print(f"Projects with titles: {projects['title'].notna().sum()} / {len(projects)} ({(projects['title'].notna().sum()/len(projects)*100):.1f}%)")
    
    if sample_size > 0:
        print(f"Language detection (sample of {len(sample_abstracts)}): {lang_counts}")
        print(f"Average abstract length: {avg_abstract_length:.0f} characters")
    else:
        print("No abstracts available for language detection")
    
    print("\n" + "="*80 + "\n")
    
    # ========================================================================
    # 9. DATA QUALITY CHECKS
    # ========================================================================
    print("9. DATA QUALITY CHECKS")
    print("-" * 40)
    
    # Check for orphaned records
    project_ids_in_participants = set(participants['project_id'].unique())
    project_ids_in_projects = set(projects['project_id'].unique())
    
    orphaned_participants = project_ids_in_participants - project_ids_in_projects
    projects_without_participants = project_ids_in_projects - project_ids_in_participants
    
    analysis['data_quality'] = {
        'orphaned_participant_records': len(orphaned_participants),
        'projects_without_participants': len(projects_without_participants),
        'projects_with_participants': len(project_ids_in_projects & project_ids_in_participants)
    }
    
    print(f"Projects in projects table: {len(project_ids_in_projects)}")
    print(f"Projects in participants table: {len(project_ids_in_participants)}")
    print(f"Orphaned participant records (no matching project): {len(orphaned_participants)}")
    print(f"Projects without participants: {len(projects_without_participants)}")
    print(f"Projects with participants: {len(project_ids_in_projects & project_ids_in_participants)}")
    
    # Null counts
    print(f"\nNull counts in projects table:")
    null_counts_projects = projects.isnull().sum()
    print(null_counts_projects[null_counts_projects > 0])
    
    print(f"\nNull counts in participants table:")
    null_counts_participants = participants.isnull().sum()
    print(null_counts_participants[null_counts_participants > 0])
    
    print("\n" + "="*80 + "\n")
    
    # ========================================================================
    # 10. SCHEMA RECOMMENDATIONS FOR QUERY PARSING
    # ========================================================================
    print("10. SCHEMA RECOMMENDATIONS FOR QUERY PARSING")
    print("-" * 40)
    
    schema_recommendations = {
        "programme": sorted([str(x) for x in projects['framework_name'].dropna().unique()]),
        "entity_type": sorted([str(x) for x in participants['organization_type'].dropna().unique()]),
        "country": sorted([str(x) for x in participants['country'].dropna().unique()]),
        "region": sorted([str(x) for x in participants[participants['country'] == 'Spain']['region'].dropna().unique()]),
        "province": sorted([str(x) for x in participants[participants['country'] == 'Spain']['province'].dropna().unique()]),
        "year_range": [
            int(projects['year_numeric'].min()), 
            int(projects['year_numeric'].max())
        ] if pd.notna(projects['year_numeric'].min()) else None
    }
    
    print("Recommended enum values for query schema:")
    for field, values in schema_recommendations.items():
        if field == "year_range":
            print(f"- {field}: {values}")
        else:
            print(f"- {field}: {len(values)} unique values")
            if len(values) <= 15:
                print(f"  Values: {values}")
            else:
                print(f"  Sample (first 10): {values[:10]}")
    
    # ========================================================================
    # SAVE ANALYSIS TO JSON
    # ========================================================================
    print("\n" + "="*80 + "\n")
    print(f"Saving analysis to '{output_file}'...")
    
    # Convert numpy/pandas types for JSON serialization
    def clean_for_json(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        else:
            return obj
    
    # Clean analysis and schema for JSON
    analysis_clean = json.loads(json.dumps(analysis, default=clean_for_json))
    schema_clean = json.loads(json.dumps(schema_recommendations, default=clean_for_json))
    
    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'analysis': analysis_clean,
            'schema_recommendations': schema_clean
        }, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Analysis saved to {output_path.resolve()}")
    print("\nAnalysis complete!")
    
    return analysis, schema_recommendations


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Analyze RIS3CAT parquet data for IMPULS project',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--data-dir',
        default='data/ris3cat',
        help='Directory containing project_db.parquet and participant_db.parquet'
    )
    parser.add_argument(
        '--output',
        default='ris3cat_analysis.json',
        help='Output JSON file path'
    )
    
    args = parser.parse_args()
    
    try:
        analysis, schema = analyze_ris3cat_data(
            data_dir=args.data_dir,
            output_file=args.output
        )
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
