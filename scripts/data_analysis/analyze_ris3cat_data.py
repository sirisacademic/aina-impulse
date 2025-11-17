import pandas as pd
import numpy as np
from datetime import datetime
import json

def analyze_ris3cat_data(filepath='R3C_data.pkl'):
    """
    Comprehensive analysis of RIS3CAT data to inform query schema design
    """
    
    # Load data
    print("Loading RIS3CAT data...")
    df = pd.read_pickle(filepath)
    print(f"Dataset shape: {df.shape}")
    print(f"Unique projects: {df['projectId'].nunique()}")
    print(f"Unique organizations: {df['organizationId'].nunique()}")
    print("\n" + "="*80 + "\n")
    
    # Store results
    analysis = {}
    
    # 1. PROGRAMME/FRAMEWORK ANALYSIS
    print("1. PROGRAMME/FRAMEWORK ANALYSIS")
    print("-" * 40)
    
    analysis['programmes'] = {
        'frameworkName': {
            'unique_values': df['frameworkName'].nunique(),
            'values': df['frameworkName'].value_counts().to_dict(),
            'nulls': df['frameworkName'].isna().sum()
        },
        'instrumentName': {
            'unique_values': df['instrumentName'].nunique(),
            'all': df['instrumentName'].value_counts().to_dict(),
            'nulls': df['instrumentName'].isna().sum()
        }
    }
    
    print(f"Framework names ({df['frameworkName'].nunique()} unique):")
    print(df['frameworkName'].value_counts())
    print(f"\nTop 10 Instruments ({df['instrumentName'].nunique()} unique):")
    print(df['instrumentName'].value_counts())
    print("\n" + "="*80 + "\n")
    
    # 2. ENTITY TYPE ANALYSIS
    print("2. ENTITY TYPE ANALYSIS")
    print("-" * 40)
    
    analysis['entity_types'] = {
        'organizationTypeId': {
            'unique_values': df['organizationTypeId'].nunique(),
            'values': df['organizationTypeId'].value_counts().to_dict(),
            'nulls': df['organizationTypeId'].isna().sum()
        }
    }
    
    print(f"Organization types ({df['organizationTypeId'].nunique()} unique):")
    print(df['organizationTypeId'].value_counts())
    print("\n" + "="*80 + "\n")
    
    # 3. GEOGRAPHIC ANALYSIS
    print("3. GEOGRAPHIC ANALYSIS")
    print("-" * 40)
    
    # Countries
    analysis['geography'] = {
        'countries': {
            'unique_values': df['organizationCountry'].nunique(),
            'all': df['organizationCountry'].value_counts().to_dict(),
            'nulls': df['organizationCountry'].isna().sum()
        },
        'nuts2': {
            'unique_values': df['organizationNuts2Name'].nunique(),
            'catalan_regions': df[df['organizationCountry'] == 'Spain']['organizationNuts2Name'].value_counts().to_dict(),
            'nulls': df['organizationNuts2Name'].isna().sum()
        },
        'nuts3': {
            'unique_values': df['organizationNuts3Name'].nunique(),
            'catalan_provinces': df[df['organizationCountry'] == 'Spain']['organizationNuts3Name'].value_counts().to_dict(),
            'nulls': df['organizationNuts3Name'].isna().sum()
        }
    }

    print(f"Top 20 Countries ({df['organizationCountry'].nunique()} unique):")
    print(df['organizationCountry'].value_counts().head(20))

    print(f"\nAll Countries ({df['organizationCountry'].nunique()} unique):")
    print(df['organizationCountry'].value_counts())
    
    print(f"\nSpanish NUTS2 regions:")
    spanish_nuts2 = df[df['organizationCountry'] == 'Spain']['organizationNuts2Name'].value_counts()
    print(spanish_nuts2)
    
    print(f"\nSpanish NUTS3 provinces:")
    spanish_nuts3 = df[df['organizationCountry'] == 'Spain']['organizationNuts3Name'].value_counts()
    print(spanish_nuts3)
    print("\n" + "="*80 + "\n")
    
    # 4. THEMATIC/RIS3CAT CLASSIFICATION
    print("4. THEMATIC/RIS3CAT CLASSIFICATION")
    print("-" * 40)
    
    analysis['thematic'] = {
        'ambit_sectorial': {
            'unique_values': df['RIS3CAT Àmbit Sectorial Líder'].nunique(),
            'values': df['RIS3CAT Àmbit Sectorial Líder'].value_counts().to_dict(),
            'nulls': df['RIS3CAT Àmbit Sectorial Líder'].isna().sum()
        },
        'tecnologia': {
            'unique_values': df['RIS3CAT Tecnologia Facilitadora Transversal'].nunique(),
            'values': df['RIS3CAT Tecnologia Facilitadora Transversal'].value_counts().to_dict(),
            'nulls': df['RIS3CAT Tecnologia Facilitadora Transversal'].isna().sum()
        }
    }
    
    print(f"RIS3CAT Àmbit Sectorial ({df['RIS3CAT Àmbit Sectorial Líder'].nunique()} unique):")
    print(df['RIS3CAT Àmbit Sectorial Líder'].value_counts())
    
    print(f"\nRIS3CAT Tecnologia ({df['RIS3CAT Tecnologia Facilitadora Transversal'].nunique()} unique):")
    print(df['RIS3CAT Tecnologia Facilitadora Transversal'].value_counts())
    print("\n" + "="*80 + "\n")
    
    # 5. SDG ANALYSIS
    print("5. SDG ANALYSIS")
    print("-" * 40)
    
    analysis['sdg'] = {
        'unique_values': df['sdgName'].nunique(),
        'values': df['sdgName'].value_counts().to_dict() if df['sdgName'].notna().any() else {},
        'nulls': df['sdgName'].isna().sum(),
        'coverage': (df['sdgName'].notna().sum() / len(df)) * 100
    }
    
    print(f"SDG coverage: {(df['sdgName'].notna().sum() / len(df)) * 100:.1f}%")
    if df['sdgName'].notna().any():
        print(f"SDG distribution ({df['sdgName'].nunique()} unique):")
        print(df['sdgName'].value_counts())
    else:
        print("No SDG data available")
    print("\n" + "="*80 + "\n")
    
    # 6. TEMPORAL ANALYSIS
    print("6. TEMPORAL ANALYSIS")
    print("-" * 40)
    
    analysis['temporal'] = {
        'year_range': {
            'min': int(df['startingYear'].min()) if pd.notna(df['startingYear'].min()) else None,
            'max': int(df['startingYear'].max()) if pd.notna(df['startingYear'].max()) else None
        },
        'year_distribution': df['startingYear'].value_counts().sort_index().to_dict()
    }
    
    print(f"Year range: {df['startingYear'].min():.0f} - {df['startingYear'].max():.0f}")
    print(f"Projects by year:")
    print(df.groupby('startingYear')['projectId'].nunique().sort_index())
    print("\n" + "="*80 + "\n")
    
    # 7. COLLABORATION ANALYSIS
    print("7. COLLABORATION ANALYSIS")
    print("-" * 40)
    
    # Projects with multiple partners
    partners_per_project = df.groupby('projectId')['organizationId'].nunique()
    
    analysis['collaboration'] = {
        'avg_partners': partners_per_project.mean(),
        'max_partners': partners_per_project.max(),
        'single_partner_projects': (partners_per_project == 1).sum(),
        'multi_partner_projects': (partners_per_project > 1).sum(),
        'international_projects': df.groupby('projectId')['organizationCountry'].nunique().gt(1).sum()
    }
    
    print(f"Average partners per project: {partners_per_project.mean():.2f}")
    print(f"Max partners in a project: {partners_per_project.max()}")
    print(f"Single-partner projects: {(partners_per_project == 1).sum()}")
    print(f"Multi-partner projects: {(partners_per_project > 1).sum()}")
    
    # International collaborations
    countries_per_project = df.groupby('projectId')['organizationCountry'].nunique()
    print(f"International projects (>1 country): {(countries_per_project > 1).sum()}")
    
    # Coordination info
    print(f"\nCoordination roles: {df['coordination'].sum()} out of {len(df)} partnerships")
    print("\n" + "="*80 + "\n")
    
    # 8. FINANCIAL ANALYSIS
    print("8. FINANCIAL ANALYSIS")
    print("-" * 40)
    
    # Get unique projects for financial analysis
    projects_df = df.groupby('projectId').first()
    
    analysis['financial'] = {
        'total_investment': {
            'min': projects_df['totalInvestment'].min(),
            'max': projects_df['totalInvestment'].max(),
            'mean': projects_df['totalInvestment'].mean(),
            'median': projects_df['totalInvestment'].median()
        },
        'total_grant': {
            'min': projects_df['totalGrant'].min(),
            'max': projects_df['totalGrant'].max(),
            'mean': projects_df['totalGrant'].mean(),
            'median': projects_df['totalGrant'].median()
        }
    }
    
    print(f"Total Investment range: €{projects_df['totalInvestment'].min():,.0f} - €{projects_df['totalInvestment'].max():,.0f}")
    print(f"Average investment: €{projects_df['totalInvestment'].mean():,.0f}")
    print(f"Median investment: €{projects_df['totalInvestment'].median():,.0f}")
    print("\n" + "="*80 + "\n")
    
    # 9. TEXT FIELDS ANALYSIS
    print("9. TEXT FIELDS ANALYSIS")
    print("-" * 40)
    
    # Sample abstracts for language detection
    sample_abstracts = df[df['projectAbstract'].notna()]['projectAbstract'].sample(min(100, df['projectAbstract'].notna().sum()))
    
    # Simple language detection based on common words
    lang_indicators = {
        'en': ['the', 'and', 'of', 'to', 'in', 'for', 'with'],
        'es': ['el', 'la', 'de', 'en', 'por', 'para', 'con'],
        'ca': ['el', 'la', 'de', 'per', 'amb', 'dels', 'les']
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
    
    analysis['text_fields'] = {
        'abstracts_available': df['projectAbstract'].notna().sum(),
        'titles_available': df['projectTitle'].notna().sum(),
        'language_sample': lang_counts,
        'avg_abstract_length': df[df['projectAbstract'].notna()]['projectAbstract'].str.len().mean()
    }
    
    print(f"Projects with abstracts: {df['projectAbstract'].notna().sum()} / {df['projectId'].nunique()}")
    print(f"Projects with titles: {df['projectTitle'].notna().sum()} / {df['projectId'].nunique()}")
    print(f"Language detection (sample of {len(sample_abstracts)}): {lang_counts}")
    print(f"Average abstract length: {df[df['projectAbstract'].notna()]['projectAbstract'].str.len().mean():.0f} characters")
    print("\n" + "="*80 + "\n")
    
    # 10. GENERATE SCHEMA RECOMMENDATIONS
    print("10. SCHEMA RECOMMENDATIONS FOR QUERY PARSING")
    print("-" * 40)
    
    schema_recommendations = {
        "programme": list(df['frameworkName'].dropna().unique()),
        "entity_type": list(df['organizationTypeId'].dropna().unique()),
        "country": list(df['organizationCountry'].value_counts().index),  # Top 50 countries
        "nuts2_region": list(df[df['organizationCountry'] == 'Spain']['organizationNuts2Name'].dropna().unique()),
        "nuts3_province": list(df[df['organizationCountry'] == 'Spain']['organizationNuts3Name'].dropna().unique()),
        "ambit_sectorial": [x for x in df['RIS3CAT Àmbit Sectorial Líder'].dropna().unique() if x != 'Sense classificar per àmbit sectorial'],
        "tecnologia": [x for x in df['RIS3CAT Tecnologia Facilitadora Transversal'].dropna().unique() if x != 'Sense classificar per tecnologia'],
        "year_range": [int(df['startingYear'].min()), int(df['startingYear'].max())] if pd.notna(df['startingYear'].min()) else None
    }
    
    print("Recommended enum values for schema:")
    for field, values in schema_recommendations.items():
        if field == "year_range":
            print(f"- {field}: {values}")
        else:
            print(f"- {field}: {len(values)} unique values")
            if len(values) <= 10:
                print(f"  Values: {values}")
    
    # Save analysis to JSON
    print("\n" + "="*80 + "\n")
    print("Saving analysis to 'ris3cat_analysis.json'...")
    
    # Convert numpy/pandas types for JSON serialization
    def clean_for_json(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    analysis_clean = json.loads(json.dumps(analysis, default=clean_for_json))
    schema_clean = json.loads(json.dumps(schema_recommendations, default=clean_for_json))
    
    with open('ris3cat_analysis.json', 'w', encoding='utf-8') as f:
        json.dump({
            'analysis': analysis_clean,
            'schema_recommendations': schema_clean
        }, f, ensure_ascii=False, indent=2)
    
    print("Analysis complete!")
    
    return analysis, schema_recommendations

# Run the analysis
if __name__ == "__main__":
    analysis, schema = analyze_ris3cat_data('../../data/R3C_data.pkl')
