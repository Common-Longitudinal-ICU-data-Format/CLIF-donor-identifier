#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SRTR Linkage Utility Functions
Provides functions for matching CLIF potential donors to SRTR actual donors
"""

import json
import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================
# Configuration and Constants
# ============================================

class SRTRLinkageConfig:
    """Configuration for SRTR linkage parameters"""

    # Matching tier definitions
    MATCH_TIERS = {
        1: {
            'name': 'Exact Match',
            'date_window_days': 3,
            'age_tolerance_years': 1,
            'require_exact_sex': True,
            'require_exact_race': True,
            'height_tolerance_cm': None,
            'weight_tolerance_kg': None,
            'creatinine_tolerance': None
        },
        2: {
            'name': 'Clinical Match',
            'date_window_days': 7,
            'age_tolerance_years': 2,
            'require_exact_sex': True,
            'require_exact_race': False,
            'height_tolerance_cm': 5,
            'weight_tolerance_kg': 5,
            'creatinine_tolerance': None
        },
        3: {
            'name': 'Composite Match',
            'date_window_days': 14,
            'age_tolerance_years': 3,
            'require_exact_sex': True,
            'require_exact_race': False,
            'height_tolerance_cm': 10,
            'weight_tolerance_kg': 10,
            'creatinine_tolerance': 0.5
        }
    }

    # SRTR variable mapping - only fields that actually exist in SRTR data
    SRTR_VARIABLES = {
        'donor_id': 'DONOR_ID',
        'age': 'DON_AGE',
        'sex': 'DON_GENDER',
        'race': 'DON_RACE_SRTR',
        'ethnicity': 'DON_ETHNICITY_SRTR',
        'height_cm': 'DON_HGT_CM',
        'weight_kg': 'DON_WGT_KG',
        'creatinine': 'DON_CREAT',
        'recovery_date': 'DON_RECOV_DT',
        'cause_of_death': 'DON_CAD_DON_COD',
        'diabetes': 'DON_HIST_DIAB',
        'hypertension': 'DON_HIST_HYPERTEN',
        'dcd_withdraw_date': 'DON_DCD_SUPPORT_WITHDRAW_DT'
    }


# ============================================
# Hospital Provider Mapping Functions
# ============================================

def load_provider_mapping(config_path: Path = None) -> Dict[str, Dict]:
    """
    Load hospital provider number mapping from JSON config

    Parameters:
    -----------
    config_path : Path
        Path to hospital_provider_mapping.json

    Returns:
    --------
    Dict mapping site codes to provider information
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / 'config' / 'hospital_provider_mapping.json'

    try:
        with open(config_path, 'r') as f:
            mapping = json.load(f)
        logger.info(f"Loaded provider mapping for {len(mapping)} sites")
        return mapping
    except FileNotFoundError:
        logger.error(f"Provider mapping file not found: {config_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing provider mapping JSON: {e}")
        raise


def get_provider_numbers(site_name: str, mapping: Dict = None) -> List[int]:
    """
    Get CMS provider numbers for a specific site

    Parameters:
    -----------
    site_name : str
        Site identifier (e.g., 'ucmc', 'upenn')
    mapping : Dict
        Provider mapping dictionary (loads if not provided)

    Returns:
    --------
    List of provider numbers for the site
    """
    if mapping is None:
        mapping = load_provider_mapping()

    site_lower = site_name.lower()
    if site_lower not in mapping:
        raise ValueError(f"Site '{site_name}' not found in provider mapping. Available sites: {list(mapping.keys())}")

    return mapping[site_lower]['provider_numbers']


# ============================================
# Data Standardization Functions
# ============================================

def standardize_clif_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    Standardize CLIF data for linkage with SRTR

    Parameters:
    -----------
    df : pl.DataFrame
        CLIF cohort dataframe

    Returns:
    --------
    Standardized dataframe with additional columns for matching
    """
    logger.info("Standardizing CLIF data for linkage...")

    # Create standardized columns for matching
    df = df.with_columns([
        # Extract date from datetime for death
        pl.col('final_death_dttm').dt.date().alias('death_date'),

        # Standardize sex to match SRTR format (M/F)
        pl.when(pl.col('sex_category').str.to_lowercase() == 'male')
        .then(pl.lit('M'))
        .when(pl.col('sex_category').str.to_lowercase() == 'female')
        .then(pl.lit('F'))
        .otherwise(None)
        .alias('sex_std'),

        # Use existing age_at_death (already calculated)
        pl.col('age_at_death').round().cast(pl.Int32).alias('age_years'),

        # Standardize race categories
        pl.col('race_category').str.to_uppercase().alias('race_std'),

        # Use last recorded height and weight
        pl.col('last_height_cm').alias('height_cm'),
        pl.col('last_weight_kg').alias('weight_kg'),

        # Use terminal creatinine
        pl.col('creatinine_value').alias('creatinine_mg_dl')
    ])

    # Log missing data statistics
    missing_stats = {
        'death_date': df['death_date'].null_count(),
        'sex': df['sex_std'].null_count(),
        'age': df['age_years'].null_count(),
        'height': df['height_cm'].null_count(),
        'weight': df['weight_kg'].null_count(),
        'creatinine': df['creatinine_mg_dl'].null_count()
    }

    logger.info(f"Missing data in CLIF: {missing_stats}")

    return df


def standardize_srtr_data(df: pl.DataFrame, config: SRTRLinkageConfig = None) -> pl.DataFrame:
    """
    Standardize SRTR data for linkage

    Parameters:
    -----------
    df : pl.DataFrame
        SRTR donor dataframe
    config : SRTRLinkageConfig
        Configuration object with variable mappings

    Returns:
    --------
    Standardized dataframe
    """
    if config is None:
        config = SRTRLinkageConfig()

    logger.info("Standardizing SRTR data for linkage...")

    # Rename columns to standard names (only for columns that exist)
    rename_map = {v: k for k, v in config.SRTR_VARIABLES.items() if v in df.columns}
    if rename_map:
        df = df.rename(rename_map)
        logger.info(f"Renamed {len(rename_map)} columns: {list(rename_map.keys())[:5]}...")

    # Standardize date formats
    if 'recovery_date' in df.columns:
        # Handle SAS date format if needed
        df = df.with_columns([
            pl.col('recovery_date').cast(pl.Date).alias('recovery_date')
        ])

    # Standardize sex to M/F
    if 'sex' in df.columns:
        df = df.with_columns([
            pl.when(pl.col('sex').str.contains('(?i)^m'))
            .then(pl.lit('M'))
            .when(pl.col('sex').str.contains('(?i)^f'))
            .then(pl.lit('F'))
            .otherwise(None)
            .alias('sex')
        ])

    # Convert age to integer
    if 'age' in df.columns:
        df = df.with_columns([
            pl.col('age').cast(pl.Int32)
        ])

    # Standardize race categories (uppercase)
    if 'race' in df.columns:
        df = df.with_columns([
            pl.col('race').str.to_uppercase()
        ])

    # Derive donation type from DCD withdraw date (like in donor_distribution.Rmd)
    if 'dcd_withdraw_date' in df.columns:
        df = df.with_columns([
            pl.when(pl.col('dcd_withdraw_date').is_not_null())
            .then(pl.lit('DCD'))
            .otherwise(pl.lit('DBD'))
            .alias('donation_type')
        ])
        logger.info("Derived donation_type from DCD withdraw date")

    return df


# ============================================
# Matching Functions
# ============================================

def calculate_date_difference(date1: pl.Date, date2: pl.Date) -> int:
    """Calculate absolute difference in days between two dates"""
    if date1 is None or date2 is None:
        return None
    return abs((date1 - date2).days)


def deterministic_match(
    clif_df: pl.DataFrame,
    srtr_df: pl.DataFrame,
    tier: int,
    config: SRTRLinkageConfig = None
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Perform deterministic matching at specified tier

    Parameters:
    -----------
    clif_df : pl.DataFrame
        CLIF potential donors (unmatched)
    srtr_df : pl.DataFrame
        SRTR actual donors
    tier : int
        Matching tier (1, 2, or 3)
    config : SRTRLinkageConfig
        Configuration object

    Returns:
    --------
    Tuple of (matched_df, unmatched_clif_df, unmatched_srtr_df)
    """
    if config is None:
        config = SRTRLinkageConfig()

    tier_config = config.MATCH_TIERS[tier]
    logger.info(f"Performing Tier {tier} matching: {tier_config['name']}")

    # Build matching conditions based on tier
    join_conditions = []

    # Date window condition
    date_window = tier_config['date_window_days']

    # Sex must always match exactly
    if tier_config['require_exact_sex']:
        join_conditions.append(pl.col('sex_std') == pl.col('sex'))

    # Tier-specific conditions
    if tier == 1:
        # Exact match: hospital + date ±3 + age ±1 + sex + race
        # Join on sex (both should have same column name after standardization)
        matched = clif_df.join(
            srtr_df.rename({'sex': 'sex_std'}),  # Rename SRTR sex to match CLIF
            how='inner',
            on='sex_std',
            suffix='_srtr'
        ).filter(
            # Date window
            (pl.col('recovery_date_srtr') - pl.col('death_date')).dt.days().abs() <= date_window
        ).filter(
            # Age tolerance
            (pl.col('age_years') - pl.col('age_srtr')).abs() <= tier_config['age_tolerance_years']
        )

        # Race must match exactly for tier 1
        if tier_config['require_exact_race']:
            matched = matched.filter(pl.col('race_std') == pl.col('race_srtr'))

    elif tier == 2:
        # Clinical match: hospital + date ±7 + age ±2 + sex + (height OR weight)
        matched = clif_df.join(
            srtr_df.rename({'sex': 'sex_std'}),  # Rename SRTR sex to match CLIF
            how='inner',
            on='sex_std',
            suffix='_srtr'
        ).filter(
            # Date window
            (pl.col('recovery_date_srtr') - pl.col('death_date')).dt.days().abs() <= date_window
        ).filter(
            # Age tolerance
            (pl.col('age_years') - pl.col('age_srtr')).abs() <= tier_config['age_tolerance_years']
        ).filter(
            # Height OR weight must be close
            ((pl.col('height_cm') - pl.col('height_cm_srtr')).abs() <= tier_config['height_tolerance_cm']) |
            ((pl.col('weight_kg') - pl.col('weight_kg_srtr')).abs() <= tier_config['weight_tolerance_kg'])
        )

    elif tier == 3:
        # Composite match: hospital + date ±14 + sex + (2 of 4 clinical variables)
        matched = clif_df.join(
            srtr_df.rename({'sex': 'sex_std'}),  # Rename SRTR sex to match CLIF
            how='inner',
            on='sex_std',
            suffix='_srtr'
        ).filter(
            # Date window
            (pl.col('recovery_date_srtr') - pl.col('death_date')).dt.days().abs() <= date_window
        ).with_columns([
            # Count how many clinical variables match
            (
                ((pl.col('age_years') - pl.col('age_srtr')).abs() <= tier_config['age_tolerance_years']).cast(pl.Int32) +
                ((pl.col('height_cm') - pl.col('height_cm_srtr')).abs() <= tier_config['height_tolerance_cm']).cast(pl.Int32) +
                ((pl.col('weight_kg') - pl.col('weight_kg_srtr')).abs() <= tier_config['weight_tolerance_kg']).cast(pl.Int32) +
                ((pl.col('creatinine_mg_dl') - pl.col('creatinine_srtr')).abs() <= tier_config['creatinine_tolerance']).cast(pl.Int32)
            ).alias('clinical_matches')
        ]).filter(
            # At least 2 of 4 clinical variables must match
            pl.col('clinical_matches') >= 2
        )

    # Add match tier and calculate match score
    if not matched.is_empty():
        matched = matched.with_columns([
            pl.lit(tier).alias('match_tier'),
            calculate_match_score_expr(tier).alias('match_score'),
            (pl.col('recovery_date_srtr') - pl.col('death_date')).dt.days().alias('date_diff_days')
        ])

    # Get unmatched records
    matched_clif_ids = matched.select('patient_id').unique() if not matched.is_empty() else pl.DataFrame({'patient_id': []})
    matched_srtr_ids = matched.select('donor_id_srtr').unique() if not matched.is_empty() else pl.DataFrame({'donor_id_srtr': []})

    unmatched_clif = clif_df.filter(~pl.col('patient_id').is_in(matched_clif_ids['patient_id']))
    unmatched_srtr = srtr_df.filter(~pl.col('donor_id').is_in(matched_srtr_ids['donor_id_srtr']))

    logger.info(f"Tier {tier} results: {len(matched)} matches, {len(unmatched_clif)} unmatched CLIF, {len(unmatched_srtr)} unmatched SRTR")

    return matched, unmatched_clif, unmatched_srtr


def calculate_match_score_expr(tier: int) -> pl.Expr:
    """
    Create expression to calculate match confidence score

    Parameters:
    -----------
    tier : int
        Matching tier

    Returns:
    --------
    Polars expression for match score calculation
    """
    # Base score by tier
    base_scores = {1: 0.9, 2: 0.7, 3: 0.5}
    base_score = base_scores[tier]

    # Adjust based on date difference (closer dates = higher score)
    # Max adjustment of ±0.1 based on date proximity
    date_adjustment = pl.when(pl.col('date_diff_days').abs() <= 1).then(0.1)\
                       .when(pl.col('date_diff_days').abs() <= 3).then(0.05)\
                       .when(pl.col('date_diff_days').abs() <= 7).then(0.0)\
                       .otherwise(-0.05)

    return pl.lit(base_score) + date_adjustment


def perform_tiered_linkage(
    clif_df: pl.DataFrame,
    srtr_df: pl.DataFrame,
    config: SRTRLinkageConfig = None
) -> pl.DataFrame:
    """
    Perform complete tiered linkage strategy

    Parameters:
    -----------
    clif_df : pl.DataFrame
        Standardized CLIF potential donors
    srtr_df : pl.DataFrame
        Standardized SRTR actual donors
    config : SRTRLinkageConfig
        Configuration object

    Returns:
    --------
    DataFrame with all matches from all tiers
    """
    if config is None:
        config = SRTRLinkageConfig()

    all_matches = []
    remaining_clif = clif_df
    remaining_srtr = srtr_df

    # Perform matching for each tier
    for tier in [1, 2, 3]:
        if remaining_clif.is_empty() or remaining_srtr.is_empty():
            break

        matched, remaining_clif, remaining_srtr = deterministic_match(
            remaining_clif, remaining_srtr, tier, config
        )

        if not matched.is_empty():
            all_matches.append(matched)

    # Combine all matches
    if all_matches:
        combined_matches = pl.concat(all_matches)
        logger.info(f"Total matches across all tiers: {len(combined_matches)}")
        return combined_matches
    else:
        logger.warning("No matches found across any tier")
        return pl.DataFrame()


# ============================================
# Validation Functions
# ============================================

def validate_matches(matched_df: pl.DataFrame) -> Dict[str, Any]:
    """
    Validate matched records for quality and consistency

    Parameters:
    -----------
    matched_df : pl.DataFrame
        DataFrame with matched records

    Returns:
    --------
    Dictionary with validation statistics
    """
    logger.info("Validating matches...")

    validation_stats = {}

    if matched_df.is_empty():
        return {'status': 'No matches to validate'}

    # Check for duplicate CLIF patients (one CLIF → multiple SRTR)
    clif_duplicates = matched_df.group_by('patient_id').agg([
        pl.count('donor_id_srtr').alias('n_matches')
    ]).filter(pl.col('n_matches') > 1)

    validation_stats['duplicate_clif_patients'] = len(clif_duplicates)

    if len(clif_duplicates) > 0:
        logger.warning(f"Found {len(clif_duplicates)} CLIF patients matched to multiple SRTR donors")

    # Check for duplicate SRTR donors (one SRTR → multiple CLIF)
    srtr_duplicates = matched_df.group_by('donor_id_srtr').agg([
        pl.count('patient_id').alias('n_matches')
    ]).filter(pl.col('n_matches') > 1)

    validation_stats['duplicate_srtr_donors'] = len(srtr_duplicates)

    if len(srtr_duplicates) > 0:
        logger.warning(f"Found {len(srtr_duplicates)} SRTR donors matched to multiple CLIF patients")

    # Check temporal consistency
    if 'date_diff_days' in matched_df.columns:
        date_stats = matched_df.select([
            pl.col('date_diff_days').abs().mean().alias('mean_date_diff'),
            pl.col('date_diff_days').abs().median().alias('median_date_diff'),
            pl.col('date_diff_days').abs().max().alias('max_date_diff')
        ]).to_dicts()[0]

        validation_stats.update(date_stats)

        # Flag suspicious matches (recovery >30 days from death)
        suspicious = matched_df.filter(pl.col('date_diff_days').abs() > 30)
        validation_stats['suspicious_date_matches'] = len(suspicious)

        if len(suspicious) > 0:
            logger.warning(f"Found {len(suspicious)} matches with >30 day difference between death and recovery")

    # Match quality by tier
    tier_stats = matched_df.group_by('match_tier').agg([
        pl.count().alias('n_matches'),
        pl.col('match_score').mean().alias('avg_score'),
        pl.col('match_score').min().alias('min_score'),
        pl.col('match_score').max().alias('max_score')
    ]).sort('match_tier')

    validation_stats['tier_distribution'] = tier_stats.to_dicts()

    logger.info(f"Validation complete: {validation_stats}")

    return validation_stats


def resolve_duplicate_matches(matched_df: pl.DataFrame) -> pl.DataFrame:
    """
    Resolve duplicate matches by keeping the best match for each patient

    Parameters:
    -----------
    matched_df : pl.DataFrame
        DataFrame with potential duplicate matches

    Returns:
    --------
    DataFrame with duplicates resolved
    """
    if matched_df.is_empty():
        return matched_df

    # For each CLIF patient, keep the match with:
    # 1. Lowest tier number (tier 1 is best)
    # 2. Highest match score within tier
    # 3. Smallest date difference as tiebreaker

    resolved = matched_df.sort([
        'patient_id',
        'match_tier',
        pl.col('match_score').reverse(),
        pl.col('date_diff_days').abs()
    ]).group_by('patient_id').agg([
        pl.first('*')
    ])

    # Flatten the result
    resolved = resolved.select([
        pl.col('patient_id').list.first(),
        pl.all().exclude('patient_id').list.first()
    ])

    logger.info(f"Resolved duplicates: {len(matched_df)} → {len(resolved)} unique matches")

    return resolved


# ============================================
# Analysis Functions
# ============================================

def calculate_conversion_rates(
    clif_df: pl.DataFrame,
    matched_df: pl.DataFrame,
    by_demographics: bool = True
) -> Dict[str, Any]:
    """
    Calculate donor conversion rates

    Parameters:
    -----------
    clif_df : pl.DataFrame
        All CLIF potential donors
    matched_df : pl.DataFrame
        Successfully matched donors
    by_demographics : bool
        Whether to calculate rates by demographic groups

    Returns:
    --------
    Dictionary with conversion rate statistics
    """
    stats = {}

    # Overall statistics
    total_clif = len(clif_df)
    total_matched = len(matched_df)

    # By eligibility type
    calc_eligible = clif_df.filter(pl.col('calc_flag')).height
    clif_eligible = clif_df.filter(pl.col('clif_eligible_donors')).height

    calc_matched = matched_df.filter(pl.col('calc_flag')).height if not matched_df.is_empty() else 0
    clif_matched = matched_df.filter(pl.col('clif_eligible_donors')).height if not matched_df.is_empty() else 0

    stats['overall'] = {
        'potential_donors': total_clif,
        'actual_donors': total_matched,
        'conversion_rate': total_matched / total_clif if total_clif > 0 else 0
    }

    stats['calc'] = {
        'potential_donors': calc_eligible,
        'actual_donors': calc_matched,
        'conversion_rate': calc_matched / calc_eligible if calc_eligible > 0 else 0
    }

    stats['clif'] = {
        'potential_donors': clif_eligible,
        'actual_donors': clif_matched,
        'conversion_rate': clif_matched / clif_eligible if clif_eligible > 0 else 0
    }

    if by_demographics and not matched_df.is_empty():
        # Age groups
        age_groups = [
            ('18-39', 18, 39),
            ('40-54', 40, 54),
            ('55-64', 55, 64),
            ('65-75', 65, 75)
        ]

        stats['by_age'] = {}
        for label, min_age, max_age in age_groups:
            age_potential = clif_df.filter(
                (pl.col('age_years') >= min_age) & (pl.col('age_years') <= max_age)
            ).height
            age_matched = matched_df.filter(
                (pl.col('age_years') >= min_age) & (pl.col('age_years') <= max_age)
            ).height

            stats['by_age'][label] = {
                'potential': age_potential,
                'matched': age_matched,
                'rate': age_matched / age_potential if age_potential > 0 else 0
            }

        # By sex
        for sex in ['M', 'F']:
            sex_potential = clif_df.filter(pl.col('sex_std') == sex).height
            sex_matched = matched_df.filter(pl.col('sex_std') == sex).height

            sex_label = 'Male' if sex == 'M' else 'Female'
            stats[f'sex_{sex_label}'] = {
                'potential': sex_potential,
                'matched': sex_matched,
                'rate': sex_matched / sex_potential if sex_potential > 0 else 0
            }

    return stats


def generate_linkage_report(
    clif_df: pl.DataFrame,
    srtr_df: pl.DataFrame,
    matched_df: pl.DataFrame,
    validation_stats: Dict,
    conversion_stats: Dict,
    site_name: str
) -> pd.DataFrame:
    """
    Generate comprehensive linkage report

    Parameters:
    -----------
    clif_df : pl.DataFrame
        All CLIF potential donors
    srtr_df : pl.DataFrame
        All SRTR donors for site
    matched_df : pl.DataFrame
        Successfully matched donors
    validation_stats : Dict
        Validation statistics
    conversion_stats : Dict
        Conversion rate statistics
    site_name : str
        Site identifier

    Returns:
    --------
    DataFrame with linkage report
    """
    report_data = []

    # Basic counts
    report_data.extend([
        {'Metric': 'Site', 'Value': site_name},
        {'Metric': 'Total CLIF Potential Donors', 'Value': len(clif_df)},
        {'Metric': 'CALC Eligible', 'Value': clif_df.filter(pl.col('calc_flag')).height},
        {'Metric': 'CLIF Eligible', 'Value': clif_df.filter(pl.col('clif_eligible_donors')).height},
        {'Metric': 'Total SRTR Donors at Site', 'Value': len(srtr_df)},
        {'Metric': 'Successfully Matched', 'Value': len(matched_df) if matched_df is not None else 0}
    ])

    # Match distribution by tier
    if matched_df is not None and not matched_df.is_empty():
        tier_counts = matched_df.group_by('match_tier').count().sort('match_tier')
        for row in tier_counts.to_dicts():
            report_data.append({
                'Metric': f"Tier {row['match_tier']} Matches",
                'Value': row['count']
            })

    # Conversion rates
    if conversion_stats:
        report_data.extend([
            {'Metric': 'Overall Conversion Rate', 'Value': f"{conversion_stats['overall']['conversion_rate']:.1%}"},
            {'Metric': 'CALC Conversion Rate', 'Value': f"{conversion_stats['calc']['conversion_rate']:.1%}"},
            {'Metric': 'CLIF Conversion Rate', 'Value': f"{conversion_stats['clif']['conversion_rate']:.1%}"}
        ])

    # Validation metrics
    if validation_stats:
        if 'duplicate_clif_patients' in validation_stats:
            report_data.append({
                'Metric': 'Duplicate CLIF Patients',
                'Value': validation_stats['duplicate_clif_patients']
            })
        if 'mean_date_diff' in validation_stats:
            report_data.append({
                'Metric': 'Mean Date Difference (days)',
                'Value': f"{validation_stats['mean_date_diff']:.1f}"
            })

    return pd.DataFrame(report_data)


# ============================================
# Main Linkage Function
# ============================================

def link_srtr_to_clif(
    clif_df: pl.DataFrame,
    srtr_df: pl.DataFrame,
    site_name: str,
    config: SRTRLinkageConfig = None
) -> Dict[str, Any]:
    """
    Main function to perform complete SRTR to CLIF linkage

    Parameters:
    -----------
    clif_df : pl.DataFrame
        CLIF potential donors
    srtr_df : pl.DataFrame
        SRTR actual donors
    site_name : str
        Site identifier
    config : SRTRLinkageConfig
        Configuration object

    Returns:
    --------
    Dictionary with:
        - enhanced_df: CLIF data with SRTR matches
        - matched_df: Just the matched records
        - validation_stats: Validation statistics
        - conversion_stats: Conversion rate statistics
        - report_df: Summary report
    """
    if config is None:
        config = SRTRLinkageConfig()

    logger.info(f"Starting SRTR linkage for site: {site_name}")
    logger.info(f"Input: {len(clif_df)} CLIF donors, {len(srtr_df)} SRTR donors")

    # Standardize data
    clif_std = standardize_clif_data(clif_df)
    srtr_std = standardize_srtr_data(srtr_df, config)

    # Perform tiered matching
    matched_df = perform_tiered_linkage(clif_std, srtr_std, config)

    # Resolve duplicates if any
    if not matched_df.is_empty():
        matched_df = resolve_duplicate_matches(matched_df)

    # Validate matches
    validation_stats = validate_matches(matched_df)

    # Calculate conversion rates
    conversion_stats = calculate_conversion_rates(clif_std, matched_df)

    # Enhance CLIF dataset with SRTR data
    if not matched_df.is_empty():
        # Select SRTR columns to add
        srtr_cols_to_add = [
            'patient_id',
            'donor_id_srtr',  # Will be renamed to donor_id after join
            'match_tier',
            'match_score',
            'date_diff_days',
            'recovery_date_srtr',  # Will be renamed to recovery_date after join
            'cause_of_death_srtr',  # Will be renamed to cause_of_death after join
            'donation_type_srtr',  # Will be renamed to donation_type after join
            'diabetes_srtr',  # Will be renamed to diabetes after join
            'hypertension_srtr'  # Will be renamed to hypertension after join
        ]

        # Keep only necessary columns from matches
        match_subset = matched_df.select([
            col for col in srtr_cols_to_add
            if col in matched_df.columns
        ])

        # Rename _srtr columns to clean names
        rename_mapping = {
            'donor_id_srtr': 'donor_id',
            'recovery_date_srtr': 'recovery_date',
            'cause_of_death_srtr': 'cause_of_death',
            'donation_type_srtr': 'donation_type',
            'diabetes_srtr': 'diabetes',
            'hypertension_srtr': 'hypertension'
        }

        for old_name, new_name in rename_mapping.items():
            if old_name in match_subset.columns:
                match_subset = match_subset.rename({old_name: new_name})

        # Left join to preserve all CLIF records
        enhanced_df = clif_df.join(
            match_subset,
            on='patient_id',
            how='left'
        ).with_columns([
            pl.col('donor_id').is_not_null().alias('srtr_matched')
        ])
    else:
        # No matches found
        enhanced_df = clif_df.with_columns([
            pl.lit(False).alias('srtr_matched'),
            pl.lit(None).alias('donor_id'),
            pl.lit(None).alias('match_tier'),
            pl.lit(None).alias('match_score')
        ])

    # Generate report
    report_df = generate_linkage_report(
        clif_std, srtr_std, matched_df,
        validation_stats, conversion_stats, site_name
    )

    logger.info(f"Linkage complete: {len(matched_df) if matched_df is not None else 0} matches found")

    return {
        'enhanced_df': enhanced_df,
        'matched_df': matched_df,
        'validation_stats': validation_stats,
        'conversion_stats': conversion_stats,
        'report_df': report_df
    }