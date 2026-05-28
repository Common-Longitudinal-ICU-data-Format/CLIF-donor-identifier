"""
Table 1 utility for creating baseline characteristics summary tables.

This module provides functions to create Table 1 summarizing baseline
characteristics across Overall, CALC Donors, and CLIF Donors cohorts.
"""

import polars as pl
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple


def create_table_one(final_cohort_df: pl.DataFrame, output_dir: str = 'output') -> pl.DataFrame:
    """
    Create Table 1 summarizing baseline characteristics across cohorts.

    Parameters
    ----------
    final_cohort_df : pl.DataFrame
        Final cohort dataframe with all variables
    output_dir : str
        Directory to save CSV output

    Returns
    -------
    pl.DataFrame
        Table 1 with characteristics for Overall, CALC, and CLIF cohorts
    """

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ============================================
    # Define cohorts
    # ============================================
    # NIDDK Aim 1 (PDF 1) "Died_While_IMV" stratum uses the upstream
    # `died_while_imv` flag = age <=75 AND IMV <=48 h before death.
    # This is the same flag the main script registers in strobe_counts and
    # uses as the Table 2 denominator, so all three stay in lockstep.
    # Fall back to recomputing on-the-fly if older runs lack the column.
    cohort_overall = final_cohort_df
    if 'died_while_imv' in final_cohort_df.columns:
        cohort_died_imv = final_cohort_df.filter(pl.col('died_while_imv'))
    elif 'imv_48hr_expire' in final_cohort_df.columns and 'age_75_less' in final_cohort_df.columns:
        cohort_died_imv = final_cohort_df.filter(
            pl.col('imv_48hr_expire') & pl.col('age_75_less')
        )
    else:
        cohort_died_imv = final_cohort_df.head(0)
    cohort_calc = final_cohort_df.filter(pl.col('calc_flag'))
    cohort_clif = final_cohort_df.filter(pl.col('clif_eligible_donors'))

    cohorts = {
        'Overall': cohort_overall,
        'Died_While_IMV': cohort_died_imv,
        'CALC_Donors': cohort_calc,
        'CLIF_Donors': cohort_clif,
    }

    # Get cohort sizes for display
    cohort_sizes = {
        name: f"{name} (n={cohort_df['patient_id'].n_unique()})"
        for name, cohort_df in cohorts.items()
    }

    # ============================================
    # Variables to summarize
    # ============================================
    categorical_vars = [
        'race_category',
        'ethnicity_category',
        'sex_category',
        'first_admission_location',
        # Comorbidities (NIDDK Aim 1 PDF 1 — extracted from hospital_diagnosis)
        'icd10_hcv',
        'icd10_htn',
        'icd10_dm',
        'icd10_cva',
        # Combined organ eligibility criteria
        'kidney_eligible',  # Cr <4 AND not on CRRT
        'liver_eligible',   # Bili <4 AND AST <700 AND ALT <700
        'on_crrt_48h_before_death',  # CRRT status
        # Terminal lab thresholds (all patients)
        'creatinine_lt_4',
        'bilirubin_lt_4',
        'ast_lt_700',
        'alt_lt_700',
        # Terminal lab thresholds (BMI ≤50 only)
        'creatinine_lt_4_bmi50',
        'bilirubin_lt_4_bmi50',
        'ast_lt_700_bmi50',
        'alt_lt_700_bmi50'
    ]

    numerical_vars = [
        'age_at_death',
        'hospital_length_of_stay_days',
        'first_icu_los_days',
        'last_height_cm',
        'last_weight_kg',
        'bmi',
        'creatinine_value',
        'bilirubin_total_value',
        'ast_value',
        'alt_value',
        'rass_value',
        'gcs_total_value'
    ]

    # Override the default Title Case label for variables that need a
    # specific human-readable name (units, capitalization).
    NUMERICAL_LABELS = {
        'last_height_cm': 'Height (cm)',
        'last_weight_kg': 'Weight (kg)',
    }

    # ============================================
    # Build Table 1 data
    # ============================================
    summary_data = []

    # Unique patients and encounters — emit a column per cohort dynamically
    summary_data.append({
        'Variable': 'Unique Patients',
        'Category': '',
        **{name: df['patient_id'].n_unique() for name, df in cohorts.items()},
    })
    summary_data.append({
        'Variable': 'Unique Encounters',
        'Category': '',
        **{name: len(df) for name, df in cohorts.items()},
    })

    # Data collection period (min-max years from admission_dttm)
    year_ranges = {}
    for cohort_name, cohort_df in cohorts.items():
        if 'admission_dttm' in cohort_df.columns:
            # Get min and max years from admission datetime
            admission_years = cohort_df.select(
                pl.col('admission_dttm').dt.year()
            ).to_series().drop_nulls()

            if len(admission_years) > 0:
                min_year = int(admission_years.min())
                max_year = int(admission_years.max())
                if min_year == max_year:
                    year_ranges[cohort_name] = str(min_year)
                else:
                    year_ranges[cohort_name] = f"{min_year} - {max_year}"
            else:
                year_ranges[cohort_name] = "N/A"
        else:
            year_ranges[cohort_name] = "N/A"

    # Debug print to see what we calculated
    print(f"Data Collection Period - Year ranges calculated: {year_ranges}")

    summary_data.append({
        'Variable': 'Data Collection Period',
        'Category': '',
        **{name: year_ranges.get(name, 'N/A') for name in cohorts},
    })

    # Debug print to verify it was added
    print(f"Data Collection Period row added to summary_data")

    # Custom labels for boolean/threshold variables
    custom_labels = {
        # Comorbidities (from hospital_diagnosis ICD-10 prefix match)
        'icd10_hcv': 'HCV infection',
        'icd10_htn': 'Hypertension',
        'icd10_dm': 'Diabetes',
        'icd10_cva': 'Hx CVA',
        # Combined eligibility
        'kidney_eligible': 'Kidney Eligible (Cr <4 AND not on CRRT)',
        'liver_eligible': 'Liver Eligible (Bili <4 AND AST <700 AND ALT <700)',
        'on_crrt_48h_before_death': 'On CRRT within 48h of death',
        # Terminal thresholds (all patients)
        'creatinine_lt_4': 'Terminal Creatinine < 4',
        'bilirubin_lt_4': 'Terminal Bilirubin < 4',
        'ast_lt_700': 'Terminal AST < 700',
        'alt_lt_700': 'Terminal ALT < 700',
        # Terminal thresholds (BMI ≤50 only)
        'creatinine_lt_4_bmi50': 'Terminal Creatinine < 4 (BMI ≤50)',
        'bilirubin_lt_4_bmi50': 'Terminal Bilirubin < 4 (BMI ≤50)',
        'ast_lt_700_bmi50': 'Terminal AST < 700 (BMI ≤50)',
        'alt_lt_700_bmi50': 'Terminal ALT < 700 (BMI ≤50)'
    }

    # ============================================
    # Categorical variables
    # ============================================
    for var_name in categorical_vars:
        # Check if this variable exists in the dataframe
        if var_name not in final_cohort_df.columns:
            print(f"Warning: Variable {var_name} not found in dataframe, skipping...")
            continue

        # Get unique categories
        all_categories = final_cohort_df.select(var_name).unique().to_series().to_list()

        # Check if this is a boolean flag
        is_boolean = all(v in [True, False, None] for v in all_categories if v is not None)

        if is_boolean:
            # For boolean flags, only show True counts
            var_label = custom_labels.get(var_name, var_name.replace('_', ' ').title())
            row = {'Variable': var_label, 'Category': ''}

            for cohort_name, cohort_df in cohorts.items():
                n = cohort_df.filter(pl.col(var_name) == True).shape[0]
                total = cohort_df.shape[0]
                pct = (n / total * 100) if total > 0 else 0
                row[cohort_name.replace(' ', '_')] = f"{n} ({pct:.1f}%)"

            summary_data.append(row)
        else:
            # Original categorical processing
            categories = sorted([c for c in all_categories if c is not None])

            for category in categories:
                row = {'Variable': var_name.replace('_', ' ').title(), 'Category': str(category)}

                for cohort_name, cohort_df in cohorts.items():
                    n = cohort_df.filter(pl.col(var_name) == category).shape[0]
                    total = cohort_df.shape[0]
                    pct = (n / total * 100) if total > 0 else 0
                    row[cohort_name.replace(' ', '_')] = f"{n} ({pct:.1f}%)"

                summary_data.append(row)

    # ============================================
    # Numerical variables
    # Each variable produces a Median (IQR) row and an N valid row.
    # Variables in MEAN_SD_VARS additionally produce a Mean (SD) row (PDF 1
    # spec asks for mean+/-SD for Age — we report both for transparency).
    # ============================================
    MEAN_SD_VARS = {'age_at_death'}

    for var_name in numerical_vars:
        var_label = NUMERICAL_LABELS.get(var_name, var_name.replace('_', ' ').title())

        # Median (IQR) row
        row_median = {'Variable': var_label, 'Category': 'Median (IQR)'}
        for cohort_name, cohort_df in cohorts.items():
            values = cohort_df.select(var_name).to_series().drop_nulls()
            if len(values) > 0:
                median = float(values.median())
                q1 = float(values.quantile(0.25))
                q3 = float(values.quantile(0.75))
                row_median[cohort_name.replace(' ', '_')] = f"{median:.1f} ({q1:.1f}-{q3:.1f})"
            else:
                row_median[cohort_name.replace(' ', '_')] = "N/A"
        summary_data.append(row_median)

        # Mean (SD) row — only for variables in MEAN_SD_VARS
        if var_name in MEAN_SD_VARS:
            row_mean = {'Variable': var_label, 'Category': 'Mean (SD)'}
            for cohort_name, cohort_df in cohorts.items():
                values = cohort_df.select(var_name).to_series().drop_nulls()
                if len(values) > 0:
                    mean = float(values.mean())
                    sd = float(values.std())
                    row_mean[cohort_name.replace(' ', '_')] = f"{mean:.1f} ({sd:.1f})"
                else:
                    row_mean[cohort_name.replace(' ', '_')] = "N/A"
            summary_data.append(row_mean)

        # N valid (missing) row
        row_missing = {'Variable': var_label, 'Category': 'N valid (% available)'}
        for cohort_name, cohort_df in cohorts.items():
            values = cohort_df.select(var_name).to_series().drop_nulls()
            n_total = cohort_df.shape[0]
            n_valid = len(values)
            pct_valid = (n_valid / n_total * 100) if n_total > 0 else 0
            n_missing = n_total - n_valid
            row_missing[cohort_name.replace(' ', '_')] = f"{n_valid} ({pct_valid:.1f}%), {n_missing} missing"
        summary_data.append(row_missing)

    # ============================================
    # Convert to DataFrame
    # ============================================
    table_one_df = pl.DataFrame(summary_data)

    # Convert to pandas for export and printing
    table_one_pd = table_one_df.to_pandas()

    # Debug: Check if Data Collection Period is in the dataframe
    data_collection_rows = table_one_df.filter(pl.col('Variable') == 'Data Collection Period')
    print(f"Data Collection Period rows in DataFrame: {data_collection_rows}")

    # Save to CSV
    csv_path = output_path / 'table_one.csv'
    table_one_df.write_csv(csv_path)
    print(f"✓ Saved Table 1 to {csv_path}")

    # Save to HTML
    html_path = output_path / 'table_one.html'
    html_content = table_one_pd.to_html(index=False, border=0, classes='table table-striped')

    # Add CSS styling
    html_with_style = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Table 1: Baseline Characteristics</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            h1 {{
                color: #333;
                text-align: center;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                background-color: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            th {{
                background-color: #4472C4;
                color: white;
                padding: 12px;
                text-align: left;
                font-weight: bold;
                border: 1px solid #ddd;
            }}
            td {{
                padding: 10px 12px;
                border: 1px solid #ddd;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            tr:hover {{
                background-color: #f0f0f0;
            }}
            /* Highlight summary rows */
            tr:nth-child(2), tr:nth-child(3), tr:nth-child(4) {{
                background-color: #e8f2ff;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <h1>TABLE 1: BASELINE CHARACTERISTICS</h1>
        {html_content}
    </body>
    </html>
    """

    with open(html_path, 'w', encoding="utf-8") as f:
        f.write(html_with_style)
    print(f"✓ Saved Table 1 to {html_path}")

    # ============================================
    # Print to console
    # ============================================
    print("\n" + "="*160)
    print("TABLE 1: BASELINE CHARACTERISTICS")
    print("="*160)
    print(table_one_pd.to_string(index=False))
    print("="*160 + "\n")

    return table_one_df


# ==========================================================================
# NIDDK Aim 1 (PDF 1) Table 2: stratified by terminal serum creatinine
# ==========================================================================

def create_table_two_by_terminal_cr(
    final_cohort_df: pl.DataFrame,
    output_dir: str = 'output',
    cohort_filter_column: str = 'imv_48hr_expire',
) -> pl.DataFrame:
    """Build PDF 1's Table 2 — patient characteristics stratified by terminal
    serum creatinine (<2 mg/dL vs ≥2 mg/dL).

    Default population is the "Died while receiving IMV" cohort (age ≤75
    inpatient decedents who were on IMV in the relevant 48 h window). Pass a
    different boolean column name via cohort_filter_column to restrict to,
    e.g., 'clif_eligible_donors' or 'calc_flag'.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Restrict to the chosen denominator
    if cohort_filter_column and cohort_filter_column in final_cohort_df.columns:
        pop = final_cohort_df.filter(pl.col(cohort_filter_column))
    else:
        pop = final_cohort_df

    # Split by terminal Cr; track the "missing creatinine" group explicitly
    # so denominators add up: <2 + >=2 + Missing = population N
    pop_cr_low = pop.filter(pl.col('creatinine_value') < 2)
    pop_cr_high = pop.filter(pl.col('creatinine_value') >= 2)
    pop_cr_missing = pop.filter(pl.col('creatinine_value').is_null())
    strata = {
        '<2_mg_dL': pop_cr_low,
        '>=2_mg_dL': pop_cr_high,
        'missing_Cr': pop_cr_missing,
    }

    # PDF 1 Table 2 row spec — (variable_name, label, kind)
    # kind: 'n_pct_true' for boolean flags, 'median_iqr' for continuous,
    #       'n_pct_male' for sex (count of Male only), 'n_total' for N row.
    binary_yes_n = lambda series: (
        series.filter(pl.col(series.name) == True).shape[0]
        if False else None  # placeholder; we compute inline below
    )
    rows: list[dict] = []

    def fmt_n_pct(n: int, denom: int) -> str:
        return f"{n} ({n/denom*100:.1f}%)" if denom else '—'

    def fmt_median_iqr(series_name: str, df: pl.DataFrame) -> str:
        s = df.select(series_name).to_series().drop_nulls()
        if len(s) == 0:
            return '—'
        return f"{float(s.median()):.1f} ({float(s.quantile(0.25)):.1f}–{float(s.quantile(0.75)):.1f})"

    # N row
    rows.append({
        'Variable': 'N patients',
        **{name: str(len(df)) for name, df in strata.items()},
    })

    # Age: mean (SD) per PDF spec
    age_row = {'Variable': 'Age, mean (SD)'}
    for name, df in strata.items():
        s = df.select('age_at_death').to_series().drop_nulls()
        if len(s) > 0:
            age_row[name] = f"{float(s.mean()):.1f} ({float(s.std()):.1f})"
        else:
            age_row[name] = '—'
    rows.append(age_row)

    # Male sex
    male_row = {'Variable': 'Male sex, n (%)'}
    for name, df in strata.items():
        n_male = df.filter(
            pl.col('sex_category').cast(pl.Utf8).str.to_lowercase() == 'male'
        ).shape[0]
        male_row[name] = fmt_n_pct(n_male, len(df))
    rows.append(male_row)

    # Height / Weight / Terminal Cr / ICU LOS — median (IQR)
    cont_specs = [
        ('Height (cm), median (IQR)', 'last_height_cm'),
        ('Weight (kg), median (IQR)', 'last_weight_kg'),
        ('Terminal creatinine (mg/dL), median (IQR)', 'creatinine_value'),
        ('ICU LOS (days), median (IQR)', 'first_icu_los_days'),
    ]
    for label, colname in cont_specs:
        if colname not in pop.columns:
            continue
        rows.append({
            'Variable': label,
            **{name: fmt_median_iqr(colname, df) for name, df in strata.items()},
        })

    # Comorbidity flags (HCV, HTN, DM, Hx CVA)
    comorbidity_specs = [
        ('HCV infection, n (%)', 'icd10_hcv'),
        ('Hypertension, n (%)', 'icd10_htn'),
        ('Diabetes, n (%)', 'icd10_dm'),
        ('Hx CVA, n (%)', 'icd10_cva'),
    ]
    for label, colname in comorbidity_specs:
        if colname not in pop.columns:
            continue
        row = {'Variable': label}
        for name, df in strata.items():
            n = df.filter(pl.col(colname) == True).shape[0]
            row[name] = fmt_n_pct(n, len(df))
        rows.append(row)

    # Note about HgbA1c being skipped
    rows.append({'Variable': 'HgbA1c', **{n: '— (not in CLIF schema)' for n in strata}})

    table = pl.DataFrame(rows)
    pdf_path_csv = output_path / 'aim1_table_two_by_terminal_cr.csv'
    table.write_csv(pdf_path_csv)
    print(f"✓ Saved Aim 1 Table 2 (by terminal Cr) to {pdf_path_csv}")

    print("\n" + "=" * 100)
    print(f"AIM 1 TABLE 2: Stratified by terminal creatinine  |  population: {cohort_filter_column}")
    print("=" * 100)
    print(table.to_pandas().to_string(index=False))
    print("=" * 100 + "\n")
    return table
