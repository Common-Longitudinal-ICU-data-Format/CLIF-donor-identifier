import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import polars as pl
import pandas as pd
from pathlib import Path
from typing import Optional

def create_consort_diagram(steps, title="CONSORT Flow Diagram", subtitle=""):
    """
    Create a CONSORT flow diagram.

    Parameters:
    steps: list of dicts with keys:
        - 'label': str, description of the step
        - 'n': int, number in cohort
        - 'excluded': dict (optional) with 'label' and 'n' for exclusions
        - 'split': list of 2+ dicts (optional) for branching into multiple boxes
        - 'color': str (optional), 'blue', 'red', or 'green'. Default 'blue'
    """
    # Calculate total height needed - account for split boxes needing more space
    total_height = 0
    step_heights = []
    for i, step in enumerate(steps):
        if 'split' in step and len(step['split']) >= 2:
            # Steps with splits need vertical space for multiple boxes
            num_splits = len(step['split'])
            base_height = 18.0
            extra_height_per_split = 5.5
            step_height = base_height + max(0, (num_splits - 2) * extra_height_per_split)
            step_heights.append(step_height)
            total_height += step_height
        # Also check if NEXT step has splits - if so, give current step more space too
        elif i < len(steps) - 1 and 'split' in steps[i + 1] and len(steps[i + 1]['split']) >= 2:
            # Step before a split needs extra space for longer arrow
            step_heights.append(6.0)
            total_height += 6.0
        else:
            step_heights.append(5.0)
            total_height += 5.0

    fig, ax = plt.subplots(figsize=(12, 14))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, total_height + 2)
    ax.axis('off')

    # Title - centered at x=6 (middle of 0-12)
    ax.text(6, total_height + 1.7, title,
            ha='center', va='top', fontsize=16, fontweight='bold')
    if subtitle:
        ax.text(6, total_height + 1.2, subtitle,
                ha='center', va='top', fontsize=12)

    # Calculate positions for all steps upfront using cumulative heights
    step_positions = []
    current_y = total_height
    for i, step_height in enumerate(step_heights):
        step_positions.append(current_y)
        current_y -= step_height

    for i, step in enumerate(steps):
        # Get the pre-calculated y position for this step
        step_y_pos = step_positions[i]

        color_map = {
            'blue': '#ADD8E6',
            'green': '#90EE90',
            'red': '#FFB6C1'
        }
        box_color = color_map.get(step.get('color', 'blue'), '#ADD8E6')
        # edge_color is not needed for boundary-less boxes
        # edge_color = {'blue': '#000080', 'green': '#006400', 'red': '#8B0000'}.get(
        #     step.get('color', 'blue'), '#000080'
        # )

        # Main box - taller box if label has newlines or has a note
        has_note = 'note' in step
        # Count newlines in label to determine box height
        newline_count = step['label'].count('\n')
        if has_note:
            # Calculate note lines for boxes with notes
            note_lines = step['note'].count('\n') + 1
            # Compact boxes with minimal padding
            box_height = max(4.0, 0.55 * (newline_count + 1) + 0.50 * note_lines + 1.8)
        elif newline_count >= 2:
            box_height = 3.2
        elif newline_count == 1:
            box_height = 2.8
        else:
            box_height = 1.8
        box = FancyBboxPatch(
            (3, step_y_pos - box_height),
            4, box_height,
            boxstyle="round,pad=0.08",
            edgecolor='none',        # <- No box boundary
            facecolor=box_color,
            linewidth=0              # <- No shared boundary
        )
        ax.add_patch(box)

        # Text in box - adjust positioning with minimal padding
        # Check if there's a note (non-bold text)
        if 'note' in step:
            # With note: label at top, note in middle, n at bottom
            box_center = step_y_pos - box_height/2
            # Minimal offsets for compact spacing
            label_offset = box_height * 0.25
            n_offset = box_height * 0.25

            ax.text(5, box_center + label_offset, step['label'],
                    ha='center', va='center', fontsize=10, fontweight='bold')
            ax.text(5, box_center, step['note'],
                    ha='center', va='center', fontsize=8, fontweight='normal', style='italic',
                    wrap=True)
            ax.text(5, box_center - n_offset, f"n = {step['n']:,}",
                    ha='center', va='center', fontsize=9)
        else:
            # No note: label at top, n at bottom
            box_center = step_y_pos - box_height/2
            # Tighter spacing to fit in smaller boxes
            label_offset = box_height * 0.28

            ax.text(5, box_center + label_offset, step['label'],
                    ha='center', va='center', fontsize=11, fontweight='bold')
            ax.text(5, box_center - label_offset, f"n = {step['n']:,}",
                    ha='center', va='center', fontsize=10)

        # Split into multiple boxes (if exists) - positions them on the RIGHT side
        if 'split' in step and len(step['split']) >= 2:
            num_splits = len(step['split'])
            split_boxes = []

            # Calculate heights for all split boxes
            for split_item in step['split']:
                split_box_color = color_map.get(split_item.get('color', 'blue'), '#ADD8E6')
                label_lines = split_item['label'].count('\n') + 1

                if 'note' in split_item and split_item['note']:
                    note_lines = split_item['note'].count('\n') + 1
                    # Box height with compact spacing
                    split_box_height = max(4.5, 0.70 * label_lines + 0.50 * note_lines + 2.5)
                else:
                    split_box_height = max(1.8, 0.50 * label_lines + 1.0)

                split_boxes.append({
                    'item': split_item,
                    'height': split_box_height,
                    'color': split_box_color
                })

            # Calculate total height needed and distribute boxes evenly
            main_box_center_y = step_y_pos - box_height/2
            total_split_height = sum(b['height'] for b in split_boxes)
            gap_between_boxes = 2.8  # Reduced from 3.5 for tighter packing
            total_gaps = gap_between_boxes * (num_splits - 1)

            # Start from the top and work down
            current_y = step_y_pos - 0.15

            for idx, split_box_info in enumerate(split_boxes):
                split_item = split_box_info['item']
                split_box_height = split_box_info['height']
                split_box_color = split_box_info['color']

                # Calculate box position
                box_top = current_y
                box_bottom = box_top - split_box_height
                box_center_y = (box_top + box_bottom) / 2

                # Draw the split box
                box = FancyBboxPatch(
                    (7.5, box_bottom), 4.5, split_box_height,
                    boxstyle="round,pad=0.05",
                    edgecolor='none',
                    facecolor=split_box_color,
                    linewidth=0
                )
                ax.add_patch(box)

                # Add text to split box
                if 'note' in split_item and split_item['note']:
                    # With note: stack elements from top with fixed spacing
                    note_lines = split_item['note'].count('\n') + 1

                    # Position label at top
                    label_y = box_top - 0.25
                    # Position note below label
                    label_height_estimate = 0.65 * label_lines
                    note_y = label_y - label_height_estimate - 0.05
                    # Position n at a fixed distance from bottom edge
                    n_y = box_bottom + 0.15

                    ax.text(9.75, label_y, split_item['label'],
                            ha='center', va='top', fontsize=9, fontweight='bold')
                    ax.text(9.75, note_y, split_item['note'],
                            ha='center', va='top', fontsize=7, fontweight='normal', style='italic',
                            wrap=True)
                    ax.text(9.75, n_y, f"n = {split_item['n']:,}",
                            ha='center', va='bottom', fontsize=8)
                else:
                    # Without note: just label and n - centered in pink boxes
                    box_center_y = (box_top + box_bottom) / 2
                    ax.text(9.75, box_center_y + 0.3, split_item['label'],
                            ha='center', va='center', fontsize=9, fontweight='bold')
                    ax.text(9.75, box_center_y - 0.3, f"n = {split_item['n']:,}",
                            ha='center', va='center', fontsize=8)

                # Arrow from main box to this split box
                split_box_center_y = (box_top + box_bottom) / 2
                arrow = FancyArrowPatch((7, main_box_center_y), (7.5, split_box_center_y),
                                        arrowstyle='->', mutation_scale=15,
                                        linewidth=2, color='black')
                ax.add_patch(arrow)

                # Move down for next box
                current_y = box_bottom - gap_between_boxes

        # Exclusion box (if exists) - supports single or multiple exclusions
        elif 'excluded' in step:
            # Check if it's a list of exclusions or a single exclusion
            exclusions = step['excluded'] if isinstance(step['excluded'], list) else [step['excluded']]

            for idx, exclusion in enumerate(exclusions):
                exc_y = step_y_pos - box_height/2 - (idx * 1.5)  # Stack exclusions vertically with more spacing

                exc_box_height = 1.0  # Increased from 0.8
                exc_box = FancyBboxPatch(
                    (8, exc_y - exc_box_height/2), 2.8, exc_box_height,
                    boxstyle="round,pad=0.10",
                    edgecolor='none',          # <- No box boundary
                    facecolor='#FFB6C1',
                    linewidth=0
                )
                ax.add_patch(exc_box)

                ax.text(9.4, exc_y + 0.15, exclusion['label'],
                        ha='center', va='center', fontsize=9, fontweight='bold')
                ax.text(9.4, exc_y - 0.25, f"n = {exclusion['n']:,}",
                        ha='center', va='center', fontsize=8)

                # Arrow from main to exclusion
                arrow = FancyArrowPatch((7, step_y_pos - box_height/2), (8, exc_y),
                                       arrowstyle='->', mutation_scale=15,
                                       linewidth=2, color='#8B0000')
                ax.add_patch(arrow)

        # Arrow to next step (draw for all steps except last)
        if i < len(steps) - 1:
            next_step_y_pos = step_positions[i + 1]
            arrow = FancyArrowPatch((5, step_y_pos - box_height), (5, next_step_y_pos),
                                   arrowstyle='->', mutation_scale=20,
                                   linewidth=2.5, color='black')
            ax.add_patch(arrow)

    plt.tight_layout()
    return fig


def create_strobe_diagrams_for_cohorts(
    final_cohort_df: pl.DataFrame,
    output_dir: str = 'output',
    save_figures: bool = True,
    save_csvs: bool = True
) -> dict:
    """
    Create separate STROBE diagrams for CALC and CLIF definitions with detailed dropout information.

    Parameters
    ----------
    final_cohort_df : pl.DataFrame
        Final cohort dataframe with all variables and flags
    output_dir : str
        Directory to save output figures and CSVs
    save_figures : bool
        Whether to save diagram figures as PNG
    save_csvs : bool
        Whether to save stage data as CSV files

    Returns
    -------
    dict
        Dictionary with 'CALC' and 'CLIF' keys, each containing figure object and stage data
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {}

    # ============================================
    # CALC DEFINITION STROBE
    # ============================================
    print("\n" + "="*80)
    print("CALC DEFINITION STROBE DIAGRAM")
    print("="*80)

    calc_stages = []
    calc_csv_data = []

    # Stage 1: All inpatient hospital deaths
    df_stage1 = final_cohort_df
    n_stage1 = len(df_stage1)
    calc_stages.append({
        'label': 'All inpatient hospital deaths',
        'n': n_stage1,
        'color': 'blue'
    })
    calc_csv_data.append({
        'Definition': 'CALC',
        'Stage': 1,
        'Description': 'All inpatient hospital deaths',
        'Count': n_stage1,
        'Excluded': 0,
        'Percentage_of_previous': 100.0
    })
    print(f"Stage 1 - All inpatient hospital deaths: n={n_stage1:,}")

    # Stage 2: Age ≤ 75 years
    df_stage2 = df_stage1.filter(pl.col('age_75_less') == True)
    n_stage2 = len(df_stage2)
    excluded_age = n_stage1 - n_stage2
    pct_stage2 = (n_stage2 / n_stage1 * 100) if n_stage1 > 0 else 0

    calc_stages.append({
        'label': 'Patients aged ≤75 at death',
        'n': n_stage2,
        'excluded': {'label': f'Age >75', 'n': excluded_age},
        'color': 'blue'
    })
    calc_csv_data.append({
        'Definition': 'CALC',
        'Stage': 2,
        'Description': 'Patients aged ≤75 at death',
        'Count': n_stage2,
        'Excluded': excluded_age,
        'Percentage_of_previous': pct_stage2
    })
    print(f"Stage 2 - Age ≤75: n={n_stage2:,} (excluded: {excluded_age:,})")

    # Stage 3: Cause of death (any of three conditions)
    df_stage3 = df_stage2.filter(
        (pl.col('icd10_ischemic') == True) |
        (pl.col('icd10_cerebro') == True) |
        (pl.col('icd10_external') == True)
    )
    n_stage3 = len(df_stage3)
    excluded_cause = n_stage2 - n_stage3
    pct_stage3 = (n_stage3 / n_stage2 * 100) if n_stage2 > 0 else 0

    calc_stages.append({
        'label': 'Cause of death consistent with donation\n(Ischemic, Cerebrovascular, or External causes)',
        'n': n_stage3,
        'excluded': {'label': 'Cause not suitable', 'n': excluded_cause},
        'color': 'blue'
    })
    calc_csv_data.append({
        'Definition': 'CALC',
        'Stage': 3,
        'Description': 'Cause of death consistent with donation',
        'Count': n_stage3,
        'Excluded': excluded_cause,
        'Percentage_of_previous': pct_stage3
    })
    print(f"Stage 3 - Suitable cause of death: n={n_stage3:,} (excluded: {excluded_cause:,})")

    # Stage 4: No contraindications (sepsis/cancer)
    df_stage4 = df_stage3.filter(pl.col('icd10_contraindication') == False)
    n_stage4 = len(df_stage4)
    excluded_contra = n_stage3 - n_stage4
    pct_stage4 = (n_stage4 / n_stage3 * 100) if n_stage3 > 0 else 0

    calc_stages.append({
        'label': 'No contraindications\n(Sepsis/Cancer by Goldberg et al)',
        'n': n_stage4,
        'excluded': {'label': 'Sepsis or cancer contraindication', 'n': excluded_contra},
        'color': 'blue'
    })
    calc_csv_data.append({
        'Definition': 'CALC',
        'Stage': 4,
        'Description': 'No contraindications (Sepsis/Cancer)',
        'Count': n_stage4,
        'Excluded': excluded_contra,
        'Percentage_of_previous': pct_stage4
    })
    print(f"Stage 4 - No contraindications: n={n_stage4:,} (excluded: {excluded_contra:,})")

    # Create CALC STROBE diagram
    calc_title = "CALC Definition - Potential Organ Donors"
    calc_fig = create_consort_diagram(calc_stages, title=calc_title)

    if save_figures:
        calc_figure_path = output_path / 'strobe_calc_definition.png'
        calc_fig.savefig(calc_figure_path, dpi=300, bbox_inches='tight')
        print(f"✓ CALC STROBE saved to: {calc_figure_path}")

    if save_csvs:
        calc_csv_path = output_path / 'strobe_calc_definition.csv'
        calc_csv_df = pd.DataFrame(calc_csv_data)
        calc_csv_df.to_csv(calc_csv_path, index=False)
        print(f"✓ CALC stage data saved to: {calc_csv_path}")

    results['CALC'] = {
        'figure': calc_fig,
        'stages': calc_csv_data
    }

    # ============================================
    # CLIF DEFINITION STROBE
    # ============================================
    print("\n" + "="*80)
    print("CLIF DEFINITION STROBE DIAGRAM")
    print("="*80)

    clif_stages = []
    clif_csv_data = []

    # Stage 1: All inpatient hospital deaths
    df_stage1 = final_cohort_df
    n_stage1 = len(df_stage1)
    clif_stages.append({
        'label': 'All inpatient hospital deaths\n(ED, Ward, Stepdown, ICU)',
        'n': n_stage1,
        'color': 'blue'
    })
    clif_csv_data.append({
        'Definition': 'CLIF',
        'Stage': 1,
        'Description': 'All inpatient hospital deaths',
        'Count': n_stage1,
        'Excluded': 0,
        'Percentage_of_previous': 100.0
    })
    print(f"Stage 1 - All inpatient hospital deaths: n={n_stage1:,}")

    # Stage 2: Age ≤ 75 years
    df_stage2 = df_stage1.filter(pl.col('age_75_less') == True)
    n_stage2 = len(df_stage2)
    excluded_age = n_stage1 - n_stage2
    pct_stage2 = (n_stage2 / n_stage1 * 100) if n_stage1 > 0 else 0

    clif_stages.append({
        'label': 'Patients aged ≤75 at death',
        'n': n_stage2,
        'excluded': {'label': f'Age >75', 'n': excluded_age},
        'color': 'blue'
    })
    clif_csv_data.append({
        'Definition': 'CLIF',
        'Stage': 2,
        'Description': 'Patients aged ≤75 at death',
        'Count': n_stage2,
        'Excluded': excluded_age,
        'Percentage_of_previous': pct_stage2
    })
    print(f"Stage 2 - Age ≤75: n={n_stage2:,} (excluded: {excluded_age:,})")

    # Stage 3: On invasive mechanical ventilation within 48hrs of death
    df_stage3 = df_stage2.filter(pl.col('imv_48hr_expire') == True)
    n_stage3 = len(df_stage3)
    excluded_imv = n_stage2 - n_stage3
    pct_stage3 = (n_stage3 / n_stage2 * 100) if n_stage2 > 0 else 0

    clif_stages.append({
        'label': 'On invasive mechanical ventilation\nwithin 48 hours of death',
        'n': n_stage3,
        'excluded': {'label': 'No IMV within 48hrs', 'n': excluded_imv},
        'color': 'blue'
    })
    clif_csv_data.append({
        'Definition': 'CLIF',
        'Stage': 3,
        'Description': 'On invasive mechanical ventilation within 48hrs',
        'Count': n_stage3,
        'Excluded': excluded_imv,
        'Percentage_of_previous': pct_stage3
    })
    print(f"Stage 3 - On IMV within 48hrs: n={n_stage3:,} (excluded: {excluded_imv:,})")

    # Stage 4: No contraindications (no positive cultures in 48hrs AND no sepsis/cancer)
    df_stage4 = df_stage3.filter(
        (pl.col('no_positive_culture_48hrs') == True) &
        (pl.col('icd10_contraindication') == False)
    )
    n_stage4 = len(df_stage4)

    # Track exclusions separately
    df_no_culture = df_stage3.filter(pl.col('no_positive_culture_48hrs') == False)
    excluded_culture = len(df_no_culture)

    df_no_contra = df_stage3.filter(pl.col('icd10_contraindication') == True)
    excluded_contra = len(df_no_contra)

    total_excluded_stage4 = n_stage3 - n_stage4
    pct_stage4 = (n_stage4 / n_stage3 * 100) if n_stage3 > 0 else 0

    clif_stages.append({
        'label': 'No contraindications\n(No positive cultures & no Sepsis/Cancer)',
        'n': n_stage4,
        'excluded': [
            {'label': f'Positive culture within 48hrs', 'n': excluded_culture},
            {'label': f'Sepsis/Cancer contraindication', 'n': excluded_contra}
        ],
        'color': 'blue'
    })
    clif_csv_data.append({
        'Definition': 'CLIF',
        'Stage': 4,
        'Description': 'No contraindications (no cultures & no sepsis/cancer)',
        'Count': n_stage4,
        'Excluded': total_excluded_stage4,
        'Percentage_of_previous': pct_stage4,
        'Excluded_cultures': excluded_culture,
        'Excluded_contraindications': excluded_contra
    })
    print(f"Stage 4 - No contraindications: n={n_stage4:,} (excluded: {total_excluded_stage4:,})")
    print(f"  - Positive culture: {excluded_culture:,}")
    print(f"  - Sepsis/Cancer: {excluded_contra:,}")

    # Stage 5: Pass organ quality assessment
    df_stage5 = df_stage4.filter(pl.col('organ_check_pass') == True)
    n_stage5 = len(df_stage5)

    # Track specific organ eligibility failures
    df_no_kidney = df_stage4.filter(pl.col('kidney_eligible') == False)
    excluded_kidney = len(df_no_kidney)

    df_no_liver = df_stage4.filter(pl.col('liver_eligible') == False)
    excluded_liver = len(df_no_liver)

    df_no_bmi = df_stage4.filter(pl.col('bmi_eligible') == False)
    excluded_bmi = len(df_no_bmi)

    # Patients could fail multiple criteria, so we track total excluded
    total_excluded_stage5 = n_stage4 - n_stage5
    pct_stage5 = (n_stage5 / n_stage4 * 100) if n_stage4 > 0 else 0

    clif_stages.append({
        'label': 'Pass organ quality assessment\n(Kidney, Liver, BMI eligible)',
        'n': n_stage5,
        'excluded': [
            {'label': f'Kidney not eligible', 'n': excluded_kidney},
            {'label': f'Liver not eligible', 'n': excluded_liver},
            {'label': f'BMI not eligible', 'n': excluded_bmi}
        ],
        'color': 'blue'
    })
    clif_csv_data.append({
        'Definition': 'CLIF',
        'Stage': 5,
        'Description': 'Pass organ quality assessment',
        'Count': n_stage5,
        'Excluded': total_excluded_stage5,
        'Percentage_of_previous': pct_stage5,
        'Excluded_kidney': excluded_kidney,
        'Excluded_liver': excluded_liver,
        'Excluded_bmi': excluded_bmi
    })
    print(f"Stage 5 - Organ quality assessment pass: n={n_stage5:,} (excluded: {total_excluded_stage5:,})")
    print(f"  - Kidney not eligible: {excluded_kidney:,}")
    print(f"  - Liver not eligible: {excluded_liver:,}")
    print(f"  - BMI not eligible: {excluded_bmi:,}")

    # Create CLIF STROBE diagram
    clif_title = "CLIF Definition - Eligible Deceased Organ Donors"
    clif_fig = create_consort_diagram(clif_stages, title=clif_title)

    if save_figures:
        clif_figure_path = output_path / 'strobe_clif_definition.png'
        clif_fig.savefig(clif_figure_path, dpi=300, bbox_inches='tight')
        print(f"✓ CLIF STROBE saved to: {clif_figure_path}")

    if save_csvs:
        clif_csv_path = output_path / 'strobe_clif_definition.csv'
        clif_csv_df = pd.DataFrame(clif_csv_data)
        clif_csv_df.to_csv(clif_csv_path, index=False)
        print(f"✓ CLIF stage data saved to: {clif_csv_path}")

    results['CLIF'] = {
        'figure': clif_fig,
        'stages': clif_csv_data
    }

    print("\n" + "="*80)
    print("STROBE DIAGRAM GENERATION COMPLETE")
    print("="*80 + "\n")

    return results