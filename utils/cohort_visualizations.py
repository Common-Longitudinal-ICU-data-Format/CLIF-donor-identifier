"""
Complete Cohort Visualization Script
=====================================

This script:
1. Takes a polars DataFrame (final_cohort_df)
2. Generates cohort_numbers.csv
3. Creates individual funnel charts for CALC and CLIF
4. Creates side-by-side funnel comparison
5. Creates concentric circle diagrams for both definitions

Usage:
------
import polars as pl
from cohort_visualizations import create_all_visualizations

# Load your data
final_cohort_df = pl.read_csv('your_data.csv')

# Create everything
create_all_visualizations(final_cohort_df, output_dir='../output/final')
"""

import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle, FancyArrowPatch, Circle
import numpy as np
import matplotlib.patheffects as path_effects
import os


def calculate_cohort_stages(df, definition='calc'):
    """
    Calculate cohort sizes at each filtering stage from polars DataFrame.
    
    Parameters:
    -----------
    df : polars.DataFrame
        The final cohort dataframe
    definition : str
        Either 'calc' or 'clif'
    
    Returns:
    --------
    list : List of dictionaries with stage information
    """
    
    if definition.lower() == 'calc':
        # Definition 1: CALC
        stages = []
        
        # Stage 1: All inpatient hospital deaths
        df_stage1 = df
        stages.append({
            'stage': 1,
            'label': 'All inpatient hospital deaths',
            'n': len(df_stage1),
            'percentage': 100.0
        })
        
        # Stage 2: Age ≤ 75 years
        df_stage2 = df_stage1.filter(pl.col('age_75_less') == True)
        stages.append({
            'stage': 2,
            'label': 'Patients aged <=75 at death',
            'n': len(df_stage2),
            'percentage': (len(df_stage2) / len(df_stage1)) * 100
        })
        
        # Stage 3: Cause of death (any of the three conditions)
        df_stage3 = df_stage2.filter(
            (pl.col('icd10_ischemic') == True) | 
            (pl.col('icd10_cerebro') == True) | 
            (pl.col('icd10_external') == True)
        )
        stages.append({
            'stage': 3,
            'label': 'Cause',
            'n': len(df_stage3),
            'percentage': (len(df_stage3) / len(df_stage1)) * 100
        })
        
        # Stage 4: No contraindications
        df_stage4 = df_stage3.filter(pl.col('icd10_contraindication') == False)
        stages.append({
            'stage': 4,
            'label': 'No contraindications',
            'n': len(df_stage4),
            'percentage': (len(df_stage4) / len(df_stage1)) * 100
        })
        
    elif definition.lower() == 'clif':
        # Definition 2: CLIF-eligible-donors
        stages = []
        
        # Stage 1: All inpatient hospital deaths
        df_stage1 = df
        stages.append({
            'stage': 1,
            'label': 'All inpatient hospital deaths',
            'n': len(df_stage1),
            'percentage': 100.0
        })
        
        # Stage 2: Age ≤ 75 years
        df_stage2 = df_stage1.filter(pl.col('age_75_less') == True)
        stages.append({
            'stage': 2,
            'label': 'Patients aged <=75 at death',
            'n': len(df_stage2),
            'percentage': (len(df_stage2) / len(df_stage1)) * 100
        })
        
        # Stage 3: On IMV within 48hrs of death
        df_stage3 = df_stage2.filter(pl.col('imv_48hr_expire') == True)
        stages.append({
            'stage': 3,
            'label': 'IMV within 48hrs of death',
            'n': len(df_stage3),
            'percentage': (len(df_stage3) / len(df_stage1)) * 100
        })
        
        # Stage 4: No contraindications
        df_stage4 = df_stage3.filter(
            (pl.col('no_positive_culture_48hrs') == True) & 
            (pl.col('icd10_contraindication') == False)
        )
        stages.append({
            'stage': 4,
            'label': 'No contraindications',
            'n': len(df_stage4),
            'percentage': (len(df_stage4) / len(df_stage1)) * 100
        })
        
        # Stage 5: Pass organ quality check
        df_stage5 = df_stage4.filter(pl.col('organ_check_pass') == True)
        stages.append({
            'stage': 5,
            'label': 'Pass organ quality assessment',
            'n': len(df_stage5),
            'percentage': (len(df_stage5) / len(df_stage1)) * 100
        })
    
    return stages


def generate_csv_from_dataframe(df, output_path='cohort_numbers.csv'):
    """
    Generate cohort_numbers.csv from polars DataFrame.
    
    Parameters:
    -----------
    df : polars.DataFrame
        The final cohort dataframe
    output_path : str
        Path to save the CSV file
    
    Returns:
    --------
    pandas.DataFrame : The generated summary dataframe
    """
    
    # Calculate stages for both definitions
    stages_calc = calculate_cohort_stages(df, 'calc')
    stages_clif = calculate_cohort_stages(df, 'clif')
    
    # Create summary dataframe
    rows = []
    
    # CALC definition
    for stage in stages_calc:
        rows.append({
            'Definition': 'CALC',
            'Stage': stage['stage'],
            'Filter_Description': stage['label'],
            'N': stage['n'],
            'Percentage': round(stage['percentage'], 2)
        })
    
    # CLIF definition
    for stage in stages_clif:
        rows.append({
            'Definition': 'CLIF',
            'Stage': stage['stage'],
            'Filter_Description': stage['label'],
            'N': stage['n'],
            'Percentage': round(stage['percentage'], 2)
        })
    
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_path, index=False)
    
    print(f"✓ CSV saved to: {output_path}")
    return summary_df


def create_nested_funnel_from_csv(csv_path, definition='CALC', output_path=None):
    """
    Create nested ellipse funnel diagram from CSV data.
    Based on the reference code provided.
    
    Parameters:
    -----------
    csv_path : str
        Path to the cohort_numbers.csv file
    definition : str
        'CALC' or 'CLIF'
    output_path : str
        Path to save the figure (optional)
    
    Returns:
    --------
    matplotlib.figure.Figure : The generated figure
    """
    
    # Read CSV
    df = pd.read_csv(csv_path)
    df_filtered = df[df['Definition'] == definition].sort_values('Stage')
    
    # Prepare steps data
    steps = []
    for _, row in df_filtered.iterrows():
        steps.append({
            'label': row['Filter_Description'],
            'n': row['N'],
            'split': []
        })
    
    # Create figure
    fig, ax = plt.subplots(figsize=(24, 16))
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    # Title
    title = f"COHORT SELECTION: {definition} Definition"
    ax.text(12, 15.2, title, ha='center', va='top',
            fontsize=24, fontweight='bold')
    
    # Define VERY distinct colors - light to dark blue gradient
    distinct_colors = [
        '#D6EAF8',  # Very light blue
        '#AED6F1',  # Light blue
        '#85C1E2',  # Medium-light blue  
        '#5DADE2',  # Medium blue
        '#3498DB',  # Medium-dark blue
        '#2E86C1',  # Dark blue
        '#1F618D',  # Very dark blue
    ]
    
    # Center X position
    center_x = 11
    
    # Bottom Y position
    bottom_y = 3
    
    num_main = len(steps)
    
    # Fixed proportional sizing
    size_reduction = 0.78
    ellipses_data = []
    
    for i, step in enumerate(steps):
        scale = size_reduction ** i
        width = 15 * scale
        height = 10 * scale
        color = distinct_colors[min(i, len(distinct_colors)-1)]
        
        center_y = bottom_y + height / 2
        
        ellipses_data.append({
            'width': width,
            'height': height,
            'center_x': center_x,
            'center_y': center_y,
            'color': color,
            'label': step['label'],
            'n': step['n'],
            'index': i
        })
    
    # Draw ellipses from largest to smallest
    for i in range(len(ellipses_data)):
        data = ellipses_data[i]
        zorder_value = i + 1
        
        ellipse = Ellipse((data['center_x'], data['center_y']), 
                         data['width'], data['height'],
                         facecolor=data['color'], 
                         edgecolor='white',
                         linewidth=4,
                         zorder=zorder_value,
                         alpha=1.0)
        ax.add_patch(ellipse)
    
    # Add percentage and n INSIDE ellipses
    total_n = steps[0]['n']
    
    for i, data in enumerate(ellipses_data):
        percentage = (data['n'] / total_n) * 100
        
        # Determine text color
        if i <= 2:
            text_color = '#000000'
        elif i == 3:
            text_color = '#1a1a1a'
        else:
            text_color = 'white'
        
        # Font size
        if i >= 6:
            font_size = 11
        elif i >= 5:
            font_size = 12
        elif i >= 4:
            font_size = 13
        else:
            font_size = 14
        
        # Simple text: percentage and n
        stats_text = f"{percentage:.1f}%\nn={data['n']:,}"
        
        # Position text in the middle of the VISIBLE RING
        if i == len(ellipses_data) - 1:
            # Innermost - just shift up a bit from center
            text_y = data['center_y'] + data['height'] * 0.15
        else:
            # For rings: position between current top and next inner top
            current_top = data['center_y'] + data['height'] / 2
            next_inner_top = ellipses_data[i + 1]['center_y'] + ellipses_data[i + 1]['height'] / 2
            # Position in middle of the ring space
            text_y = (current_top + next_inner_top) / 2
        
        # Create text with outline effect
        txt = ax.text(center_x, text_y, stats_text,
                     ha='center', va='center', 
                     fontsize=font_size, 
                     fontweight='heavy',
                     color=text_color, 
                     zorder=100,
                     linespacing=1.2)
        
        # Add white outline for dark text on light backgrounds
        if i <= 3:
            txt.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground='white'),
                path_effects.Normal()
            ])
    
    # Add descriptive labels on the LEFT side in boxes
    label_box_x = 0.5
    label_box_width = 4
    label_box_height = 0.75
    label_spacing = 0.5
    
    current_y = 13.5
    
    for i, data in enumerate(ellipses_data):
        # Label text only (no box)
        ax.text(label_box_x + label_box_width/2, current_y + label_box_height/2,
               data['label'],
               ha='center', va='center', fontsize=12,
               fontweight='bold')

        # Connecting line
        box_right_x = label_box_x + label_box_width
        box_center_y = current_y + label_box_height/2
        
        a = data['width'] / 2
        b = data['height'] / 2
        ellipse_center_y = data['center_y']
        
        y_target = box_center_y - ellipse_center_y
        y_target = np.clip(y_target, -b * 0.85, b * 0.85)
        
        x_offset_squared = a**2 * (1 - (y_target**2 / b**2))
        x_offset = np.sqrt(max(0, x_offset_squared))
        
        ellipse_left_x = center_x - x_offset
        ellipse_y = ellipse_center_y + y_target
        
        line = FancyArrowPatch(
            (box_right_x + 0.1, box_center_y),
            (ellipse_left_x, ellipse_y),
            arrowstyle='-', 
            linewidth=2, color='black', zorder=50
        )
        ax.add_patch(line)
        
        current_y -= (label_box_height + label_spacing)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Funnel saved to: {output_path}")
    
    return fig


def create_side_by_side_funnels(csv_path, output_path=None):
    """
    Create side-by-side funnel diagrams for CALC and CLIF definitions.

    Parameters:
    -----------
    csv_path : str
        Path to the cohort_numbers.csv file
    output_path : str
        Path to save the figure (optional)

    Returns:
    --------
    matplotlib.figure.Figure : The generated figure
    """

    # Read CSV
    df = pd.read_csv(csv_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Create funnel for each definition
    for ax, definition in [(ax1, 'CALC'), (ax2, 'CLIF')]:
        df_filtered = df[df['Definition'] == definition].sort_values('Stage')

        ax.set_xlim(0, 24)
        ax.set_ylim(0, 16)
        ax.axis('off')

        # Title
        title = f"{definition} Definition"
        ax.text(12, 15.2, title, ha='center', va='top',
                fontsize=22, fontweight='bold')

        # Define distinct colors - light to dark blue gradient
        distinct_colors = [
            '#D6EAF8',  # Very light blue
            '#AED6F1',  # Light blue
            '#85C1E2',  # Medium-light blue
            '#5DADE2',  # Medium blue
            '#3498DB',  # Medium-dark blue
            '#2E86C1',  # Dark blue
            '#1F618D',  # Very dark blue
        ]

        # Center X position
        center_x = 11

        # Bottom Y position
        bottom_y = 3

        # Prepare steps data
        steps = []
        for _, row in df_filtered.iterrows():
            steps.append({
                'label': row['Filter_Description'],
                'n': row['N']
            })

        num_main = len(steps)

        # Fixed proportional sizing
        size_reduction = 0.78
        ellipses_data = []

        for i, step in enumerate(steps):
            scale = size_reduction ** i
            width = 15 * scale
            height = 10 * scale
            color = distinct_colors[min(i, len(distinct_colors)-1)]

            center_y = bottom_y + height / 2

            ellipses_data.append({
                'width': width,
                'height': height,
                'center_x': center_x,
                'center_y': center_y,
                'color': color,
                'label': step['label'],
                'n': step['n'],
                'index': i
            })

        # Draw ellipses from largest to smallest
        for i in range(len(ellipses_data)):
            data = ellipses_data[i]
            zorder_value = i + 1

            ellipse = Ellipse((data['center_x'], data['center_y']),
                             data['width'], data['height'],
                             facecolor=data['color'],
                             edgecolor='white',
                             linewidth=4,
                             zorder=zorder_value,
                             alpha=1.0)
            ax.add_patch(ellipse)

        # Add percentage and n INSIDE ellipses
        total_n = steps[0]['n']

        for i, data in enumerate(ellipses_data):
            percentage = (data['n'] / total_n) * 100

            # Determine text color
            if i <= 2:
                text_color = '#000000'
            elif i == 3:
                text_color = '#1a1a1a'
            else:
                text_color = 'white'

            # Font size
            if i >= 6:
                font_size = 11
            elif i >= 5:
                font_size = 12
            elif i >= 4:
                font_size = 13
            else:
                font_size = 14

            # Simple text: percentage and n
            stats_text = f"{percentage:.1f}%\nn={data['n']:,}"

            # Position text in the middle of the VISIBLE RING
            if i == len(ellipses_data) - 1:
                # Innermost - just shift up a bit from center
                text_y = data['center_y'] + data['height'] * 0.15
            else:
                # For rings: position between current top and next inner top
                current_top = data['center_y'] + data['height'] / 2
                next_inner_top = ellipses_data[i + 1]['center_y'] + ellipses_data[i + 1]['height'] / 2
                # Position in middle of the ring space
                text_y = (current_top + next_inner_top) / 2

            # Create text with outline effect
            txt = ax.text(center_x, text_y, stats_text,
                         ha='center', va='center',
                         fontsize=font_size,
                         fontweight='heavy',
                         color=text_color,
                         zorder=100,
                         linespacing=1.2)

            # Add white outline for dark text on light backgrounds
            if i <= 3:
                txt.set_path_effects([
                    path_effects.Stroke(linewidth=3, foreground='white'),
                    path_effects.Normal()
                ])

        # Add descriptive labels on the LEFT side in boxes
        label_box_x = 0.5
        label_box_width = 4
        label_box_height = 0.75
        label_spacing = 0.5

        current_y = 13.5

        for i, data in enumerate(ellipses_data):
            # Label text only (no box)
            ax.text(label_box_x + label_box_width/2, current_y + label_box_height/2,
                   data['label'],
                   ha='center', va='center', fontsize=11,
                   fontweight='bold')

            # Connecting line
            box_right_x = label_box_x + label_box_width
            box_center_y = current_y + label_box_height/2

            a = data['width'] / 2
            b = data['height'] / 2
            ellipse_center_y = data['center_y']

            y_target = box_center_y - ellipse_center_y
            y_target = np.clip(y_target, -b * 0.85, b * 0.85)

            x_offset_squared = a**2 * (1 - (y_target**2 / b**2))
            x_offset = np.sqrt(max(0, x_offset_squared))

            ellipse_left_x = center_x - x_offset
            ellipse_y = ellipse_center_y + y_target

            line = FancyArrowPatch(
                (box_right_x + 0.1, box_center_y),
                (ellipse_left_x, ellipse_y),
                arrowstyle='-',
                linewidth=2, color='black', zorder=50
            )
            ax.add_patch(line)

            current_y -= (label_box_height + label_spacing)

    plt.suptitle('Potential Deceased Organ Donors', fontsize=26, fontweight='bold', y=0.98)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Side-by-side funnels saved to: {output_path}")

    return fig


def create_concentric_circles_side_by_side(csv_path, output_path=None):
    """
    Create side-by-side concentric circle diagrams for both definitions.
    Circles indented to the right, final donors in GREEN.

    Parameters:
    -----------
    csv_path : str
        Path to the cohort_numbers.csv file
    output_path : str
        Path to save the figure (optional)

    Returns:
    --------
    matplotlib.figure.Figure : The generated figure
    """

    # Read CSV
    df = pd.read_csv(csv_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Process both definitions
    for ax, definition, subplot_label in [(ax1, 'CALC', '(A)'), (ax2, 'CLIF', '(B)')]:
        df_filtered = df[df['Definition'] == definition].sort_values('Stage')

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')

        # Get data
        steps = []
        for _, row in df_filtered.iterrows():
            steps.append({
                'n': row['N'],
                'label': row['Filter_Description'],
                'stage': row['Stage']
            })

        initial_n = steps[0]['n']

        # Definition-specific colors
        if definition == 'CALC':
            colors_map = {
                1: ('#D3D3D3', 'none'),   # Grey: All inpatient deaths
                2: ('#000000', 'none'),   # Black: Age 75 or less
                3: ('#2196F3', 'none'),   # Blue: Cause consistent with donation
                4: ('#F44336', 'none'),   # Red: No contraindications
            }
        else:  # CLIF
            colors_map = {
                1: ('#D3D3D3', 'none'),   # Grey: All inpatient deaths
                2: ('#000000', 'none'),   # Black: Age 75 or less
                3: ('#9C27B0', 'none'),   # Purple: IMV within 48h
                4: ('#F44336', 'none'),   # Red: No contraindications
                5: ('#4CAF50', 'none'),   # Green: Pass organ quality assessment
            }

        # Get max radius for offset calculation
        max_radius = 1.0 * np.sqrt(steps[0]['n'] / initial_n)

        # Base center position
        base_center_x = -0.3
        center_y = 0

        # Draw circles
        for i in range(len(steps)-1, -1, -1):
            step = steps[i]
            stage_num = step['stage']
            radius = 1.0 * np.sqrt(step['n'] / initial_n)
            edge_color, face_color = colors_map.get(stage_num, ('#808080', 'none'))

            # Offset center_x based on radius for indented effect
            # Smaller circles (inner) shift right, larger circles (outer) stay left
            indent_amount = 0.6 * (1 - radius / max_radius)
            center_x = base_center_x + indent_amount

            # Fill innermost circle with light blue for both CALC and CLIF
            if stage_num == len(steps):
                face_color = '#ADD8E6'  # Light blue fill
                alpha = 0.7
                fill = True
            else:
                face_color = 'none'
                alpha = 1.0
                fill = False

            circle = Circle((center_x, center_y), radius,
                           facecolor=face_color,
                           edgecolor=edge_color,
                           linewidth=2.5,
                           alpha=alpha,
                           fill=fill)
            ax.add_patch(circle)

        # Add title (centered at base position)
        title = f"{subplot_label} {definition} Definition"
        ax.text(base_center_x, 1.35, title, ha='center', va='center',
               fontsize=12, fontweight='bold')

    # Add comprehensive legend at the bottom including all stages from both definitions
    legend_labels = []
    legend_handles = []

    # Define all line stages and their labels/colors
    legend_entries = [
        (1, 'All inpatient hospital deaths', '#D3D3D3'),
        (2, 'Patients aged <=75 at death', '#000000'),
        (3, 'Cause consistent with donation (CALC)', '#2196F3'),  # Blue for CALC
        (3.5, 'IMV within 48hrs (CLIF)', '#9C27B0'),  # Purple for CLIF
        (4, 'No contraindications', '#F44336'),
        (5, 'Pass organ quality assessment (CLIF)', '#4CAF50'),
    ]

    # Add all line entries first
    for _, label, color in legend_entries:
        legend_handles.append(plt.Line2D([0], [0], color=color, linewidth=3))
        legend_labels.append(label)

    # Add light blue filled circle entry for potential donors last (applies to both CALC and CLIF)
    legend_handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor='#ADD8E6', markersize=10))
    legend_labels.append('Potential deceased donors')

    fig.legend(legend_handles, legend_labels, loc='lower center',
              ncol=7, frameon=True, fontsize=8,
              bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Circles saved to: {output_path}")

    return fig


def create_all_visualizations(df, output_dir='../outputs'):
    """
    Complete pipeline: Generate CSV and create all visualizations.
    
    Parameters:
    -----------
    df : polars.DataFrame
        The final cohort dataframe with required columns
    output_dir : str
        Directory to save all outputs
    
    Returns:
    --------
    pandas.DataFrame : Summary dataframe with cohort numbers
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("COHORT VISUALIZATION PIPELINE")
    print("="*80)
    
    # Step 1: Generate CSV
    print("\n[1/3] Generating cohort_numbers.csv...")
    csv_path = os.path.join(output_dir, 'cohort_numbers.csv')
    summary_df = generate_csv_from_dataframe(df, csv_path)
    
    # Step 2: Create funnel charts
    print("\n[2/4] Creating funnel charts...")
    print("  → CALC funnel...")
    fig1 = create_nested_funnel_from_csv(
        csv_path,
        'CALC',
        os.path.join(output_dir, 'funnel_calc.png')
    )
    plt.close(fig1)

    print("  → CLIF funnel...")
    fig2 = create_nested_funnel_from_csv(
        csv_path,
        'CLIF',
        os.path.join(output_dir, 'funnel_clif.png')
    )
    plt.close(fig2)

    print("  → Side-by-side funnels...")
    fig_sidebyside = create_side_by_side_funnels(
        csv_path,
        os.path.join(output_dir, 'funnels_side_by_side.png')
    )
    plt.close(fig_sidebyside)
    
    # Step 3: Create concentric circles
    print("\n[3/4] Creating concentric circle diagrams...")
    print("  → Side-by-side comparison...")
    fig3 = create_concentric_circles_side_by_side(
        csv_path,
        os.path.join(output_dir, 'circles_side_by_side.png')
    )
    plt.close(fig3)
    
    print("\n" + "="*80)
    print("✅ ALL VISUALIZATIONS CREATED SUCCESSFULLY!")
    print("="*80)
    print(f"\nOutput directory: {output_dir}/")
    print("\nFiles created:")
    print("  1. cohort_numbers.csv")
    print("  2. funnel_calc.png")
    print("  3. funnel_clif.png")
    print("  4. funnels_side_by_side.png (NEW)")
    print("  5. circles_side_by_side.png")
    
    print("\n" + "="*80)
    print("COHORT SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)
    
    return summary_df


# Example usage
if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║  COHORT VISUALIZATION SCRIPT - COMPLETE PIPELINE                        ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝

This script will:
  1. Generate cohort_numbers.csv from your polars DataFrame
  2. Create funnel charts for CALC and CLIF definitions
  3. Create side-by-side concentric circle comparison

USAGE:
------
import polars as pl
from cohort_visualizations_final import create_all_visualizations

# Load your data
final_cohort_df = pl.read_csv('your_data.csv')

# Create everything
summary_df = create_all_visualizations(final_cohort_df)

REQUIRED COLUMNS:
-----------------
For CALC:
  - age_75_less (bool)
  - icd10_ischemic (bool)
  - icd10_cerebro (bool)
  - icd10_external (bool)
  - icd10_contraindication (bool)

For CLIF:
  - age_75_less (bool)
  - imv_48hr_expire (bool)
  - no_positive_culture_48hrs (bool)
  - icd10_contraindication (bool)
  - organ_check_pass (bool)
    """)