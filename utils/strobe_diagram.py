import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def create_consort_diagram(steps, title="CONSORT Flow Diagram", subtitle=""):
    """
    Create a CONSORT flow diagram.

    Parameters:
    steps: list of dicts with keys:
        - 'label': str, description of the step
        - 'n': int, number in cohort
        - 'excluded': dict (optional) with 'label' and 'n' for exclusions
        - 'split': list of 2 dicts (optional) for branching into two boxes
        - 'color': str (optional), 'blue', 'red', or 'green'. Default 'blue'
    """
    # Calculate total height needed
    total_height = len(steps) * 2  # Each step takes 2 units (splits are now on the right, not below)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, total_height + 2)  # Reduced extra space for title
    ax.axis('off')

    # Title - centered at x=6 (middle of 0-12)
    ax.text(6, total_height + 1.7, title,
            ha='center', va='top', fontsize=16, fontweight='bold')
    if subtitle:
        ax.text(6, total_height + 1.2, subtitle,
                ha='center', va='top', fontsize=12)

    y_pos = total_height

    for i, step in enumerate(steps):
        color_map = {
            'blue': '#ADD8E6',
            'green': '#90EE90',
            'red': '#FFB6C1'
        }
        box_color = color_map.get(step.get('color', 'blue'), '#ADD8E6')
        edge_color = {'blue': '#000080', 'green': '#006400', 'red': '#8B0000'}.get(
            step.get('color', 'blue'), '#000080'
        )

        # Main box - taller box if label has newlines or has a note
        has_note = 'note' in step
        box_height = 1.2 if (('\n' in step['label'] and step['label'].count('\n') > 1) or has_note) else 0.8
        box = FancyBboxPatch((3, y_pos - box_height), 4, box_height,
                            boxstyle="round,pad=0.05",
                            edgecolor=edge_color, facecolor=box_color,
                            linewidth=2.5)
        ax.add_patch(box)

        # Text in box - adjust positioning for taller boxes
        # Check if there's a note (non-bold text)
        if 'note' in step:
            # With note: label at top, note in middle, n at bottom
            label_y = y_pos - box_height/2 + 0.25
            note_y = y_pos - box_height/2
            n_y = y_pos - box_height/2 - 0.25

            ax.text(5, label_y, step['label'],
                    ha='center', va='center', fontsize=11, fontweight='bold')
            ax.text(5, note_y, step['note'],
                    ha='center', va='center', fontsize=9, fontweight='normal', style='italic')
            ax.text(5, n_y, f"n = {step['n']:,}",
                    ha='center', va='center', fontsize=10)
        else:
            # No note: label at top, n at bottom
            label_y = y_pos - box_height/2 + 0.15
            n_y = y_pos - box_height/2 - 0.15
            ax.text(5, label_y, step['label'],
                    ha='center', va='center', fontsize=11, fontweight='bold')
            ax.text(5, n_y, f"n = {step['n']:,}",
                    ha='center', va='center', fontsize=10)

        # Split into two boxes (if exists) - positions them on the RIGHT side
        if 'split' in step and len(step['split']) == 2:
            # First split box (top right)
            top_box_color = color_map.get(step['split'][0].get('color', 'blue'), '#ADD8E6')
            top_edge_color = {'blue': '#000080', 'green': '#006400', 'red': '#8B0000'}.get(
                step['split'][0].get('color', 'blue'), '#000080'
            )
            top_box = FancyBboxPatch((7.5, y_pos - box_height/2 + 0.2), 3.5, 0.8,
                                     boxstyle="round,pad=0.05",
                                     edgecolor=top_edge_color, facecolor=top_box_color,
                                     linewidth=2.5)
            ax.add_patch(top_box)
            ax.text(9.25, y_pos - box_height/2 + 0.6, step['split'][0]['label'],
                    ha='center', va='center', fontsize=10, fontweight='bold')
            ax.text(9.25, y_pos - box_height/2 + 0.4, f"n = {step['split'][0]['n']:,}",
                    ha='center', va='center', fontsize=9)

            # Second split box (bottom right)
            bottom_box_color = color_map.get(step['split'][1].get('color', 'blue'), '#ADD8E6')
            bottom_edge_color = {'blue': '#000080', 'green': '#006400', 'red': '#8B0000'}.get(
                step['split'][1].get('color', 'blue'), '#000080'
            )
            bottom_box = FancyBboxPatch((7.5, y_pos - box_height/2 - 1), 3.5, 0.8,
                                      boxstyle="round,pad=0.05",
                                      edgecolor=bottom_edge_color, facecolor=bottom_box_color,
                                      linewidth=2.5)
            ax.add_patch(bottom_box)
            ax.text(9.25, y_pos - box_height/2 - 0.6, step['split'][1]['label'],
                    ha='center', va='center', fontsize=10, fontweight='bold')
            ax.text(9.25, y_pos - box_height/2 - 0.8, f"n = {step['split'][1]['n']:,}",
                    ha='center', va='center', fontsize=9)

            # Arrows from main box to split boxes (on the right)
            # Arrow to top split box
            arrow_top = FancyArrowPatch((7, y_pos - box_height/2 + 0.2), (7.5, y_pos - box_height/2 + 0.2),
                                        arrowstyle='->', mutation_scale=15,
                                        linewidth=2, color='black')
            ax.add_patch(arrow_top)

            # Arrow to bottom split box
            arrow_bottom = FancyArrowPatch((7, y_pos - box_height/2 - 0.2), (7.5, y_pos - box_height/2 - 0.6),
                                         arrowstyle='->', mutation_scale=15,
                                         linewidth=2, color='black')
            ax.add_patch(arrow_bottom)

            # Arrow to next step continues from main box
            if i < len(steps) - 1:
                arrow = FancyArrowPatch((5, y_pos - box_height), (5, y_pos - 2),
                                       arrowstyle='->', mutation_scale=20,
                                       linewidth=2.5, color='black')
                ax.add_patch(arrow)

            y_pos -= 2

        # Exclusion box (if exists) - supports single or multiple exclusions
        elif 'excluded' in step:
            # Check if it's a list of exclusions or a single exclusion
            exclusions = step['excluded'] if isinstance(step['excluded'], list) else [step['excluded']]

            for idx, exclusion in enumerate(exclusions):
                exc_y = y_pos - box_height/2 - (idx * 1.2)  # Stack exclusions vertically

                exc_box = FancyBboxPatch((8, exc_y - 0.4), 2.5, 0.8,
                                        boxstyle="round,pad=0.05",
                                        edgecolor='#8B0000', facecolor='#FFB6C1',
                                        linewidth=2.5)
                ax.add_patch(exc_box)

                ax.text(9.25, exc_y, exclusion['label'],
                        ha='center', va='center', fontsize=10, fontweight='bold')
                ax.text(9.25, exc_y - 0.2, f"n = {exclusion['n']:,}",
                        ha='center', va='center', fontsize=9)

                # Arrow from main to exclusion
                arrow = FancyArrowPatch((7, y_pos - box_height/2), (8, exc_y),
                                       arrowstyle='->', mutation_scale=15,
                                       linewidth=2, color='#8B0000')
                ax.add_patch(arrow)

            y_pos -= 2

        # Arrow to next step (if not last and no split/exclusion)
        if i < len(steps) - 1 and 'split' not in step and 'excluded' not in step:
            arrow = FancyArrowPatch((5, y_pos - box_height), (5, y_pos - 2),
                                   arrowstyle='->', mutation_scale=20,
                                   linewidth=2.5, color='black')
            ax.add_patch(arrow)
            y_pos -= 2
        elif 'split' not in step and 'excluded' not in step:
            # Last step with no split/exclusion, still need to move y_pos
            y_pos -= 2

    plt.tight_layout()
    return fig