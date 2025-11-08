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
    # Calculate total height needed - account for split boxes needing more space
    total_height = 0
    step_heights = []
    for i, step in enumerate(steps):
        if 'split' in step and len(step['split']) == 2:
            # Steps with splits need much more vertical space to prevent overlap
            step_heights.append(7.0)  # Increased from 5.0 to give even more room
            total_height += 7.0
        # Also check if NEXT step has splits - if so, give current step more space too
        elif i < len(steps) - 1 and 'split' in steps[i + 1] and len(steps[i + 1]['split']) == 2:
            # Step before a split needs extra space for longer arrow
            step_heights.append(4.0)  # Increased from 3.5
            total_height += 4.0
        else:
            step_heights.append(2.5)
            total_height += 2.5

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
        # Count newlines in label to determine box height (increased for better padding)
        newline_count = step['label'].count('\n')
        if newline_count >= 2 or has_note:
            box_height = 1.7  # Increased from 1.5
        elif newline_count == 1:
            box_height = 1.2  # Increased from 1.0
        else:
            box_height = 1.0  # Increased from 0.8
        box = FancyBboxPatch(
            (3, step_y_pos - box_height),
            4, box_height,
            boxstyle="round,pad=0.05",
            edgecolor='none',        # <- No box boundary
            facecolor=box_color,
            linewidth=0              # <- No shared boundary
        )
        ax.add_patch(box)

        # Text in box - adjust positioning for taller boxes with better padding
        # Check if there's a note (non-bold text)
        if 'note' in step:
            # With note: label at top, note in middle, n at bottom
            label_y = step_y_pos - box_height/2 + 0.4
            note_y = step_y_pos - box_height/2
            n_y = step_y_pos - box_height/2 - 0.4

            ax.text(5, label_y, step['label'],
                    ha='center', va='center', fontsize=11, fontweight='bold')
            ax.text(5, note_y, step['note'],
                    ha='center', va='center', fontsize=9, fontweight='normal', style='italic')
            ax.text(5, n_y, f"n = {step['n']:,}",
                    ha='center', va='center', fontsize=10)
        else:
            # No note: label at top, n at bottom - adjust based on box height with more padding
            if box_height >= 1.5:
                label_y = step_y_pos - box_height/2 + 0.4
                n_y = step_y_pos - box_height/2 - 0.4
            elif box_height >= 1.0:
                label_y = step_y_pos - box_height/2 + 0.3
                n_y = step_y_pos - box_height/2 - 0.3
            else:
                label_y = step_y_pos - box_height/2 + 0.2
                n_y = step_y_pos - box_height/2 - 0.2
            ax.text(5, label_y, step['label'],
                    ha='center', va='center', fontsize=11, fontweight='bold')
            ax.text(5, n_y, f"n = {step['n']:,}",
                    ha='center', va='center', fontsize=10)

        # Split into two boxes (if exists) - positions them on the RIGHT side
        if 'split' in step and len(step['split']) == 2:
            # Calculate heights for both boxes first
            # First split box (top right)
            top_box_color = color_map.get(step['split'][0].get('color', 'blue'), '#ADD8E6')

            # Calculate top box height based on content - be very generous with space
            top_label_lines = step['split'][0]['label'].count('\n') + 1
            if 'note' in step['split'][0] and step['split'][0]['note']:
                top_note_lines = step['split'][0]['note'].count('\n') + 1
                # Much taller boxes for notes - each line needs space
                top_box_height = max(2.0, 0.4 * top_label_lines + 0.35 * top_note_lines + 0.35 + 0.6)
            else:
                top_box_height = max(1.0, 0.4 * top_label_lines + 0.35 + 0.35)

            # Calculate bottom box height based on content - be very generous with space
            bottom_box_color = color_map.get(step['split'][1].get('color', 'blue'), '#ADD8E6')
            bottom_label_lines = step['split'][1]['label'].count('\n') + 1
            if 'note' in step['split'][1] and step['split'][1]['note']:
                bottom_note_lines = step['split'][1]['note'].count('\n') + 1
                # Much taller boxes for notes
                bottom_box_height = max(2.0, 0.4 * bottom_label_lines + 0.35 * bottom_note_lines + 0.35 + 0.6)
            else:
                bottom_box_height = max(1.0, 0.4 * bottom_label_lines + 0.35 + 0.35)

            # Position split boxes more explicitly with clear separation
            # Main box: top at step_y_pos, bottom at (step_y_pos - box_height)

            # Calculate positions to ensure boxes don't touch
            main_box_center_y = step_y_pos - box_height/2
            gap_between_boxes = 0.3  # Space between the two split boxes

            # Top split box: positioned in upper half
            top_box_top = step_y_pos - 0.15
            top_box_bottom = top_box_top - top_box_height
            top_box_y_center = (top_box_top + top_box_bottom) / 2

            # Bottom split box: positioned in lower half with clear gap from top box
            main_box_bottom = step_y_pos - box_height
            bottom_box_bottom = main_box_bottom + 0.15
            bottom_box_top = bottom_box_bottom + bottom_box_height
            bottom_box_y_center = (bottom_box_top + bottom_box_bottom) / 2

            # Ensure there's a visible gap between the boxes
            min_gap = top_box_bottom - bottom_box_top
            if min_gap < gap_between_boxes:
                # Adjust positions to create gap
                overlap = gap_between_boxes - min_gap
                top_box_bottom = top_box_bottom + overlap/2
                top_box_top = top_box_bottom + top_box_height
                top_box_y_center = (top_box_top + top_box_bottom) / 2

                bottom_box_top = bottom_box_top - overlap/2
                bottom_box_bottom = bottom_box_top - bottom_box_height
                bottom_box_y_center = (bottom_box_top + bottom_box_bottom) / 2

            # Now draw the top box with adjusted position
            top_box = FancyBboxPatch(
                (7.5, top_box_bottom), 4.5, top_box_height,
                boxstyle="round,pad=0.05",
                edgecolor='none',
                facecolor=top_box_color,
                linewidth=0
            )
            ax.add_patch(top_box)

            # Add text to top split box
            if 'note' in step['split'][0] and step['split'][0]['note']:
                # With note: position from top to bottom with clear separation
                # Label at top
                ax.text(9.75, top_box_top - 0.25, step['split'][0]['label'],
                        ha='center', va='top', fontsize=9.5, fontweight='bold')
                # Note in middle
                ax.text(9.75, top_box_y_center, step['split'][0]['note'],
                        ha='center', va='center', fontsize=7, fontweight='normal', style='italic')
                # n at bottom
                ax.text(9.75, top_box_bottom + 0.25, f"n = {step['split'][0]['n']:,}",
                        ha='center', va='bottom', fontsize=8.5)
            else:
                # Without note: just label and n
                ax.text(9.75, top_box_y_center + 0.25, step['split'][0]['label'],
                        ha='center', va='center', fontsize=10, fontweight='bold')
                ax.text(9.75, top_box_y_center - 0.25, f"n = {step['split'][0]['n']:,}",
                        ha='center', va='center', fontsize=9)

            # Now draw the bottom box with adjusted position
            bottom_box = FancyBboxPatch(
                (7.5, bottom_box_bottom), 4.5, bottom_box_height,
                boxstyle="round,pad=0.05",
                edgecolor='none',
                facecolor=bottom_box_color,
                linewidth=0
            )
            ax.add_patch(bottom_box)

            # Add text to bottom split box
            if 'note' in step['split'][1] and step['split'][1]['note']:
                # With note: position from top to bottom with clear separation
                # Label at top
                ax.text(9.75, bottom_box_top - 0.25, step['split'][1]['label'],
                        ha='center', va='top', fontsize=9.5, fontweight='bold')
                # Note in middle
                ax.text(9.75, bottom_box_y_center, step['split'][1]['note'],
                        ha='center', va='center', fontsize=7, fontweight='normal', style='italic')
                # n at bottom
                ax.text(9.75, bottom_box_bottom + 0.25, f"n = {step['split'][1]['n']:,}",
                        ha='center', va='bottom', fontsize=8.5)
            else:
                # Without note: just label and n
                ax.text(9.75, bottom_box_y_center + 0.25, step['split'][1]['label'],
                        ha='center', va='center', fontsize=10, fontweight='bold')
                ax.text(9.75, bottom_box_y_center - 0.25, f"n = {step['split'][1]['n']:,}",
                        ha='center', va='center', fontsize=9)

            # Arrows from main box to split boxes (angled to show clear separation)
            # Arrow from main box center, angled upward to top split box
            main_box_center_y = step_y_pos - box_height/2
            arrow_top = FancyArrowPatch((7, main_box_center_y + 0.15), (7.5, top_box_y_center),
                                        arrowstyle='->', mutation_scale=15,
                                        linewidth=2, color='black')
            ax.add_patch(arrow_top)

            # Arrow from main box center, angled downward to bottom split box
            arrow_bottom = FancyArrowPatch((7, main_box_center_y - 0.15), (7.5, bottom_box_y_center),
                                         arrowstyle='->', mutation_scale=15,
                                         linewidth=2, color='black')
            ax.add_patch(arrow_bottom)

        # Exclusion box (if exists) - supports single or multiple exclusions
        elif 'excluded' in step:
            # Check if it's a list of exclusions or a single exclusion
            exclusions = step['excluded'] if isinstance(step['excluded'], list) else [step['excluded']]

            for idx, exclusion in enumerate(exclusions):
                exc_y = step_y_pos - box_height/2 - (idx * 1.2)  # Stack exclusions vertically

                exc_box = FancyBboxPatch(
                    (8, exc_y - 0.4), 2.5, 0.8,
                    boxstyle="round,pad=0.05",
                    edgecolor='none',          # <- No box boundary
                    facecolor='#FFB6C1',
                    linewidth=0
                )
                ax.add_patch(exc_box)

                ax.text(9.25, exc_y, exclusion['label'],
                        ha='center', va='center', fontsize=10, fontweight='bold')
                ax.text(9.25, exc_y - 0.2, f"n = {exclusion['n']:,}",
                        ha='center', va='center', fontsize=9)

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