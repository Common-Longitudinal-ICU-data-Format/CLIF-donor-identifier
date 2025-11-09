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
        - 'split': list of 2+ dicts (optional) for branching into multiple boxes
        - 'color': str (optional), 'blue', 'red', or 'green'. Default 'blue'
    """
    # Calculate total height needed - account for split boxes needing more space
    total_height = 0
    step_heights = []
    for i, step in enumerate(steps):
        if 'split' in step and len(step['split']) >= 2:
            # Steps with splits need much more vertical space to prevent overlap
            # More splits = more space needed
            num_splits = len(step['split'])
            base_height = 25.0  # Massively increased from 16.0 to accommodate very tall boxes
            extra_height_per_split = 8.0  # Increased from 5.0 for more space per split
            step_height = base_height + max(0, (num_splits - 2) * extra_height_per_split)
            step_heights.append(step_height)
            total_height += step_height
        # Also check if NEXT step has splits - if so, give current step more space too
        elif i < len(steps) - 1 and 'split' in steps[i + 1] and len(steps[i + 1]['split']) >= 2:
            # Step before a split needs extra space for longer arrow
            step_heights.append(8.0)  # Increased from 6.5 to show full arrow
            total_height += 8.0
        else:
            step_heights.append(7.0)  # Increased from 5.5 to give more vertical space between boxes
            total_height += 7.0

    fig, ax = plt.subplots(figsize=(12, 16))  # Increased height from 10 to 16
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
        if has_note:
            # Calculate note lines for boxes with notes
            note_lines = step['note'].count('\n') + 1
            # Much taller boxes with generous spacing
            box_height = max(5.0, 0.80 * (newline_count + 1) + 0.70 * note_lines + 2.5)
        elif newline_count >= 2:
            box_height = 4.5  # Increased from 4.0
        elif newline_count == 1:
            box_height = 4.0  # Increased from 3.4
        else:
            box_height = 2.2  # Increased from 2.0 for better padding
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
            # Use proportional offsets based on box height for better spacing
            box_center = step_y_pos - box_height/2
            # Balance between spreading elements and keeping away from edges
            label_offset = box_height * 0.27  # Position label in top third
            n_offset = box_height * 0.27  # Position n in bottom third

            ax.text(5, box_center + label_offset, step['label'],
                    ha='center', va='center', fontsize=11, fontweight='bold')
            ax.text(5, box_center, step['note'],
                    ha='center', va='center', fontsize=9, fontweight='normal', style='italic',
                    wrap=True)
            ax.text(5, box_center - n_offset, f"n = {step['n']:,}",
                    ha='center', va='center', fontsize=10)
        else:
            # No note: label at top, n at bottom - use proportional spacing based on box height
            box_center = step_y_pos - box_height/2
            # Use proportional offsets for taller boxes, minimum offset for smaller boxes
            # Keep text well within box boundaries and spread out
            label_offset = min(1.0, box_height * 0.30)
            n_offset = min(1.0, box_height * 0.30)

            ax.text(5, box_center + label_offset, step['label'],
                    ha='center', va='center', fontsize=11, fontweight='bold')
            ax.text(5, box_center - n_offset, f"n = {step['n']:,}",
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
                    split_box_height = max(6.0, 1.0 * label_lines + 0.7 * note_lines + 3.5)
                else:
                    split_box_height = max(2.5, 0.70 * label_lines + 1.5)

                split_boxes.append({
                    'item': split_item,
                    'height': split_box_height,
                    'color': split_box_color
                })

            # Calculate total height needed and distribute boxes evenly
            main_box_center_y = step_y_pos - box_height/2
            total_split_height = sum(b['height'] for b in split_boxes)
            gap_between_boxes = 3.5  # Massively increased from 2.5 for much more space between very tall boxes
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
                    # All elements use 'top' alignment to stack downward from fixed positions

                    note_lines = split_item['note'].count('\n') + 1

                    # Position label at top (0.3 units from top edge)
                    label_y = box_top - 0.3
                    # Position note below label - calculate gap based on label lines to prevent overlap
                    # Use full unit per line to ensure no overlap with bold font
                    label_height_estimate = 1.0 * label_lines  # Full unit per line for bold labels
                    note_y = label_y - label_height_estimate - 0.1  # Minimal gap after label
                    # Position n at a fixed distance from bottom edge
                    n_y = box_bottom + 0.2  # Minimal padding from bottom

                    ax.text(9.75, label_y, split_item['label'],
                            ha='center', va='top', fontsize=10, fontweight='bold')
                    ax.text(9.75, note_y, split_item['note'],
                            ha='center', va='top', fontsize=8, fontweight='normal', style='italic',
                            wrap=True)
                    ax.text(9.75, n_y, f"n = {split_item['n']:,}",
                            ha='center', va='bottom', fontsize=9)
                else:
                    # Without note: just label and n
                    ax.text(9.75, box_center_y + 0.6, split_item['label'],
                            ha='center', va='center', fontsize=10, fontweight='bold')
                    ax.text(9.75, box_center_y - 0.6, f"n = {split_item['n']:,}",
                            ha='center', va='center', fontsize=9)

                # Arrow from main box to this split box
                arrow = FancyArrowPatch((7, main_box_center_y), (7.5, box_center_y),
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