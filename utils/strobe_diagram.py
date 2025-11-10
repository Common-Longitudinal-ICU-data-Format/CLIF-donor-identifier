import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle, FancyArrowPatch
import numpy as np
import matplotlib.patheffects as path_effects

def create_nested_funnel_diagram(steps, title="", subtitle=""):
    """
    Create a nested ellipse funnel diagram with labels on the left and stats inside ellipses.
    """
    fig, ax = plt.subplots(figsize=(24, 16))
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    # Title
    if title:
        ax.text(12, 15.2, title, ha='center', va='top', 
                fontsize=20, fontweight='bold')
    
    # Define VERY distinct colors
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
    
    # Main steps
    main_steps = steps.copy()
    num_main = len(main_steps)
    
    # Fixed proportional sizing
    size_reduction = 0.78
    ellipses_data = []
    
    for i, step in enumerate(main_steps):
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
            'split': step.get('split', []),
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
    total_n = main_steps[0]['n']
    
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
            font_size = 9
        elif i >= 5:
            font_size = 10
        elif i >= 4:
            font_size = 11
        else:
            font_size = 12
        
        # Simple text: percentage and n
        stats_text = f"{percentage:.1f}%\nn={data['n']:,}"
        
        # BETTER FIX: Position text in the middle of the VISIBLE RING
        # For each ring, find the midpoint between this ellipse top and next inner ellipse top
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
        # Draw label box
        box = Rectangle((label_box_x, current_y), label_box_width, label_box_height,
                       facecolor='white', edgecolor='black', 
                       linewidth=2)
        ax.add_patch(box)
        
        # Label text
        ax.text(label_box_x + label_box_width/2, current_y + label_box_height/2, 
               data['label'],
               ha='center', va='center', fontsize=10, 
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
    
    # Handle splits on the RIGHT side
    all_splits = []
    for idx, data in enumerate(ellipses_data):
        if data['split']:
            for split in data['split']:
                all_splits.append({
                    'split': split,
                    'ellipse_data': data,
                    'ellipse_index': idx
                })
    
    if all_splits:
        header_x = 19
        header_y = 14.5
        header_width = 4.5
        header_height = 0.9
        
        header = Rectangle((header_x, header_y), header_width, header_height,
                          facecolor='white', edgecolor='black', linewidth=2.5)
        ax.add_patch(header)
        ax.text(header_x + header_width/2, header_y + header_height/2, 
               'Data Source', ha='center', va='center', 
               fontsize=15, fontweight='bold')
        
        box_x = header_x
        box_width = header_width
        box_height = 0.85
        box_spacing = 0.5
        current_y = 13.2
        
        for box_idx, item in enumerate(all_splits):
            split_item = item['split']
            ellipse_data = item['ellipse_data']
            
            split_color = split_item.get('color', 'white')
            box_color = '#FFE6E6' if split_color == 'red' else 'white'
            
            box = Rectangle((box_x, current_y), box_width, box_height,
                           facecolor=box_color, edgecolor='black', 
                           linewidth=2)
            ax.add_patch(box)
            
            ax.text(box_x + box_width/2, current_y + box_height/2, 
                   split_item['label'],
                   ha='center', va='center', fontsize=10, 
                   fontweight='bold')
            
            a = ellipse_data['width'] / 2
            b = ellipse_data['height'] / 2
            ellipse_center_y = ellipse_data['center_y']
            
            box_center_y = current_y + box_height/2
            y_target = box_center_y - ellipse_center_y
            y_target = np.clip(y_target, -b * 0.85, b * 0.85)
            
            x_offset_squared = a**2 * (1 - (y_target**2 / b**2))
            x_offset = np.sqrt(max(0, x_offset_squared))
            
            arrow_start_x = center_x + x_offset
            arrow_start_y = ellipse_center_y + y_target
            
            arrow = FancyArrowPatch(
                (arrow_start_x, arrow_start_y),
                (box_x - 0.1, box_center_y),
                arrowstyle='->', mutation_scale=22,
                linewidth=2, color='black', zorder=50
            )
            ax.add_patch(arrow)
            
            current_y -= (box_height + box_spacing)
    
    plt.tight_layout()
    return fig