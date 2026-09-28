#!/usr/bin/env python3
"""
Visualization script for Mask-based Voting Culling algorithm
Generates a diagram showing how the voting process works
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

def create_algorithm_diagram():
    """Create a visual diagram of the voting algorithm"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Mask-based Voting Culling Algorithm', fontsize=16, fontweight='bold')
    
    # Color scheme
    bg_color = '#FF6B6B'  # Red for background
    fg_color = '#4ECDC4'  # Cyan for foreground
    gaussian_color = '#FFD93D'  # Yellow for Gaussians
    
    # Sample data
    np.random.seed(42)
    
    # Camera 1 (top row)
    ax1 = axes[0, 0]
    ax1.set_title('Camera 1 View', fontweight='bold')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    
    # Draw mask
    bg_rect = patches.Rectangle((0, 0), 10, 10, linewidth=0, 
                                 edgecolor='none', facecolor=bg_color, alpha=0.3)
    fg_circle = patches.Circle((5, 5), 3, linewidth=2, 
                               edgecolor=fg_color, facecolor=fg_color, alpha=0.5)
    ax1.add_patch(bg_rect)
    ax1.add_patch(fg_circle)
    
    # Draw projected Gaussians
    gaussians = [
        (5, 5, 'FG'),    # Center - foreground
        (3, 4, 'FG'),    # Inside - foreground
        (7, 6, 'FG'),    # Inside - foreground
        (1, 2, 'BG'),    # Outside - background
        (9, 8, 'BG'),    # Outside - background
        (2, 8, 'BG'),    # Outside - background
    ]
    
    for x, y, label in gaussians:
        color = fg_color if label == 'FG' else bg_color
        ax1.plot(x, y, 'o', color=gaussian_color, markersize=12, 
                markeredgecolor='black', markeredgewidth=2)
        ax1.text(x, y-0.5, label, ha='center', fontsize=8, fontweight='bold')
    
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.legend(['Background Mask', 'Foreground Mask'], loc='upper right')
    
    # Camera 2 (top middle)
    ax2 = axes[0, 1]
    ax2.set_title('Camera 2 View', fontweight='bold')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    
    # Slightly different mask
    bg_rect2 = patches.Rectangle((0, 0), 10, 10, linewidth=0, 
                                  edgecolor='none', facecolor=bg_color, alpha=0.3)
    fg_circle2 = patches.Circle((6, 5), 3.2, linewidth=2, 
                                edgecolor=fg_color, facecolor=fg_color, alpha=0.5)
    ax2.add_patch(bg_rect2)
    ax2.add_patch(fg_circle2)
    
    gaussians2 = [
        (6, 5, 'FG'),
        (4, 4, 'BG'),  # Different from cam1!
        (8, 6, 'FG'),
        (1, 2, 'BG'),
        (9, 8, 'BG'),
        (2, 9, 'BG'),
    ]
    
    for x, y, label in gaussians2:
        ax2.plot(x, y, 'o', color=gaussian_color, markersize=12, 
                markeredgecolor='black', markeredgewidth=2)
        ax2.text(x, y-0.5, label, ha='center', fontsize=8, fontweight='bold')
    
    ax2.set_aspect('equal')
    ax2.axis('off')
    
    # Camera 3 (top right)
    ax3 = axes[0, 2]
    ax3.set_title('Camera 3 View', fontweight='bold')
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    
    bg_rect3 = patches.Rectangle((0, 0), 10, 10, linewidth=0, 
                                  edgecolor='none', facecolor=bg_color, alpha=0.3)
    fg_circle3 = patches.Circle((5, 6), 3, linewidth=2, 
                                edgecolor=fg_color, facecolor=fg_color, alpha=0.5)
    ax3.add_patch(bg_rect3)
    ax3.add_patch(fg_circle3)
    
    gaussians3 = [
        (5, 6, 'FG'),
        (3, 5, 'FG'),
        (7, 7, 'FG'),
        (1, 1, 'BG'),
        (9, 9, 'BG'),
        (1, 9, 'BG'),
    ]
    
    for x, y, label in gaussians3:
        ax3.plot(x, y, 'o', color=gaussian_color, markersize=12, 
                markeredgecolor='black', markeredgewidth=2)
        ax3.text(x, y-0.5, label, ha='center', fontsize=8, fontweight='bold')
    
    ax3.set_aspect('equal')
    ax3.axis('off')
    
    # Voting table (bottom left)
    ax4 = axes[1, 0]
    ax4.axis('off')
    ax4.set_title('Voting Results', fontweight='bold')
    
    voting_data = [
        ['Gaussian', 'Cam1', 'Cam2', 'Cam3', 'BG Votes', 'Ratio', 'Keep?'],
        ['G1', 'FG', 'FG', 'FG', '0/3', '0.00', '✓'],
        ['G2', 'FG', 'BG', 'FG', '1/3', '0.33', '✓'],
        ['G3', 'FG', 'FG', 'FG', '0/3', '0.00', '✓'],
        ['G4', 'BG', 'BG', 'BG', '3/3', '1.00', '✗'],
        ['G5', 'BG', 'BG', 'BG', '3/3', '1.00', '✗'],
        ['G6', 'BG', 'BG', 'BG', '3/3', '1.00', '✗'],
    ]
    
    table = ax4.table(cellText=voting_data, cellLoc='center', loc='center',
                     colWidths=[0.12, 0.12, 0.12, 0.12, 0.15, 0.12, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Color the header
    for i in range(7):
        table[(0, i)].set_facecolor('#E8E8E8')
        table[(0, i)].set_text_props(weight='bold')
    
    # Color the decision column
    for i in range(1, 7):
        if voting_data[i][6] == '✓':
            table[(i, 6)].set_facecolor('#90EE90')
        else:
            table[(i, 6)].set_facecolor('#FFB6B6')
    
    # Threshold explanation (bottom middle)
    ax5 = axes[1, 1]
    ax5.axis('off')
    ax5.set_title('Threshold Settings', fontweight='bold')
    
    threshold_text = """
Threshold = 0.75 (75%)

Decision Rule:
  If BG_Ratio > 0.75:
    DELETE Gaussian
  Else:
    KEEP Gaussian

Examples:
  • 0.00 (0/3) → KEEP ✓
  • 0.33 (1/3) → KEEP ✓
  • 1.00 (3/3) → DELETE ✗

Tuning Guide:
  High (0.9): Conservative
              Keep more points
              
  Medium (0.75): Balanced
                 Recommended
                 
  Low (0.6): Aggressive
             Remove more
"""
    
    ax5.text(0.1, 0.5, threshold_text, fontsize=9, family='monospace',
            verticalalignment='center')
    
    # Summary statistics (bottom right)
    ax6 = axes[1, 2]
    ax6.axis('off')
    ax6.set_title('Impact Summary', fontweight='bold')
    
    # Create a simple bar chart
    ax6_sub = ax6.inset_axes([0.15, 0.3, 0.7, 0.5])
    
    categories = ['Initial\nGaussians', 'Deleted\n(Background)', 'Kept\n(Foreground)']
    values = [6, 3, 3]
    colors = [gaussian_color, bg_color, fg_color]
    
    bars = ax6_sub.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax6_sub.set_ylabel('Count', fontweight='bold')
    ax6_sub.set_ylim(0, 7)
    ax6_sub.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax6_sub.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val}', ha='center', va='bottom', fontweight='bold')
    
    # Add summary text
    summary_text = f"""
Results:
  • Initial: 6 Gaussians
  • Deleted: 3 (50%)
  • Kept: 3 (50%)
  
Effect:
  ✓ Cleaner mesh
  ✓ Faster rendering
  ✓ Smaller file size
"""
    ax6.text(0.1, 0.05, summary_text, fontsize=9, verticalalignment='bottom')
    
    plt.tight_layout()
    return fig

def main():
    """Generate and save the diagram"""
    output_dir = Path(__file__).parent / 'docs'
    output_dir.mkdir(exist_ok=True)
    
    fig = create_algorithm_diagram()
    
    output_file = output_dir / 'mask_voting_algorithm_diagram.png'
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Diagram saved to: {output_file}")
    
    # Also show it
    plt.show()

if __name__ == '__main__':
    main()
