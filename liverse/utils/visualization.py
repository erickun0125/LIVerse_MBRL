"""Visualization utilities for MPC planning."""

import os
import pathlib
import re

import numpy as np


def save_video(frames, filename, fps=30):
    """Save frames as MP4 video using matplotlib animation."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    frames = np.array(frames)
    if frames.shape[-1] == 3:  # RGB format
        fig, ax = plt.subplots(figsize=(frames.shape[2] / 100, frames.shape[1] / 100))
        plt.tight_layout()
        plt.axis('off')

        def animate(i):
            ax.clear()
            ax.imshow(frames[i])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
            plt.tight_layout()
            return []

        anim = FuncAnimation(fig, animate, frames=len(frames))
        writer = FFMpegWriter(fps=fps)
        anim.save(filename, writer=writer)
        plt.close()


def plot_similarity(timesteps, similarities, diff_rewards, subtask_changes=None, filename=None):
    """Plot similarity scores and difference rewards over time and save to file."""
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Similarity graph (left y-axis)
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Similarity Score', color='tab:blue')
    ax1.plot(timesteps, similarities, 'b-', label='Similarity')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Difference-based reward graph (right y-axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Difference Reward', color='tab:red')
    ax2.plot(timesteps[1:], diff_rewards[1:], 'r-', label='Difference')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Mark subtask change points
    if subtask_changes:
        for t, subtask_id in subtask_changes:
            ax1.axvline(x=t, color='g', linestyle='--', alpha=0.5)
            ax1.text(t, max(similarities) * 0.9, f'Task {subtask_id}',
                     rotation=90, verticalalignment='top')

    # Title and legend
    plt.title('Similarity Score and Difference Reward over Time')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.close()

    return fig


def get_next_available_dir(base_dir):
    """Return automatically numbered new directory path if directory already exists."""
    if not os.path.exists(base_dir):
        return base_dir

    # Extract base directory name
    base_path = pathlib.Path(base_dir)
    parent_dir = base_path.parent
    base_name = base_path.name

    # Extract number pattern (e.g., extract 1 from planet_stepwise_results1)
    pattern = re.compile(r'(.+?)(\d*)$')
    match = pattern.match(base_name)

    if match:
        prefix = match.group(1)
        # Get the number if it exists, otherwise start from 1
        num = int(match.group(2)) if match.group(2) else 1

        # Find next number if it already exists
        while True:
            next_name = f"{prefix}{num}"
            next_path = parent_dir / next_name
            if not os.path.exists(next_path):
                return str(next_path)
            num += 1

    # If no match, just add number
    i = 1
    while True:
        next_path = parent_dir / f"{base_name}{i}"
        if not os.path.exists(next_path):
            return str(next_path)
        i += 1
