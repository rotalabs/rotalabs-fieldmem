"""
Field Visualization Tools for FTCS Memory System

Provides real-time visualization of memory fields, evolution dynamics,
and memory injection/retrieval processes using matplotlib.
"""

import os
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import jax.numpy as jnp

# Set JAX to CPU for visualization
os.environ['JAX_PLATFORMS'] = 'cpu'

from ..core.fields import MemoryField


class FieldVisualizer:
    """
    Real-time visualization of FTCS memory fields.
    
    Provides heatmaps, animations, and analysis plots for understanding
    field dynamics, memory evolution, and forgetting processes.
    """
    
    def __init__(self, memory_field: MemoryField, figsize: Tuple[int, int] = (12, 8)):
        """Initialize field visualizer."""
        self.field = memory_field
        self.figsize = figsize
        
        # Create custom colormap for memory fields
        self.field_cmap = self._create_field_colormap()
        
        # History tracking for animations
        self.field_history: List[np.ndarray] = []
        self.energy_history: List[float] = []
        self.time_history: List[float] = []
        
        # Memory positions for overlay
        self.memory_positions: List[Tuple[int, int]] = []
        self.memory_importances: List[float] = []
        
    def _create_field_colormap(self) -> LinearSegmentedColormap:
        """Create custom colormap for memory field visualization."""
        colors = [
            '#000033',  # Deep blue for low values
            '#000066',
            '#003366', 
            '#006699',
            '#0099CC',  # Blue for medium-low
            '#33CCFF',  # Light blue 
            '#66FFFF',  # Cyan
            '#99FFCC',  # Light green
            '#CCFF99',  # Yellow-green
            '#FFFF66',  # Yellow for medium-high
            '#FFCC33',  # Orange
            '#FF9900',  # Deep orange
            '#FF6600',  # Red-orange
            '#FF3300',  # Red for high values
            '#CC0000',  # Deep red for maximum
        ]
        return LinearSegmentedColormap.from_list('field_memory', colors, N=256)
    
    def plot_field_state(self, title: str = "Memory Field State", 
                        show_memories: bool = True,
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot current field state as heatmap with memory overlays.
        
        Args:
            title: Plot title
            show_memories: Whether to overlay memory positions
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Get field state
        field_data = np.array(self.field.field)
        field_state = self.field.get_field_state()
        
        # Main field heatmap
        im1 = ax1.imshow(field_data, cmap=self.field_cmap, origin='lower', 
                        interpolation='bilinear')
        ax1.set_title(f"{title}\nEnergy: {field_state['energy']:.4f}")
        ax1.set_xlabel("X Position")
        ax1.set_ylabel("Y Position")
        
        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('Field Amplitude')
        
        # Overlay memory positions if available
        if show_memories and hasattr(self.field, 'memory_positions'):
            for i, ((x, y), importance) in enumerate(zip(self.memory_positions, 
                                                       self.memory_importances)):
                # Scale marker size by importance
                marker_size = 50 + importance * 100
                ax1.scatter(y, x, s=marker_size, c='white', marker='o', 
                          edgecolors='black', linewidths=2, alpha=0.8)
                ax1.annotate(f'M{i+1}', (y, x), xytext=(5, 5), 
                           textcoords='offset points', color='white', 
                           fontweight='bold', fontsize=8)
        
        # Energy distribution (absolute values)
        energy_map = field_data ** 2
        im2 = ax2.imshow(energy_map, cmap='hot', origin='lower', 
                        interpolation='bilinear')
        ax2.set_title(f"Energy Distribution\nMax: {np.max(energy_map):.4f}")
        ax2.set_xlabel("X Position")
        ax2.set_ylabel("Y Position")
        
        # Add colorbar
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label('Energy Density')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Saved field visualization to {save_path}")
        
        return fig
    
    def plot_field_evolution(self, num_steps: int = 50, 
                           step_interval: int = 5,
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot field evolution over time with energy tracking.
        
        Args:
            num_steps: Number of evolution steps to run
            step_interval: Steps between snapshots
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        print(f"= Running {num_steps} evolution steps for visualization...")
        
        # Clear history
        self.field_history.clear()
        self.energy_history.clear()
        self.time_history.clear()
        
        # Record initial state
        self.field_history.append(np.array(self.field.field))
        self.energy_history.append(self.field.compute_energy())
        self.time_history.append(self.field.time)
        
        # Evolve field and record snapshots
        for step in range(num_steps):
            self.field.step()
            
            if step % step_interval == 0:
                self.field_history.append(np.array(self.field.field))
                self.energy_history.append(self.field.compute_energy())
                self.time_history.append(self.field.time)
        
        # Create evolution plot
        num_snapshots = len(self.field_history)
        cols = min(4, num_snapshots)
        rows = (num_snapshots + cols - 1) // cols
        
        fig = plt.figure(figsize=(cols * 3, rows * 3 + 2))
        
        # Field evolution snapshots
        for i, (field_snap, energy, time) in enumerate(zip(
            self.field_history, self.energy_history, self.time_history)):
            
            ax = plt.subplot(rows + 1, cols, i + 1)
            
            im = ax.imshow(field_snap, cmap=self.field_cmap, origin='lower', 
                          interpolation='bilinear')
            ax.set_title(f"Step {i * step_interval}\nE={energy:.4f}, t={time:.2f}")
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add colorbar for first and last
            if i == 0 or i == len(self.field_history) - 1:
                plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Energy evolution plot
        ax_energy = plt.subplot(rows + 1, 1, rows + 1)
        ax_energy.plot(self.time_history, self.energy_history, 'b-', linewidth=2)
        ax_energy.set_xlabel('Time')
        ax_energy.set_ylabel('Total Energy')
        ax_energy.set_title('Energy Evolution')
        ax_energy.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"=� Saved evolution plot to {save_path}")
        
        return fig
    
    def create_evolution_animation(self, num_steps: int = 100,
                                 interval: int = 100,
                                 save_path: Optional[str] = None) -> animation.FuncAnimation:
        """
        Create animated visualization of field evolution.
        
        Args:
            num_steps: Number of animation frames
            interval: Milliseconds between frames
            save_path: Optional path to save animation (requires ffmpeg)
            
        Returns:
            Matplotlib animation object
        """
        print(f"<� Creating {num_steps}-frame animation...")
        
        # Setup figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Initialize plots
        field_data = np.array(self.field.field)
        im1 = ax1.imshow(field_data, cmap=self.field_cmap, origin='lower',
                        animated=True, interpolation='bilinear')
        ax1.set_title("Memory Field Evolution")
        ax1.set_xlabel("X Position")
        ax1.set_ylabel("Y Position")
        
        # Energy plot
        energy_line, = ax2.plot([], [], 'b-', linewidth=2)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Energy')
        ax2.set_title('Energy Evolution')
        ax2.grid(True, alpha=0.3)
        
        # Animation data
        times = []
        energies = []
        
        def animate_frame(frame):
            """Animation function for each frame."""
            # Evolve field
            self.field.step()
            
            # Update field plot
            field_data = np.array(self.field.field)
            im1.set_array(field_data)
            im1.set_clim(vmin=field_data.min(), vmax=field_data.max())
            
            # Update energy plot
            times.append(self.field.time)
            energies.append(self.field.compute_energy())
            
            energy_line.set_data(times, energies)
            if len(times) > 1:
                ax2.set_xlim(times[0], times[-1])
                ax2.set_ylim(min(energies) * 0.9, max(energies) * 1.1)
            
            return [im1, energy_line]
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, animate_frame, frames=num_steps, 
            interval=interval, blit=True, repeat=True
        )
        
        if save_path:
            try:
                anim.save(save_path, writer='ffmpeg', fps=10)
                print(f"<� Saved animation to {save_path}")
            except Exception as e:
                print(f"�  Could not save animation: {e}")
                print("   (ffmpeg required for video export)")
        
        return anim
    
    def plot_memory_injection(self, embedding: jnp.ndarray, 
                            position: Optional[Tuple[int, int]] = None,
                            importance: float = 1.0,
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize memory injection process before and after.
        
        Args:
            embedding: Memory embedding to inject
            position: Injection position (auto-selected if None)
            importance: Memory importance
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        # Capture before state
        before_field = np.array(self.field.field)
        before_energy = self.field.compute_energy()
        
        # Inject memory
        self.field.inject_memory(embedding, position, importance)
        
        # Capture after state
        after_field = np.array(self.field.field)
        after_energy = self.field.compute_energy()
        
        # Calculate difference
        diff_field = after_field - before_field
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Before injection
        im1 = axes[0, 0].imshow(before_field, cmap=self.field_cmap, origin='lower')
        axes[0, 0].set_title(f"Before Injection\nEnergy: {before_energy:.4f}")
        plt.colorbar(im1, ax=axes[0, 0])
        
        # After injection
        im2 = axes[0, 1].imshow(after_field, cmap=self.field_cmap, origin='lower')
        axes[0, 1].set_title(f"After Injection\nEnergy: {after_energy:.4f}")
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Difference (injection pattern)
        im3 = axes[1, 0].imshow(diff_field, cmap='RdBu_r', origin='lower')
        axes[1, 0].set_title(f"Injection Pattern\n�E: {after_energy - before_energy:.4f}")
        plt.colorbar(im3, ax=axes[1, 0])
        
        # Energy distribution after
        energy_map = after_field ** 2
        im4 = axes[1, 1].imshow(energy_map, cmap='hot', origin='lower')
        axes[1, 1].set_title("Energy Distribution")
        plt.colorbar(im4, ax=axes[1, 1])
        
        # Mark injection position if known
        if position:
            for ax in axes.flat:
                ax.scatter(position[1], position[0], s=100, c='white', 
                          marker='x', linewidths=3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"=� Saved injection visualization to {save_path}")
        
        return fig
    
    def add_memory_marker(self, position: Tuple[int, int], importance: float):
        """Add memory position for overlay visualization."""
        self.memory_positions.append(position)
        self.memory_importances.append(importance)
    
    def clear_memory_markers(self):
        """Clear all memory position markers."""
        self.memory_positions.clear()
        self.memory_importances.clear()
    
    def show(self):
        """Display all plots."""
        plt.show()


def quick_field_demo(field_size: Tuple[int, int] = (64, 64)) -> None:
    """
    Quick demonstration of field visualization capabilities.
    
    Args:
        field_size: Size of the memory field to create
    """
    print(f"<� FTCS Field Visualization Demo ({field_size[0]}x{field_size[1]})")
    print("=" * 50)
    
    # Import here to avoid circular imports
    from ..core.fields import FieldConfig
    from jax import random
    
    # Create field
    config = FieldConfig(shape=field_size, diffusion_rate=0.01, temperature=0.05)
    field = MemoryField(config)
    
    # Create visualizer
    viz = FieldVisualizer(field)
    
    # Inject some test memories
    print("=� Injecting test memories...")
    rng_key = random.PRNGKey(42)
    
    for i in range(3):
        embedding = random.normal(rng_key, (32,))
        position = (field_size[0]//4 + i*field_size[0]//4, field_size[1]//2)
        importance = 1.0 - i * 0.3
        
        viz.plot_memory_injection(embedding, position, importance)
        viz.add_memory_marker(position, importance)
        
        rng_key, _ = random.split(rng_key)
    
    # Show current state
    print("=� Plotting current field state...")
    viz.plot_field_state("Demo Field with 3 Memories")
    
    # Show evolution
    print("=� Plotting field evolution...")
    viz.plot_field_evolution(num_steps=30, step_interval=3)
    
    print("( Demo complete! Close the plots to continue.")
    viz.show()


if __name__ == "__main__":
    quick_field_demo()