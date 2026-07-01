import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

class DiagnosticVisualizer:
    def __init__(self, theme: str = "darkgrid"):
        """
        Initializes the plotting suite using Seaborn for publication-ready graphics.
        """
        sns.set_theme(style=theme)
        self.colors = sns.color_palette("deep")
        
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        self.logger = logging.getLogger("Diagnostic_Plotter")

    def plot_session(self, fused_csv_path: str, output_image_path: str = None):
        """
        Ingests the windowed features CSV and generates a stacked chronological 
        timeline of behavioral signals for diagnostic validation.
        """
        csv_path = Path(fused_csv_path)
        if not csv_path.exists():
            self.logger.error(f"Cannot find data at {csv_path}")
            return
            
        df = pd.read_csv(csv_path)
        
        # Verify required columns exist before attempting to plot (Defensive Programming)
        required_cols = ['window_id', 'left_hand_face_distance_min', 'right_hand_face_distance_min', 'motion_energy_mean']
        for col in required_cols:
            if col not in df.columns:
                self.logger.error(f"Fatal: Missing required column '{col}' in {csv_path.name}")
                return

        # Calculate absolute timeline from window timestamps if available, else estimate
        if 'start_time_ms' in df.columns:
            df['time_sec'] = df['start_time_ms'] / 1000.0
        else:
            df['time_sec'] = df['window_id'] * 2.0
        
        self.logger.info(f"Generating temporal diagnostics for {csv_path.stem}...")

        # Create a 4-tier stacked plot sharing the exact same X-axis (Timeline)
        fig, axes = plt.subplots(4, 1, figsize=(16, 16), sharex=True)
        fig.suptitle(f"Multimodal Behavioral Diagnostics: {csv_path.stem}", fontsize=18, fontweight='bold', y=0.98)

        # --- TIER 1: Hand-to-Face Distance (Pacifying / Grooming) ---
        ax1 = axes[0]
        sns.lineplot(data=df, x='time_sec', y='left_hand_face_distance_min', label='Left Hand', ax=ax1, color=self.colors[0], linewidth=2)
        sns.lineplot(data=df, x='time_sec', y='right_hand_face_distance_min', label='Right Hand', ax=ax1, color=self.colors[1], linewidth=2)
        
        ax1.set_title("Macro-Kinematics: Hand-to-Face Proximity", fontsize=14, pad=10)
        ax1.set_ylabel("Distance (Pixels)", fontsize=12)
        
        # CRITICAL: Invert the Y-axis so a "spike" UPWARDS means the hand touched the face
        ax1.invert_yaxis() 
        ax1.legend(loc="upper right")

        # --- TIER 2: Motion Energy (Agitation / Freeze Responses) ---
        ax2 = axes[1]
        sns.lineplot(data=df, x='time_sec', y='motion_energy_mean', ax=ax2, color=self.colors[2], linewidth=2)
        
        ax2.set_title("Macro-Kinematics: Global Motion Energy", fontsize=14, pad=10)
        ax2.set_ylabel("Velocity Magnitude", fontsize=12)
        ax2.fill_between(df['time_sec'], df['motion_energy_mean'], alpha=0.2, color=self.colors[2])

        # --- TIER 3: Key Action Units (Cognitive Load + Duchenne Smile) ---
        ax3 = axes[2]
        au_plot_targets = {
            "AU4_mean": ("AU4 (Brow Lowerer — Cognitive Load)", self.colors[3]),
            "AU12_mean": ("AU12 (Lip Corner Pull — Smile)", self.colors[4]),
            "AU1_mean": ("AU1 (Inner Brow Raise — Surprise)", self.colors[5]),
        }
        for col_name, (label, color) in au_plot_targets.items():
            if col_name in df.columns:
                sns.lineplot(data=df, x='time_sec', y=col_name, label=label, ax=ax3, color=color, linewidth=1.5)
        
        ax3.set_title("Micro-Geometry: Action Unit Intensity Channels", fontsize=14, pad=10)
        ax3.set_ylabel("AU Intensity", fontsize=12)
        ax3.legend(loc="upper right", fontsize=9)

        # --- TIER 4: Gaze Aversion (Z-component = looking toward/away from camera) ---
        ax4 = axes[3]
        if 'gaze_z_mean' in df.columns:
            sns.lineplot(data=df, x='time_sec', y='gaze_z_mean', ax=ax4, color=self.colors[6] if len(self.colors) > 6 else 'purple', linewidth=2)
            ax4.fill_between(df['time_sec'], df['gaze_z_mean'], alpha=0.2, color=self.colors[6] if len(self.colors) > 6 else 'purple')
        ax4.set_title("Gaze: Depth Component (Aversion Detection)", fontsize=14, pad=10)
        ax4.set_ylabel("Gaze Z", fontsize=12)
        ax4.set_xlabel("Timeline (Seconds)", fontsize=14)

        # --- Formatting and Output ---
        plt.tight_layout()
        
        if output_image_path:
            plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"✅ High-Res Diagnostic Graph saved to: {output_image_path}")
            # Force memory release to prevent Linux OOM killer during batch runs
            plt.close(fig) 
        else:
            plt.show()
            plt.close(fig)

# --- Execution Block ---
if __name__ == "__main__":
    visualizer = DiagnosticVisualizer()
    # Example test:
    # visualizer.plot_session("pipeline_system_outputs/SESSION_001/SESSION_001_fused_features.csv", "diagnostic_plot.png")