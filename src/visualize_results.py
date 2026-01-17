"""
Visualization Script for Paper Figures
=======================================
Generate publication-quality figures for conference paper:

1. Performance comparison bar charts (RGB vs. engineered features)
2. Foundation model comparison
3. Computational cost-benefit scatter plots
4. Qualitative prediction visualizations
5. Failure mode examples

All figures follow academic publication standards:
- High resolution (300 DPI)
- Clear legends and labels
- Colorblind-friendly palettes
- Appropriate statistical annotations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple
import json

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("colorblind")
sns.set_context("paper", font_scale=1.2)

# Configure matplotlib for publication
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


class PublicationVisualizer:
    """Create publication-quality figures"""
    
    def __init__(
        self,
        results_csv: str = 'model_evaluation_results.csv',
        output_dir: str = 'figures'
    ):
        """
        Initialize visualizer
        
        Args:
            results_csv: Path to evaluation results CSV
            output_dir: Directory to save figures
        """
        self.results_df = pd.read_csv(results_csv)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"Visualizer initialized")
        print(f"  Loaded {len(self.results_df)} models")
        print(f"  Output directory: {self.output_dir}")
    
    def plot_rgb_vs_engineered(self, metric: str = 'iou_mean') -> None:
        """
        Figure 1: RGB vs. Engineered Features Comparison
        
        Key finding for paper: Does color space engineering help?
        
        Bar chart comparing RGB vs. Luminance, Chrominance, All
        for each architecture (UNet, DeepLabv3+)
        """
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        architectures = ['UNet', 'DeepLabv3+']
        feature_configs = ['rgb', 'luminance', 'chrominance', 'all']
        colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
        
        for idx, arch in enumerate(architectures):
            ax = axes[idx]
            
            # Filter data
            arch_data = self.results_df[self.results_df['architecture'] == arch]
            
            # Extract metrics
            means = []
            stds = []
            labels = []
            
            for config in feature_configs:
                config_data = arch_data[arch_data['feature_config'] == config]
                if not config_data.empty:
                    means.append(config_data[metric].iloc[0])
                    std_col = metric.replace('_mean', '_std')
                    stds.append(config_data[std_col].iloc[0] if std_col in config_data.columns else 0)
                    labels.append(config.capitalize())
            
            # Create bar chart
            x = np.arange(len(labels))
            bars = ax.bar(x, means, yerr=stds, capsize=5, 
                         color=colors[:len(labels)], alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Highlight RGB (baseline)
            bars[0].set_edgecolor('black')
            bars[0].set_linewidth(2)
            
            # Labels
            ax.set_xlabel('Feature Configuration')
            ax.set_ylabel('IoU Score')
            ax.set_title(f'{arch}')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=0)
            ax.set_ylim([0, max(means) * 1.1])
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + std,
                       f'{mean:.3f}',
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        output_path = self.output_dir / 'fig1_rgb_vs_engineered.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def plot_architecture_comparison(self) -> None:
        """
        Figure 2: Architecture Comparison (All Models)
        
        Grouped bar chart showing all architectures and feature configs
        """
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Group by architecture
        architectures = self.results_df['architecture'].unique()
        
        # Prepare data
        data_by_arch = []
        labels = []
        
        for arch in architectures:
            arch_data = self.results_df[self.results_df['architecture'] == arch]
            for _, row in arch_data.iterrows():
                data_by_arch.append({
                    'model': row['model'],
                    'architecture': row['architecture'],
                    'feature': row.get('feature_config', 'N/A'),
                    'iou': row['iou_mean'],
                    'iou_std': row['iou_std']
                })
        
        df_plot = pd.DataFrame(data_by_arch)
        
        # Create grouped bar chart
        x = np.arange(len(df_plot))
        bars = ax.bar(x, df_plot['iou'], yerr=df_plot['iou_std'],
                     capsize=3, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Color by model family
        colors = []
        for _, row in df_plot.iterrows():
            model_row = self.results_df[self.results_df['model'] == row['model']].iloc[0]
            if 'Baseline' in model_row['model_family']:
                colors.append('#3498db')
            elif 'Self-supervised' in model_row['model_family']:
                colors.append('#e74c3c')
            else:  # Segmentation-specific
                colors.append('#2ecc71')
        
        for bar, color in zip(bars, colors):
            bar.set_facecolor(color)
        
        # Labels
        ax.set_xlabel('Model')
        ax.set_ylabel('IoU Score')
        ax.set_title('Performance Comparison: All Models')
        ax.set_xticks(x)
        ax.set_xticklabels(df_plot['model'], rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#3498db', edgecolor='black', label='Baseline CNN'),
            Patch(facecolor='#e74c3c', edgecolor='black', label='DINOv2 (Self-supervised)'),
            Patch(facecolor='#2ecc71', edgecolor='black', label='SAM (Seg-specific)')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        output_path = self.output_dir / 'fig2_all_models_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def plot_computational_tradeoff(self) -> None:
        """
        Figure 3: Computational Cost-Benefit Analysis
        
        Scatter plot: IoU vs. Inference Time
        Shows Pareto frontier for deployment decisions
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Prepare data
        model_families = self.results_df['model_family'].unique()
        colors = {'Baseline CNN': '#3498db', 
                 'Foundation Model (Self-supervised)': '#e74c3c',
                 'Foundation Model (Segmentation-specific)': '#2ecc71'}
        markers = {'Baseline CNN': 'o', 
                  'Foundation Model (Self-supervised)': 's',
                  'Foundation Model (Segmentation-specific)': '^'}
        
        # Plot each model family
        for family in model_families:
            family_data = self.results_df[self.results_df['model_family'] == family]
            
            ax.scatter(
                family_data['inference_time_mean_ms'],
                family_data['iou_mean'],
                s=100,
                c=colors.get(family, 'gray'),
                marker=markers.get(family, 'o'),
                alpha=0.7,
                edgecolors='black',
                linewidth=0.5,
                label=family
            )
            
            # Annotate points
            for _, row in family_data.iterrows():
                # Only annotate best and extremes
                if (row['iou_mean'] == family_data['iou_mean'].max() or 
                    row['inference_time_mean_ms'] == family_data['inference_time_mean_ms'].min()):
                    ax.annotate(
                        row['model'].replace('-', '\n'),
                        (row['inference_time_mean_ms'], row['iou_mean']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=7, alpha=0.8
                    )
        
        # Pareto frontier (simplified - just connect best points)
        pareto_df = self.results_df.sort_values('inference_time_mean_ms')
        pareto_iou = []
        pareto_time = []
        current_max_iou = 0
        
        for _, row in pareto_df.iterrows():
            if row['iou_mean'] > current_max_iou:
                pareto_iou.append(row['iou_mean'])
                pareto_time.append(row['inference_time_mean_ms'])
                current_max_iou = row['iou_mean']
        
        if len(pareto_time) > 1:
            ax.plot(pareto_time, pareto_iou, 'k--', alpha=0.3, linewidth=1, label='Pareto frontier')
        
        # Labels
        ax.set_xlabel('Inference Time (ms)')
        ax.set_ylabel('IoU Score')
        ax.set_title('Computational Cost-Benefit Analysis')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Add zones
        ax.axhline(y=self.results_df['iou_mean'].max() * 0.95, 
                  color='green', linestyle=':', alpha=0.3, linewidth=1)
        ax.text(ax.get_xlim()[1] * 0.7, self.results_df['iou_mean'].max() * 0.96,
               '95% of best', fontsize=7, alpha=0.5)
        
        plt.tight_layout()
        output_path = self.output_dir / 'fig3_computational_tradeoff.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def plot_metric_comparison_radar(self) -> None:
        """
        Figure 4: Multi-Metric Radar Chart
        
        Compare top 3 models across multiple metrics:
        - IoU
        - Dice
        - Boundary IoU
        - F1
        - Inference Speed (inverted)
        """
        # Select top 3 models by IoU
        top_models = self.results_df.nlargest(3, 'iou_mean')
        
        # Metrics to compare (normalized to 0-1)
        metrics = ['iou_mean', 'dice_mean', 'boundary_iou_mean', 'f1']
        metric_labels = ['IoU', 'Dice', 'Boundary\nIoU', 'F1']
        
        # Normalize metrics
        normalized_data = []
        for _, model in top_models.iterrows():
            values = []
            for metric in metrics:
                max_val = self.results_df[metric].max()
                min_val = self.results_df[metric].min()
                norm_val = (model[metric] - min_val) / (max_val - min_val) if max_val > min_val else 1.0
                values.append(norm_val)
            normalized_data.append(values)
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        for idx, (_, model) in enumerate(top_models.iterrows()):
            values = normalized_data[idx]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, 
                   color=colors[idx], label=model['model'], alpha=0.7)
            ax.fill(angles, values, alpha=0.15, color=colors[idx])
        
        # Labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        plt.title('Multi-Metric Performance Comparison\n(Top 3 Models)', 
                 size=11, pad=20)
        
        output_path = self.output_dir / 'fig4_radar_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def plot_feature_importance_heatmap(self) -> None:
        """
        Figure 5: Feature Configuration Performance Heatmap
        
        Heatmap showing IoU for each architecture × feature config combination
        """
        # Pivot data
        baseline_only = self.results_df[self.results_df['model_family'] == 'Baseline CNN']
        
        if baseline_only.empty:
            print("⚠ No baseline models for heatmap")
            return
        
        pivot_data = baseline_only.pivot_table(
            values='iou_mean',
            index='architecture',
            columns='feature_config',
            aggfunc='first'
        )
        
        # Reorder columns
        col_order = ['rgb', 'luminance', 'chrominance', 'all']
        pivot_data = pivot_data[[c for c in col_order if c in pivot_data.columns]]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(8, 4))
        
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt='.4f',
            cmap='RdYlGn',
            center=pivot_data.mean().mean(),
            cbar_kws={'label': 'IoU Score'},
            linewidths=0.5,
            ax=ax
        )
        
        ax.set_xlabel('Feature Configuration')
        ax.set_ylabel('Architecture')
        ax.set_title('Performance Heatmap: Architecture × Feature Configuration')
        
        plt.tight_layout()
        output_path = self.output_dir / 'fig5_feature_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def generate_all_figures(self) -> None:
        """Generate all publication figures"""
        
        print("\n" + "="*70)
        print("GENERATING PUBLICATION FIGURES")
        print("="*70)
        
        print("\nFigure 1: RGB vs. Engineered Features...")
        self.plot_rgb_vs_engineered()
        
        print("\nFigure 2: All Models Comparison...")
        self.plot_architecture_comparison()
        
        print("\nFigure 3: Computational Tradeoff...")
        self.plot_computational_tradeoff()
        
        print("\nFigure 4: Multi-Metric Radar...")
        self.plot_metric_comparison_radar()
        
        print("\nFigure 5: Feature Configuration Heatmap...")
        self.plot_feature_importance_heatmap()
        
        print("\n" + "="*70)
        print(f"✓ All figures saved to: {self.output_dir}")
        print("="*70)
    
    def create_latex_table(self, output_file: str = 'results_table.tex') -> None:
        """
        Generate LaTeX table for paper
        
        Creates publication-ready table with model comparisons
        """
        # Select key columns
        table_df = self.results_df[[
            'model', 'architecture', 'feature_config',
            'iou_mean', 'iou_std', 'dice_mean', 'f1',
            'inference_time_mean_ms'
        ]].copy()
        
        # Round values
        table_df['iou_mean'] = table_df['iou_mean'].round(4)
        table_df['iou_std'] = table_df['iou_std'].round(4)
        table_df['dice_mean'] = table_df['dice_mean'].round(4)
        table_df['f1'] = table_df['f1'].round(4)
        table_df['inference_time_mean_ms'] = table_df['inference_time_mean_ms'].round(1)
        
        # Create IoU with std
        table_df['IoU'] = table_df.apply(
            lambda row: f"{row['iou_mean']:.4f} $\\pm$ {row['iou_std']:.4f}", axis=1
        )
        
        # Select final columns
        final_df = table_df[[
            'model', 'IoU', 'dice_mean', 'f1', 'inference_time_mean_ms'
        ]]
        final_df.columns = ['Model', 'IoU (mean ± std)', 'Dice', 'F1', 'Time (ms)']
        
        # Convert to LaTeX
        latex_str = final_df.to_latex(
            index=False,
            escape=False,
            column_format='l|c|c|c|c',
            caption='Model Performance Comparison',
            label='tab:results'
        )
        
        # Save
        output_path = self.output_dir / output_file
        with open(output_path, 'w') as f:
            f.write(latex_str)
        
        print(f"✓ LaTeX table saved to: {output_path}")


def main():
    """Main visualization script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate publication figures')
    parser.add_argument('--results_csv', type=str, default='model_evaluation_results.csv',
                       help='CSV file with evaluation results')
    parser.add_argument('--output_dir', type=str, default='figures',
                       help='Output directory for figures')
    
    args = parser.parse_args()
    
    # Check if results exist
    if not Path(args.results_csv).exists():
        print(f"✗ Results file not found: {args.results_csv}")
        print("  Run evaluate_models.py first")
        return 1
    
    # Create visualizer
    viz = PublicationVisualizer(
        results_csv=args.results_csv,
        output_dir=args.output_dir
    )
    
    # Generate all figures
    viz.generate_all_figures()
    
    # Generate LaTeX table
    viz.create_latex_table()
    
    print("\n✓ Visualization complete!")
    return 0


if __name__ == '__main__':
    exit(main())
