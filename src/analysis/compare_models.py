"""
Model Comparison and Results Visualization
===========================================
Compares trained models across feature configurations and architectures.

Generates:
1. Performance comparison tables
2. Statistical significance tests
3. Visualization plots
4. LaTeX-formatted tables for paper
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
from scipy import stats
import json

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class ModelComparator:
    """
    Compare segmentation models across configurations
    """
    
    def __init__(self, results_dir: str = 'experiments/results'):
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / 'comparison'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Model Comparator initialized")
        print(f"Results directory: {self.results_dir}")
        print(f"Output directory: {self.output_dir}")
    
    def collect_results_from_wandb(
        self,
        project_name: str = 'uav-water-segmentation'
    ) -> pd.DataFrame:
        """
        Collect results from W&B API
        
        Args:
            project_name: W&B project name
        
        Returns:
            DataFrame with all run results
        """
        if not WANDB_AVAILABLE:
            print("Warning: wandb not available")
            return self._collect_results_from_files()
        
        print(f"\nCollecting results from W&B project: {project_name}")
        
        api = wandb.Api()
        runs = api.runs(f"your-entity/{project_name}")
        
        results = []
        for run in runs:
            if run.state == 'finished':
                results.append({
                    'run_name': run.name,
                    'model': run.config.get('model_type') or run.config.get('model'),
                    'feature_config': run.config.get('feature_config'),
                    'val_iou': run.summary.get('val/iou'),
                    'val_dice': run.summary.get('val/dice'),
                    'val_f1': run.summary.get('val/f1'),
                    'val_precision': run.summary.get('val/precision'),
                    'val_recall': run.summary.get('val/recall'),
                    'train_time': run.summary.get('_runtime', 0) / 3600,  # hours
                    'params': run.summary.get('params', 0),
                    'best_epoch': run.summary.get('best_epoch', 0)
                })
        
        df = pd.DataFrame(results)
        print(f"✓ Collected {len(df)} completed runs")
        return df
    
    def _collect_results_from_files(self) -> pd.DataFrame:
        """
        Fallback: Collect results from saved history files
        
        Returns:
            DataFrame with results
        """
        print("\nCollecting results from saved files...")
        
        results = []
        
        # Search for training_history.json files
        history_files = list(self.results_dir.rglob('training_history.json'))
        
        print(f"Found {len(history_files)} history files")
        
        for history_file in history_files:
            try:
                with open(history_file, 'r') as f:
                    history = json.load(f)
                
                # Extract model and feature config from path
                parts = history_file.parts
                model = parts[-4] if len(parts) >= 4 else 'unknown'
                feature_config = parts[-3] if len(parts) >= 3 else 'unknown'
                
                # Get best validation metrics
                val_history = history.get('val', [])
                if val_history:
                    best_idx = history.get('best_epoch', len(val_history) - 1)
                    best_metrics = val_history[min(best_idx, len(val_history) - 1)]
                    
                    results.append({
                        'model': model,
                        'feature_config': feature_config,
                        'val_iou': best_metrics.get('iou'),
                        'val_dice': best_metrics.get('dice'),
                        'val_f1': best_metrics.get('f1'),
                        'val_precision': best_metrics.get('precision'),
                        'val_recall': best_metrics.get('recall'),
                        'best_epoch': best_idx
                    })
            except Exception as e:
                print(f"  Error reading {history_file}: {e}")
        
        df = pd.DataFrame(results)
        print(f"✓ Collected {len(df)} results")
        return df
    
    def create_comparison_table(
        self,
        df: pd.DataFrame,
        metric: str = 'val_iou'
    ) -> pd.DataFrame:
        """
        Create pivot table: Models × Features
        
        Args:
            df: Results dataframe
            metric: Metric to display
        
        Returns:
            Pivot table
        """
        pivot = df.pivot_table(
            values=metric,
            index='model',
            columns='feature_config',
            aggfunc='mean'
        )
        
        # Reorder columns
        col_order = ['rgb', 'luminance', 'chrominance', 'all']
        existing_cols = [c for c in col_order if c in pivot.columns]
        pivot = pivot[existing_cols]
        
        return pivot
    
    def generate_latex_table(
        self,
        df: pd.DataFrame,
        caption: str = "Model Performance Comparison (IoU)",
        label: str = "tab:model_comparison"
    ) -> str:
        """
        Generate LaTeX table for paper
        
        Args:
            df: Comparison dataframe (pivot table)
            caption: Table caption
            label: LaTeX label
        
        Returns:
            LaTeX table string
        """
        latex = "\\begin{table}[htbp]\n"
        latex += "\\centering\n"
        latex += f"\\caption{{{caption}}}\n"
        latex += f"\\label{{{label}}}\n"
        latex += "\\begin{tabular}{l" + "c" * len(df.columns) + "}\n"
        latex += "\\toprule\n"
        
        # Header
        header = "Model & " + " & ".join(df.columns) + " \\\\\n"
        latex += header
        latex += "\\midrule\n"
        
        # Rows
        for idx, row in df.iterrows():
            values = [f"{v:.4f}" if pd.notna(v) else "-" for v in row]
            row_str = f"{idx} & " + " & ".join(values) + " \\\\\n"
            latex += row_str
        
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        return latex
    
    def plot_model_comparison(
        self,
        df: pd.DataFrame,
        metrics: List[str] = ['val_iou', 'val_dice', 'val_f1']
    ):
        """
        Create comparison plots
        
        Args:
            df: Results dataframe
            metrics: Metrics to plot
        """
        print("\nCreating comparison plots...")
        
        sns.set_style('whitegrid')
        
        # 1. Grouped bar plot: Models × Features
        for metric in metrics:
            pivot = self.create_comparison_table(df, metric)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            pivot.plot(kind='bar', ax=ax, width=0.8)
            ax.set_ylabel(metric.replace('val_', '').upper())
            ax.set_xlabel('Model')
            ax.set_title(f'{metric.replace("val_", "").upper()} by Model and Feature Config')
            ax.legend(title='Feature Config', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'{metric}_comparison.png', dpi=150, bbox_inches='tight')
            print(f"✓ Saved: {metric}_comparison.png")
            plt.close()
        
        # 2. Scatter plot: Parameters vs Performance
        if 'params' in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for model in df['model'].unique():
                model_df = df[df['model'] == model]
                ax.scatter(
                    model_df['params'] / 1e6,  # Convert to millions
                    model_df['val_iou'],
                    label=model,
                    s=100,
                    alpha=0.7
                )
            
            ax.set_xlabel('Parameters (millions)')
            ax.set_ylabel('Val IoU')
            ax.set_title('Model Efficiency: Parameters vs Performance')
            ax.legend()
            ax.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'params_vs_performance.png', dpi=150, bbox_inches='tight')
            print(f"✓ Saved: params_vs_performance.png")
            plt.close()
        
        # 3. Feature config impact across models
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for feature_config in df['feature_config'].unique():
            config_df = df[df['feature_config'] == feature_config]
            models = config_df['model'].values
            ious = config_df['val_iou'].values
            
            ax.plot(models, ious, marker='o', label=feature_config, linewidth=2)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Val IoU')
        ax.set_title('Feature Configuration Impact Across Models')
        ax.legend(title='Feature Config')
        ax.grid(alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_impact.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved: feature_impact.png")
        plt.close()
        
        print(f"\n✓ All plots saved to {self.output_dir}")
    
    def statistical_tests(
        self,
        df: pd.DataFrame,
        model_a: str,
        model_b: str,
        metric: str = 'val_iou'
    ) -> Dict:
        """
        Perform statistical significance tests
        
        Args:
            df: Results dataframe
            model_a: First model name
            model_b: Second model name
            metric: Metric to compare
        
        Returns:
            Dictionary with test results
        """
        print(f"\n{'='*70}")
        print(f"Statistical Significance Test: {model_a} vs {model_b}")
        print(f"{'='*70}")
        
        # Get data for each model
        data_a = df[df['model'] == model_a][metric].dropna().values
        data_b = df[df['model'] == model_b][metric].dropna().values
        
        if len(data_a) == 0 or len(data_b) == 0:
            print("Insufficient data for statistical test")
            return {}
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(data_a, data_b)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(data_a) - 1) * np.std(data_a)**2 + 
                              (len(data_b) - 1) * np.std(data_b)**2) / 
                             (len(data_a) + len(data_b) - 2))
        cohens_d = (np.mean(data_a) - np.mean(data_b)) / pooled_std
        
        results = {
            'model_a': model_a,
            'model_b': model_b,
            'metric': metric,
            'mean_a': np.mean(data_a),
            'mean_b': np.mean(data_b),
            'std_a': np.std(data_a),
            'std_b': np.std(data_b),
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant': p_value < 0.05
        }
        
        print(f"\n{model_a}:")
        print(f"  Mean: {results['mean_a']:.4f} ± {results['std_a']:.4f}")
        print(f"  N: {len(data_a)}")
        
        print(f"\n{model_b}:")
        print(f"  Mean: {results['mean_b']:.4f} ± {results['std_b']:.4f}")
        print(f"  N: {len(data_b)}")
        
        print(f"\nStatistical Test:")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Cohen's d: {cohens_d:.4f}")
        print(f"  Significant (α=0.05): {'Yes' if results['significant'] else 'No'}")
        
        return results
    
    def generate_summary_report(self, df: pd.DataFrame) -> str:
        """
        Generate text summary of results
        
        Args:
            df: Results dataframe
        
        Returns:
            Summary text
        """
        report = []
        report.append("="*70)
        report.append("MODEL COMPARISON SUMMARY")
        report.append("="*70)
        report.append("")
        
        # Best model overall
        best_idx = df['val_iou'].idxmax()
        best_run = df.loc[best_idx]
        report.append(f"Best Model Overall:")
        report.append(f"  Model: {best_run['model']}")
        report.append(f"  Feature Config: {best_run['feature_config']}")
        report.append(f"  Val IoU: {best_run['val_iou']:.4f}")
        report.append("")
        
        # Best per model type
        report.append("Best Configuration per Model:")
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            best_config_idx = model_df['val_iou'].idxmax()
            best_config = model_df.loc[best_config_idx]
            report.append(f"  {model}:")
            report.append(f"    Feature Config: {best_config['feature_config']}")
            report.append(f"    Val IoU: {best_config['val_iou']:.4f}")
        report.append("")
        
        # Feature config comparison
        report.append("Average Performance by Feature Config:")
        for config in df['feature_config'].unique():
            config_df = df[df['feature_config'] == config]
            avg_iou = config_df['val_iou'].mean()
            report.append(f"  {config}: {avg_iou:.4f}")
        report.append("")
        
        return "\n".join(report)


def main():
    """Run model comparison analysis"""
    
    import argparse
    parser = argparse.ArgumentParser(description='Compare trained models')
    parser.add_argument('--results_dir', type=str, default='experiments/results',
                       help='Results directory')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Collect results from W&B')
    parser.add_argument('--wandb_project', type=str, default='uav-water-segmentation',
                       help='W&B project name')
    
    args = parser.parse_args()
    
    # Initialize comparator
    comparator = ModelComparator(results_dir=args.results_dir)
    
    # Collect results
    if args.use_wandb:
        df = comparator.collect_results_from_wandb(args.wandb_project)
    else:
        df = comparator._collect_results_from_files()
    
    if len(df) == 0:
        print("\nNo results found. Train some models first!")
        return
    
    # Create comparison tables
    print("\n" + "="*70)
    print("COMPARISON TABLES")
    print("="*70)
    
    for metric in ['val_iou', 'val_dice', 'val_f1']:
        pivot = comparator.create_comparison_table(df, metric)
        print(f"\n{metric.upper()}:")
        print(pivot.to_string())
        
        # Save to CSV
        pivot.to_csv(comparator.output_dir / f'{metric}_table.csv')
        
        # Generate LaTeX
        latex = comparator.generate_latex_table(
            pivot,
            caption=f"Model Performance ({metric.replace('val_', '').upper()})",
            label=f"tab:{metric}"
        )
        with open(comparator.output_dir / f'{metric}_table.tex', 'w') as f:
            f.write(latex)
        print(f"✓ Saved LaTeX table: {metric}_table.tex")
    
    # Create visualizations
    comparator.plot_model_comparison(df)
    
    # Statistical tests (example: DINOv2 vs SAM)
    models = df['model'].unique()
    if 'dinov2' in models and 'sam' in models:
        comparator.statistical_tests(df, 'dinov2', 'sam', 'val_iou')
    
    # Generate summary report
    summary = comparator.generate_summary_report(df)
    print("\n" + summary)
    
    with open(comparator.output_dir / 'summary_report.txt', 'w') as f:
        f.write(summary)
    print(f"\n✓ Summary report saved to: {comparator.output_dir / 'summary_report.txt'}")
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {comparator.output_dir}")


if __name__ == '__main__':
    main()
