"""
Compare Multiple Models on Test Dataset
=======================================
Compare performance of multiple trained models side-by-side
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import json
import argparse

# Import the tester
from test_model_performance import ModelTester


class ModelComparator:
    """Compare multiple models on the same test dataset"""
    
    def __init__(
        self,
        test_image_dir: str,
        test_mask_dir: str,
        output_dir: str = 'comparison_results'
    ):
        """
        Initialize model comparator
        
        Args:
            test_image_dir: Directory containing test images
            test_mask_dir: Directory containing test masks
            output_dir: Directory to save comparison results
        """
        self.test_image_dir = test_image_dir
        self.test_mask_dir = test_mask_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = []
        self.results = []
        
    def add_model(
        self,
        model_path: str,
        feature_config: str,
        name: str = None
    ):
        """
        Add a model to compare
        
        Args:
            model_path: Path to model checkpoint
            feature_config: Feature configuration ('rgb', 'all', etc.)
            name: Display name for the model
        """
        if name is None:
            name = f"{feature_config}_{Path(model_path).stem}"
        
        self.models.append({
            'name': name,
            'path': model_path,
            'feature_config': feature_config
        })
        
        print(f"Added model: {name}")
    
    def run_comparison(self, device: str = None) -> Dict:
        """
        Run testing on all models
        
        Args:
            device: Device to use
            
        Returns:
            Dictionary containing comparison results
        """
        print(f"\n{'='*70}")
        print(f"COMPARING {len(self.models)} MODELS")
        print(f"{'='*70}\n")
        
        self.results = []
        
        for model_info in self.models:
            print(f"\nTesting: {model_info['name']}")
            print("-"*70)
            
            # Create tester
            tester = ModelTester(
                model_path=model_info['path'],
                test_image_dir=self.test_image_dir,
                test_mask_dir=self.test_mask_dir,
                feature_config=model_info['feature_config'],
                device=device,
                save_predictions=False,
                output_dir=self.output_dir / model_info['name']
            )
            
            # Test model
            results = tester.test()
            results['model_name'] = model_info['name']
            
            # Print brief summary
            stats = results['statistics']
            print(f"\n  Summary:")
            print(f"    IoU:       {stats['IoU']['mean']:.4f} ± {stats['IoU']['std']:.4f}")
            print(f"    Dice:      {stats['Dice']['mean']:.4f} ± {stats['Dice']['std']:.4f}")
            print(f"    F1:        {stats['F1']['mean']:.4f} ± {stats['F1']['std']:.4f}")
            print(f"    Precision: {stats['Precision']['mean']:.4f} ± {stats['Precision']['std']:.4f}")
            print(f"    Recall:    {stats['Recall']['mean']:.4f} ± {stats['Recall']['std']:.4f}")
            
            self.results.append(results)
        
        return self.create_comparison_report()
    
    def create_comparison_report(self) -> Dict:
        """Create detailed comparison report"""
        
        comparison = {
            'models': [r['model_name'] for r in self.results],
            'num_samples': self.results[0]['num_samples'],
            'metrics': {}
        }
        
        metric_names = ['IoU', 'Dice', 'F1', 'Precision', 'Recall', 'Accuracy', 'Specificity']
        
        for metric_name in metric_names:
            comparison['metrics'][metric_name] = {}
            
            for result in self.results:
                model_name = result['model_name']
                stats = result['statistics'][metric_name]
                
                comparison['metrics'][metric_name][model_name] = {
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'min': stats['min'],
                    'max': stats['max']
                }
        
        return comparison
    
    def print_comparison_table(self):
        """Print comparison table"""
        print("\n" + "="*90)
        print("MODEL COMPARISON TABLE")
        print("="*90)
        
        # Create DataFrame
        data = []
        for result in self.results:
            row = {'Model': result['model_name']}
            
            for metric in ['IoU', 'Dice', 'F1', 'Precision', 'Recall']:
                stats = result['statistics'][metric]
                row[metric] = f"{stats['mean']:.4f} ± {stats['std']:.4f}"
            
            data.append(row)
        
        df = pd.DataFrame(data)
        print("\n" + df.to_string(index=False))
        
        # Find best model for each metric
        print("\n" + "-"*90)
        print("BEST MODEL PER METRIC:")
        print("-"*90)
        
        for metric in ['IoU', 'Dice', 'F1', 'Precision', 'Recall']:
            values = [r['statistics'][metric]['mean'] for r in self.results]
            best_idx = np.argmax(values)
            best_model = self.results[best_idx]['model_name']
            best_value = values[best_idx]
            
            print(f"  {metric:12s}: {best_model:20s} ({best_value:.4f})")
        
        print("="*90)
    
    def create_comparison_plots(self):
        """Create comparison visualization plots"""
        print("\nCreating comparison plots...")
        
        metrics = ['IoU', 'Dice', 'F1', 'Precision', 'Recall']
        n_metrics = len(metrics)
        n_models = len(self.results)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
        
        # Flatten axes for easier iteration
        axes_flat = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes_flat[idx]
            
            # Extract data for this metric
            model_names = [r['model_name'] for r in self.results]
            means = [r['statistics'][metric]['mean'] for r in self.results]
            stds = [r['statistics'][metric]['std'] for r in self.results]
            
            # Create bar plot
            x = np.arange(len(model_names))
            bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8,
                         color=['steelblue', 'coral', 'lightgreen', 'gold'][:n_models])
            
            ax.set_ylabel('Score', fontweight='bold')
            ax.set_title(f'{metric}', fontweight='bold', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{mean:.3f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Hide extra subplot
        axes_flat[5].axis('off')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / 'comparison_plot.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  Saved comparison plot to: {plot_path}")
        
        plt.close()
        
        # Create box plots
        self._create_boxplots()
    
    def _create_boxplots(self):
        """Create box plots for metric distributions"""
        
        metrics = ['IoU', 'Dice', 'F1', 'Precision', 'Recall']
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(20, 5))
        fig.suptitle('Metric Distribution Comparison', fontsize=16, fontweight='bold')
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # Collect all sample values for this metric
            data_for_plot = []
            labels = []
            
            for result in self.results:
                values = [s['metrics'][metric] for s in result['sample_results']]
                data_for_plot.append(values)
                labels.append(result['model_name'])
            
            # Create box plot
            bp = ax.boxplot(data_for_plot, labels=labels, patch_artist=True)
            
            # Color boxes
            colors = ['steelblue', 'coral', 'lightgreen', 'gold']
            for patch, color in zip(bp['boxes'], colors[:len(data_for_plot)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_ylabel('Score', fontweight='bold')
            ax.set_title(f'{metric}', fontweight='bold', fontsize=12)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / 'distribution_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  Saved distribution plot to: {plot_path}")
        
        plt.close()
    
    def save_comparison(self, filename: str = 'comparison_results.json'):
        """Save comparison results to JSON"""
        output_file = self.output_dir / filename
        
        # Prepare data for saving
        save_data = {
            'models': [r['model_name'] for r in self.results],
            'results': self.results
        }
        
        with open(output_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\nComparison results saved to: {output_file}")
    
    def export_to_csv(self):
        """Export comparison table to CSV"""
        # Create summary DataFrame
        data = []
        for result in self.results:
            row = {
                'Model': result['model_name'],
                'Feature_Config': result['feature_config'],
                'Num_Samples': result['num_samples']
            }
            
            for metric in ['IoU', 'Dice', 'F1', 'Precision', 'Recall', 'Accuracy', 'Specificity']:
                stats = result['statistics'][metric]
                row[f'{metric}_mean'] = stats['mean']
                row[f'{metric}_std'] = stats['std']
                row[f'{metric}_min'] = stats['min']
                row[f'{metric}_max'] = stats['max']
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Save to CSV
        csv_path = self.output_dir / 'comparison_results.csv'
        df.to_csv(csv_path, index=False)
        print(f"Exported to CSV: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare multiple models on test dataset')
    
    # Test data arguments
    parser.add_argument('--test_image_dir', type=str, required=True,
                       help='Directory containing test images')
    parser.add_argument('--test_mask_dir', type=str, required=True,
                       help='Directory containing test masks')
    parser.add_argument('--output_dir', type=str, default='comparison_results',
                       help='Output directory for results')
    
    # Model arguments (can specify multiple models)
    parser.add_argument('--models', nargs='+', required=True,
                       help='Model checkpoint paths')
    parser.add_argument('--configs', nargs='+', required=True,
                       help='Feature configs for each model (rgb, all, etc.)')
    parser.add_argument('--names', nargs='+', default=None,
                       help='Display names for models (optional)')
    
    # Device
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if len(args.models) != len(args.configs):
        raise ValueError("Number of models must match number of configs")
    
    if args.names and len(args.names) != len(args.models):
        raise ValueError("If provided, number of names must match number of models")
    
    # Create comparator
    comparator = ModelComparator(
        test_image_dir=args.test_image_dir,
        test_mask_dir=args.test_mask_dir,
        output_dir=args.output_dir
    )
    
    # Add models
    for i, (model_path, config) in enumerate(zip(args.models, args.configs)):
        name = args.names[i] if args.names else None
        comparator.add_model(model_path, config, name)
    
    # Run comparison
    comparator.run_comparison(device=args.device)
    
    # Print results
    comparator.print_comparison_table()
    
    # Create visualizations
    comparator.create_comparison_plots()
    
    # Save results
    comparator.save_comparison()
    comparator.export_to_csv()
    
    print("\n✓ Comparison complete!")


if __name__ == '__main__':
    main()
