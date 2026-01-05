"""
Feature Importance Analysis for River Segmentation
===================================================
Quantifies the relative contribution of luminance vs chrominance features
using Random Forest classifiers and SHAP values.

This analysis addresses the research question:
"Do engineered color-space features provide practical benefits over RGB-only inputs?"

Methods:
1. Random Forest - Mean Decrease in Impurity (MDI)
2. Permutation Importance
3. SHAP (SHapley Additive exPlanations) values
4. Luminance vs Chrominance group analysis
"""

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import shap

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data.feature_extraction import FeatureExtractor


class FeatureImportanceAnalyzer:
    """
    Analyze feature importance for water segmentation
    
    Compares luminance features (L_LAB, L_range, L_texture) vs
    chrominance features (H, S, a, b, Cb, Cr, Intensity) for
    water detection in shadow-dominated UAV imagery.
    """
    
    def __init__(
        self,
        output_dir: str = 'experiments/feature_importance',
        random_state: int = 42
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.feature_extractor = FeatureExtractor()
        self.feature_names = self.feature_extractor.feature_names
        
        # Feature groups
        self.luminance_features = ['L_LAB', 'L_range', 'L_texture']
        self.chrominance_features = ['H_HSV', 'S_HSV', 'a_LAB', 'b_LAB', 
                                     'Cb_YCbCr', 'Cr_YCbCr', 'Intensity']
        
        self.random_state = random_state
        self.rf_model = None
        
        print(f"Feature Importance Analyzer initialized")
        print(f"Output directory: {self.output_dir}")
    
    def load_and_prepare_data(
        self,
        data_dir: str,
        max_images: Optional[int] = None,
        samples_per_image: int = 5000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load images and extract pixel-level features
        
        Args:
            data_dir: Directory with images/ and masks/ subdirectories
            max_images: Maximum number of images to process (None = all)
            samples_per_image: Number of pixels to sample per image
        
        Returns:
            X: Feature matrix (N_samples, 10)
            y: Labels (N_samples,) - 0=non-water, 1=water
        """
        print(f"\nLoading data from {data_dir}")
        
        data_path = Path(data_dir)
        image_paths = sorted(list((data_path / 'images').glob('*.png')) + 
                           list((data_path / 'images').glob('*.jpg')))
        
        if max_images:
            image_paths = image_paths[:max_images]
        
        print(f"Processing {len(image_paths)} images...")
        
        all_features = []
        all_labels = []
        
        for img_path in tqdm(image_paths, desc="Extracting features"):
            # Load image and mask
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            mask_path = data_path / 'masks' / img_path.name
            if not mask_path.exists():
                mask_path = data_path / 'masks' / f"{img_path.stem}.png"
            
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.uint8)
            
            # Extract features
            features = self.feature_extractor.extract_all_features(image)
            features = self.feature_extractor.normalize_features(features)
            
            # Flatten spatial dimensions
            H, W, C = features.shape
            features_flat = features.reshape(-1, C)  # (H*W, 10)
            labels_flat = mask.flatten()  # (H*W,)
            
            # Stratified sampling to balance water/non-water
            water_indices = np.where(labels_flat == 1)[0]
            non_water_indices = np.where(labels_flat == 0)[0]
            
            # Sample equal amounts from each class
            n_samples_per_class = samples_per_image // 2
            
            if len(water_indices) > 0 and len(non_water_indices) > 0:
                # Sample with replacement if needed
                water_sample = np.random.choice(
                    water_indices,
                    size=min(n_samples_per_class, len(water_indices)),
                    replace=False
                )
                non_water_sample = np.random.choice(
                    non_water_indices,
                    size=min(n_samples_per_class, len(non_water_indices)),
                    replace=False
                )
                
                selected_indices = np.concatenate([water_sample, non_water_sample])
                
                all_features.append(features_flat[selected_indices])
                all_labels.append(labels_flat[selected_indices])
        
        # Combine all samples
        X = np.vstack(all_features)
        y = np.concatenate(all_labels)
        
        print(f"\nDataset prepared:")
        print(f"  Total samples: {len(X):,}")
        print(f"  Water pixels: {(y == 1).sum():,} ({100 * (y == 1).mean():.2f}%)")
        print(f"  Non-water pixels: {(y == 0).sum():,} ({100 * (y == 0).mean():.2f}%)")
        print(f"  Features: {X.shape[1]}")
        
        return X, y
    
    def train_random_forest(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_estimators: int = 100,
        max_depth: int = 20,
        test_size: float = 0.2
    ) -> Dict:
        """
        Train Random Forest classifier
        
        Args:
            X: Feature matrix
            y: Labels
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            test_size: Test set proportion
        
        Returns:
            Dictionary with model and metrics
        """
        print(f"\n{'='*70}")
        print("Training Random Forest Classifier")
        print(f"{'='*70}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"Train samples: {len(X_train):,}")
        print(f"Test samples: {len(X_test):,}")
        
        # Train model
        print(f"\nTraining with {n_estimators} trees...")
        self.rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1
        )
        
        self.rf_model.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.rf_model.score(X_train, y_train)
        test_score = self.rf_model.score(X_test, y_test)
        
        print(f"\nModel Performance:")
        print(f"  Train accuracy: {train_score:.4f}")
        print(f"  Test accuracy: {test_score:.4f}")
        
        # Detailed metrics
        y_pred = self.rf_model.predict(X_test)
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Non-water', 'Water']))
        
        # Save model
        model_path = self.output_dir / 'random_forest_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.rf_model, f)
        print(f"\n✓ Model saved to {model_path}")
        
        return {
            'model': self.rf_model,
            'train_score': train_score,
            'test_score': test_score,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
    
    def compute_mdi_importance(self) -> pd.DataFrame:
        """
        Compute Mean Decrease in Impurity (MDI) feature importance
        
        Returns:
            DataFrame with feature importances
        """
        if self.rf_model is None:
            raise ValueError("Train Random Forest model first")
        
        print(f"\n{'='*70}")
        print("Computing MDI Feature Importance")
        print(f"{'='*70}")
        
        importances = self.rf_model.feature_importances_
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances,
            'group': ['Luminance' if f in self.luminance_features else 'Chrominance' 
                     for f in self.feature_names]
        })
        
        df = df.sort_values('importance', ascending=False)
        
        print("\nMDI Importance Ranking:")
        print(df.to_string(index=False))
        
        return df
    
    def compute_permutation_importance(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        n_repeats: int = 10
    ) -> pd.DataFrame:
        """
        Compute Permutation Importance
        
        Args:
            X_test: Test features
            y_test: Test labels
            n_repeats: Number of permutation repeats
        
        Returns:
            DataFrame with permutation importances
        """
        if self.rf_model is None:
            raise ValueError("Train Random Forest model first")
        
        print(f"\n{'='*70}")
        print("Computing Permutation Importance")
        print(f"{'='*70}")
        print(f"Repeats: {n_repeats}")
        
        perm_importance = permutation_importance(
            self.rf_model, X_test, y_test,
            n_repeats=n_repeats,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std,
            'group': ['Luminance' if f in self.luminance_features else 'Chrominance' 
                     for f in self.feature_names]
        })
        
        df = df.sort_values('importance_mean', ascending=False)
        
        print("\nPermutation Importance Ranking:")
        print(df.to_string(index=False))
        
        return df
    
    def compute_shap_values(
        self,
        X_sample: np.ndarray,
        max_samples: int = 1000
    ) -> Tuple[shap.Explanation, np.ndarray]:
        """
        Compute SHAP values
        
        Args:
            X_sample: Sample of features for SHAP computation
            max_samples: Maximum samples to use (SHAP is computationally expensive)
        
        Returns:
            SHAP explanation object and base values
        """
        if self.rf_model is None:
            raise ValueError("Train Random Forest model first")
        
        print(f"\n{'='*70}")
        print("Computing SHAP Values")
        print(f"{'='*70}")
        
        # Sample data for SHAP (it's expensive)
        if len(X_sample) > max_samples:
            indices = np.random.choice(len(X_sample), max_samples, replace=False)
            X_shap = X_sample[indices]
        else:
            X_shap = X_sample
        
        print(f"Computing SHAP for {len(X_shap):,} samples...")
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(self.rf_model)
        shap_values = explainer(X_shap)
        
        print("✓ SHAP values computed")
        
        return shap_values, explainer.expected_value
    
    def analyze_luminance_vs_chrominance(
        self,
        mdi_df: pd.DataFrame,
        perm_df: pd.DataFrame
    ) -> Dict:
        """
        Analyze luminance vs chrominance contribution
        
        Args:
            mdi_df: MDI importance dataframe
            perm_df: Permutation importance dataframe
        
        Returns:
            Dictionary with group analysis
        """
        print(f"\n{'='*70}")
        print("Luminance vs Chrominance Analysis")
        print(f"{'='*70}")
        
        # MDI group sums
        mdi_lum = mdi_df[mdi_df['group'] == 'Luminance']['importance'].sum()
        mdi_chrom = mdi_df[mdi_df['group'] == 'Chrominance']['importance'].sum()
        
        # Permutation group sums
        perm_lum = perm_df[perm_df['group'] == 'Luminance']['importance_mean'].sum()
        perm_chrom = perm_df[perm_df['group'] == 'Chrominance']['importance_mean'].sum()
        
        results = {
            'mdi': {
                'luminance': mdi_lum,
                'chrominance': mdi_chrom,
                'ratio': mdi_lum / mdi_chrom if mdi_chrom > 0 else float('inf')
            },
            'permutation': {
                'luminance': perm_lum,
                'chrominance': perm_chrom,
                'ratio': perm_lum / perm_chrom if perm_chrom > 0 else float('inf')
            }
        }
        
        print("\nMDI Importance:")
        print(f"  Luminance:   {mdi_lum:.4f} ({100*mdi_lum/(mdi_lum+mdi_chrom):.1f}%)")
        print(f"  Chrominance: {mdi_chrom:.4f} ({100*mdi_chrom/(mdi_lum+mdi_chrom):.1f}%)")
        print(f"  Ratio (L/C): {results['mdi']['ratio']:.2f}")
        
        print("\nPermutation Importance:")
        print(f"  Luminance:   {perm_lum:.4f} ({100*perm_lum/(perm_lum+perm_chrom):.1f}%)")
        print(f"  Chrominance: {perm_chrom:.4f} ({100*perm_chrom/(perm_lum+perm_chrom):.1f}%)")
        print(f"  Ratio (L/C): {results['permutation']['ratio']:.2f}")
        
        return results
    
    def visualize_results(
        self,
        mdi_df: pd.DataFrame,
        perm_df: pd.DataFrame,
        shap_values: Optional[shap.Explanation] = None,
        group_analysis: Optional[Dict] = None
    ):
        """
        Create comprehensive visualizations
        
        Args:
            mdi_df: MDI importance dataframe
            perm_df: Permutation importance dataframe
            shap_values: SHAP explanation object
            group_analysis: Luminance vs chrominance analysis
        """
        print(f"\n{'='*70}")
        print("Creating Visualizations")
        print(f"{'='*70}")
        
        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.dpi'] = 150
        
        # 1. MDI Importance Bar Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#3498db' if g == 'Luminance' else '#e74c3c' 
                 for g in mdi_df['group']]
        ax.barh(mdi_df['feature'], mdi_df['importance'], color=colors)
        ax.set_xlabel('MDI Importance')
        ax.set_title('Feature Importance (Mean Decrease in Impurity)')
        ax.invert_yaxis()
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#3498db', label='Luminance'),
            Patch(facecolor='#e74c3c', label='Chrominance')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'mdi_importance.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved: mdi_importance.png")
        plt.close()
        
        # 2. Permutation Importance with Error Bars
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#3498db' if g == 'Luminance' else '#e74c3c' 
                 for g in perm_df['group']]
        ax.barh(perm_df['feature'], perm_df['importance_mean'], 
               xerr=perm_df['importance_std'], color=colors, capsize=3)
        ax.set_xlabel('Permutation Importance')
        ax.set_title('Feature Importance (Permutation)')
        ax.invert_yaxis()
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'permutation_importance.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved: permutation_importance.png")
        plt.close()
        
        # 3. SHAP Summary Plot
        if shap_values is not None:
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_values,
                feature_names=self.feature_names,
                show=False,
                max_display=10
            )
            plt.tight_layout()
            plt.savefig(self.output_dir / 'shap_summary.png', dpi=150, bbox_inches='tight')
            print(f"✓ Saved: shap_summary.png")
            plt.close()
            
            # SHAP Bar Plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_values,
                feature_names=self.feature_names,
                plot_type='bar',
                show=False,
                max_display=10
            )
            plt.tight_layout()
            plt.savefig(self.output_dir / 'shap_bar.png', dpi=150, bbox_inches='tight')
            print(f"✓ Saved: shap_bar.png")
            plt.close()
        
        # 4. Luminance vs Chrominance Comparison
        if group_analysis is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # MDI
            groups = ['Luminance', 'Chrominance']
            values_mdi = [
                group_analysis['mdi']['luminance'],
                group_analysis['mdi']['chrominance']
            ]
            ax1.bar(groups, values_mdi, color=['#3498db', '#e74c3c'])
            ax1.set_ylabel('MDI Importance')
            ax1.set_title('MDI: Luminance vs Chrominance')
            ax1.set_ylim(0, max(values_mdi) * 1.2)
            
            # Add percentage labels
            total = sum(values_mdi)
            for i, v in enumerate(values_mdi):
                ax1.text(i, v + 0.02, f'{100*v/total:.1f}%', 
                        ha='center', va='bottom', fontweight='bold')
            
            # Permutation
            values_perm = [
                group_analysis['permutation']['luminance'],
                group_analysis['permutation']['chrominance']
            ]
            ax2.bar(groups, values_perm, color=['#3498db', '#e74c3c'])
            ax2.set_ylabel('Permutation Importance')
            ax2.set_title('Permutation: Luminance vs Chrominance')
            ax2.set_ylim(0, max(values_perm) * 1.2)
            
            # Add percentage labels
            total = sum(values_perm)
            for i, v in enumerate(values_perm):
                ax2.text(i, v + max(values_perm)*0.02, f'{100*v/total:.1f}%', 
                        ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'luminance_vs_chrominance.png', 
                       dpi=150, bbox_inches='tight')
            print(f"✓ Saved: luminance_vs_chrominance.png")
            plt.close()
        
        print(f"\n✓ All visualizations saved to {self.output_dir}")


def main():
    """Run complete feature importance analysis"""
    
    import argparse
    parser = argparse.ArgumentParser(
        description='Feature Importance Analysis for River Segmentation'
    )
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Data directory with images/ and masks/')
    parser.add_argument('--max_images', type=int, default=None,
                       help='Maximum images to process (None = all)')
    parser.add_argument('--samples_per_image', type=int, default=5000,
                       help='Pixels to sample per image')
    parser.add_argument('--output_dir', type=str, 
                       default='experiments/feature_importance',
                       help='Output directory')
    parser.add_argument('--n_estimators', type=int, default=100,
                       help='Number of RF trees')
    parser.add_argument('--max_depth', type=int, default=20,
                       help='Maximum tree depth')
    parser.add_argument('--shap_samples', type=int, default=1000,
                       help='Samples for SHAP computation')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = FeatureImportanceAnalyzer(
        output_dir=args.output_dir
    )
    
    # Load and prepare data
    X, y = analyzer.load_and_prepare_data(
        data_dir=args.data_dir,
        max_images=args.max_images,
        samples_per_image=args.samples_per_image
    )
    
    # Train Random Forest
    rf_results = analyzer.train_random_forest(
        X, y,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth
    )
    
    # Compute feature importances
    mdi_df = analyzer.compute_mdi_importance()
    perm_df = analyzer.compute_permutation_importance(
        rf_results['X_test'],
        rf_results['y_test'],
        n_repeats=10
    )
    
    # Compute SHAP values
    shap_values, _ = analyzer.compute_shap_values(
        rf_results['X_test'],
        max_samples=args.shap_samples
    )
    
    # Analyze luminance vs chrominance
    group_analysis = analyzer.analyze_luminance_vs_chrominance(mdi_df, perm_df)
    
    # Create visualizations
    analyzer.visualize_results(mdi_df, perm_df, shap_values, group_analysis)
    
    # Save results to CSV
    mdi_df.to_csv(analyzer.output_dir / 'mdi_importance.csv', index=False)
    perm_df.to_csv(analyzer.output_dir / 'permutation_importance.csv', index=False)
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {analyzer.output_dir}")
    print("\nKey Findings:")
    print(f"  Luminance/Chrominance Ratio (MDI): {group_analysis['mdi']['ratio']:.2f}")
    print(f"  Luminance/Chrominance Ratio (Perm): {group_analysis['permutation']['ratio']:.2f}")


if __name__ == '__main__':
    main()
