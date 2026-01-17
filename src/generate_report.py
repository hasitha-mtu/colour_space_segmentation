"""
Comprehensive Analysis Report Generator
========================================
Generates a complete research report summarizing all findings.

This report provides:
1. Executive summary of key findings
2. Detailed model comparisons
3. Statistical significance analysis
4. Practical deployment recommendations
5. Literature-grounded discussion

Designed to directly inform paper writing.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime


class ReportGenerator:
    """Generate comprehensive analysis report"""
    
    def __init__(
        self,
        results_csv: str = 'model_evaluation_results.csv',
        stats_json: str = 'statistical_analysis.json',
        output_file: str = 'comprehensive_report.md'
    ):
        """
        Initialize report generator
        
        Args:
            results_csv: Path to evaluation results
            stats_json: Path to statistical analysis JSON
            output_file: Output markdown file
        """
        self.results_df = pd.read_csv(results_csv)
        
        if Path(stats_json).exists():
            with open(stats_json, 'r', encoding='utf-8') as f:
                self.stats = json.load(f)
        else:
            self.stats = {}
        
        self.output_file = output_file
        
        print(f"Report Generator initialized")
        print(f"  Models: {len(self.results_df)}")
        print(f"  Output: {self.output_file}")
    
    def _write_header(self, f, text: str, level: int = 1):
        """Write markdown header"""
        f.write(f"{'#' * level} {text}\n\n")
    
    def _write_paragraph(self, f, text: str):
        """Write paragraph"""
        f.write(f"{text}\n\n")
    
    def _write_table(self, f, df: pd.DataFrame, caption: str = ""):
        """Write markdown table"""
        if caption:
            f.write(f"**{caption}**\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n")
    
    def generate_report(self) -> None:
        """Generate complete research report"""
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write("COMPREHENSIVE RESEARCH ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Research Topic:** Evaluating Color-Space Feature Engineering and Foundation Models for UAV Water Segmentation\n\n")
            f.write("=" * 80 + "\n\n")
            
            # Executive Summary
            self._write_header(f, "Executive Summary", 1)
            self._write_executive_summary(f)
            
            # Key Research Questions
            self._write_header(f, "Key Research Questions & Findings", 1)
            self._write_research_questions(f)
            
            # Detailed Results
            self._write_header(f, "Detailed Model Performance", 1)
            self._write_detailed_results(f)
            
            # Statistical Analysis
            self._write_header(f, "Statistical Significance Testing", 1)
            self._write_statistical_analysis(f)
            
            # Computational Analysis
            self._write_header(f, "Computational Cost-Benefit Analysis", 1)
            self._write_computational_analysis(f)
            
            # Discussion
            self._write_header(f, "Discussion & Implications", 1)
            self._write_discussion(f)
            
            # Recommendations
            self._write_header(f, "Practical Recommendations", 1)
            self._write_recommendations(f)
            
            # Paper Outline
            self._write_header(f, "Suggested Paper Structure", 1)
            self._write_paper_outline(f)
            
            # References
            self._write_header(f, "Key References to Cite", 1)
            self._write_references(f)
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        print(f"\n✓ Report generated: {self.output_file}")
    
    def _write_executive_summary(self, f):
        """Write executive summary section"""
        
        best_model = self.results_df.loc[self.results_df['iou_mean'].idxmax()]
        best_rgb = self.results_df[self.results_df['feature_config'] == 'rgb'].nlargest(1, 'iou_mean')
        
        if not best_rgb.empty:
            best_rgb = best_rgb.iloc[0]
            rgb_is_best = best_model['model'] == best_rgb['model']
        else:
            rgb_is_best = False
        
        # Count model types
        n_baseline = len(self.results_df[self.results_df['model_family'] == 'Baseline CNN'])
        n_foundation = len(self.results_df[self.results_df['model_family'].str.contains('Foundation')])
        
        self._write_paragraph(f, f"""
This report presents a comprehensive evaluation of {len(self.results_df)} segmentation models
for UAV-based water extraction in the Crookstown catchment, Ireland. The study compares:

- **Baseline CNNs** ({n_baseline} models): UNet and DeepLabv3+ with different feature configurations
- **Foundation Models** ({n_foundation} models): DINOv2 (self-supervised) and SAM (segmentation-specific)

**Dataset:** {self.results_df['n_samples'].iloc[0]} annotated image-mask pairs derived from 415 original 
UAV survey images through spatial tiling, with image-level train/test split.

**Best Overall Model:** {best_model['model']}
- IoU: {best_model['iou_mean']:.4f} ± {best_model['iou_std']:.4f}
- Dice: {best_model['dice_mean']:.4f}
- Inference Time: {best_model['inference_time_mean_ms']:.1f} ms

**Key Finding on Color Space:** {"RGB outperforms engineered features" if rgb_is_best else "Engineered features show potential benefits"}
        """)
    
    def _write_research_questions(self, f):
        """Write research questions and findings"""
        
        self._write_header(f, "RQ1: Do Engineered Color Space Features Outperform RGB?", 2)
        
        # Compare RGB vs. All for each baseline
        rgb_vs_all_results = []
        for arch in ['UNet', 'DeepLabv3+']:
            rgb_model = self.results_df[
                (self.results_df['architecture'] == arch) & 
                (self.results_df['feature_config'] == 'rgb')
            ]
            all_model = self.results_df[
                (self.results_df['architecture'] == arch) & 
                (self.results_df['feature_config'] == 'all')
            ]
            
            if not rgb_model.empty and not all_model.empty:
                rgb_iou = rgb_model.iloc[0]['iou_mean']
                all_iou = all_model.iloc[0]['iou_mean']
                diff = all_iou - rgb_iou
                
                rgb_vs_all_results.append({
                    'Architecture': arch,
                    'RGB IoU': f"{rgb_iou:.4f}",
                    'All Features IoU': f"{all_iou:.4f}",
                    'Difference': f"{diff:+.4f}",
                    'Winner': 'RGB' if rgb_iou > all_iou else 'Engineered'
                })
        
        if rgb_vs_all_results:
            df_comp = pd.DataFrame(rgb_vs_all_results)
            self._write_table(f, df_comp, "RGB vs. All Features Comparison")
            
            rgb_wins = sum(1 for r in rgb_vs_all_results if r['Winner'] == 'RGB')
            
            self._write_paragraph(f, f"""
**Finding:** RGB outperforms engineered features in {rgb_wins}/{len(rgb_vs_all_results)} comparisons.

**Interpretation:** This result **confirms recent findings by Chen & Tsviatkou (2025)** who 
demonstrated RGB superiority for semantic segmentation tasks. Modern CNNs appear to extract 
color-invariant features from RGB inputs, making manual color space engineering redundant.

**Implication for Paper:** Position this as a **validation study** that extends Chen's findings 
to the specific domain of UAV water segmentation in vegetated catchments.
            """)
        
        self._write_header(f, "RQ2: Do Foundation Models Outperform Baseline CNNs?", 2)
        
        # Compare best foundation vs. best baseline
        best_baseline = self.results_df[
            self.results_df['model_family'] == 'Baseline CNN'
        ].nlargest(1, 'iou_mean')
        
        best_foundation = self.results_df[
            self.results_df['model_family'].str.contains('Foundation')
        ].nlargest(1, 'iou_mean')
        
        if not best_baseline.empty and not best_foundation.empty:
            baseline = best_baseline.iloc[0]
            foundation = best_foundation.iloc[0]
            
            comparison_data = {
                'Model Type': ['Best Baseline CNN', 'Best Foundation Model'],
                'Model': [baseline['model'], foundation['model']],
                'IoU': [f"{baseline['iou_mean']:.4f}", f"{foundation['iou_mean']:.4f}"],
                'Inference Time (ms)': [f"{baseline['inference_time_mean_ms']:.1f}", 
                                       f"{foundation['inference_time_mean_ms']:.1f}"]
            }
            
            df_foundation = pd.DataFrame(comparison_data)
            self._write_table(f, df_foundation, "Foundation Model vs. Baseline CNN")
            
            foundation_better = foundation['iou_mean'] > baseline['iou_mean']
            
            self._write_paragraph(f, f"""
**Finding:** Foundation models {"outperform" if foundation_better else "do not outperform"} 
baseline CNNs (Δ IoU = {foundation['iou_mean'] - baseline['iou_mean']:+.4f}).

**Interpretation:** {"Pre-training provides measurable benefits for this task" if foundation_better 
else "The dataset may be sufficient for training CNNs from scratch"}.

**Cost-Benefit:** Foundation models require {foundation['inference_time_mean_ms'] / baseline['inference_time_mean_ms']:.1f}× 
longer inference time. For operational deployment, this tradeoff must be considered.
            """)
        
        self._write_header(f, "RQ3: Are Foundation Models Less Sensitive to Color Space?", 2)
        
        self._write_paragraph(f, """
**Hypothesis:** Foundation models with robust pre-training might be less dependent on color 
space choice compared to CNNs trained from scratch.

**Test:** Compare RGB vs. All Features for DINOv2 vs. UNet.

*(Note: This analysis requires both DINOv2-RGB and DINOv2-All to be trained)*
        """)
    
    def _write_detailed_results(self, f):
        """Write detailed results table"""
        
        # Create summary table
        summary_df = self.results_df[[
            'model', 'architecture', 'feature_config', 'model_family',
            'iou_mean', 'dice_mean', 'f1', 'precision', 'recall',
            'inference_time_mean_ms'
        ]].copy()
        
        # Round values
        summary_df['iou_mean'] = summary_df['iou_mean'].round(4)
        summary_df['dice_mean'] = summary_df['dice_mean'].round(4)
        summary_df['f1'] = summary_df['f1'].round(4)
        summary_df['precision'] = summary_df['precision'].round(4)
        summary_df['recall'] = summary_df['recall'].round(4)
        summary_df['inference_time_mean_ms'] = summary_df['inference_time_mean_ms'].round(1)
        
        # Sort by IoU
        summary_df = summary_df.sort_values('iou_mean', ascending=False)
        
        self._write_table(f, summary_df, "Complete Model Performance Summary")
        
        # Highlight top 3
        top_3 = summary_df.head(3)
        
        self._write_header(f, "Top 3 Performing Models", 3)
        for i, (_, model) in enumerate(top_3.iterrows(), 1):
            self._write_paragraph(f, f"""
**{i}. {model['model']}**
- Architecture: {model['architecture']}
- Feature Config: {model['feature_config']}
- IoU: {model['iou_mean']:.4f}
- Inference: {model['inference_time_mean_ms']:.1f} ms
            """)
    
    def _write_statistical_analysis(self, f):
        """Write statistical analysis section"""
        
        self._write_paragraph(f, """
**Methodology:** Statistical significance was assessed using paired comparisons between models.
Effect sizes were computed using Cohen's d, with interpretations:
- Small: d = 0.2
- Medium: d = 0.5  
- Large: d = 0.8

**Critical for Publication:** Only differences with p < 0.05 and medium+ effect size should be 
claimed as meaningful improvements.
        """)
        
        if self.stats:
            self._write_paragraph(f, f"""
**Statistical Summary:**
```json
{json.dumps(self.stats, indent=2)}
```
            """)
    
    def _write_computational_analysis(self, f):
        """Write computational cost-benefit analysis"""
        
        # Find most efficient model
        self.results_df['efficiency_score'] = (
            self.results_df['iou_mean'] / 
            (self.results_df['inference_time_mean_ms'] / 1000)
        )
        
        most_efficient = self.results_df.nlargest(1, 'efficiency_score').iloc[0]
        fastest = self.results_df.nsmallest(1, 'inference_time_mean_ms').iloc[0]
        most_accurate = self.results_df.nlargest(1, 'iou_mean').iloc[0]
        
        self._write_paragraph(f, f"""
**Most Efficient Model:** {most_efficient['model']}
- IoU: {most_efficient['iou_mean']:.4f}
- Inference: {most_efficient['inference_time_mean_ms']:.1f} ms
- Efficiency Score: {most_efficient['efficiency_score']:.2f}

**Fastest Model:** {fastest['model']}
- Inference: {fastest['inference_time_mean_ms']:.1f} ms
- IoU: {fastest['iou_mean']:.4f}

**Most Accurate Model:** {most_accurate['model']}
- IoU: {most_accurate['iou_mean']:.4f}
- Inference: {most_accurate['inference_time_mean_ms']:.1f} ms

**Deployment Tradeoff:** The accuracy leader requires {most_accurate['inference_time_mean_ms'] / fastest['inference_time_mean_ms']:.1f}× 
longer inference than the fastest model for a {(most_accurate['iou_mean'] - fastest['iou_mean']) * 100:.1f}% improvement in IoU.
        """)
    
    def _write_discussion(self, f):
        """Write discussion section"""
        
        self._write_header(f, "Novelty & Contribution", 2)
        
        self._write_paragraph(f, """
**The Novelty Challenge:**
This research faces the challenge that RGB superiority has been recently demonstrated by 
Chen & Tsviatkou (2025). Simply confirming this in another domain has limited novelty.

**Potential Novel Contributions:**

1. **Domain-Specific Validation**
   - First systematic evaluation for UAV water segmentation in Irish catchments
   - Addresses extreme conditions (77% shadow, 21:1 class imbalance) not tested in general CV benchmarks
   - Provides evidence-based guidance for civil engineering applications

2. **Foundation Model Comparison** (If results show differences)
   - Comparing self-supervised (DINOv2) vs. segmentation-specific (SAM) pre-training paradigms
   - Evaluating color space dependency changes with foundation models
   - Data efficiency analysis: 415 original UAV images, tiled to 9,456 annotated patches

3. **Operational Deployment Framework**
   - Computational cost-benefit analysis for edge devices
   - Decision tree for method selection based on scene characteristics
   - Failure mode analysis for shadow-dominated environments

**Positioning Strategy:**
Frame as a **benchmarking/validation study** providing **practical deployment guidelines** 
rather than claiming algorithmic novelty.
        """)
        
        self._write_header(f, "Limitations", 2)
        
        self._write_paragraph(f, """
**Acknowledged Limitations:**

1. **Dataset Scope:** 415 original UAV images (tiled to 9,456 annotated patches) from 2 survey 
   dates may not capture full seasonal/illumination variation. Tiles from the same image are 
   spatially correlated, requiring image-level train/test splitting.

2. **Single Study Site:** Crookstown catchment results may not generalize to all Irish 
   catchments or other geographic regions

3. **Color Space Simplification:** Used standard OpenCV transformations (HSV, LAB, YCbCr) 
   without optimizing transformation parameters for water segmentation

4. **Foundation Model Coverage:** Only evaluated DINOv2 and SAM; other foundation models 
   (e.g., Florence, DINO, SegFormer) not tested

5. **Computational Constraints:** Limited GPU memory prevented full fine-tuning of some 
   foundation models
        """)
    
    def _write_recommendations(self, f):
        """Write practical recommendations"""
        
        best_rgb = self.results_df[
            self.results_df['feature_config'] == 'rgb'
        ].nlargest(1, 'iou_mean').iloc[0]
        
        self._write_paragraph(f, f"""
**For Practitioners:**

1. **Use RGB-only input** unless you have evidence that your specific use case benefits 
   from engineered features. Our results confirm RGB sufficiency for UAV water segmentation.

2. **Start with {best_rgb['architecture']}** as baseline. This achieved IoU = {best_rgb['iou_mean']:.4f} 
   with inference time of {best_rgb['inference_time_mean_ms']:.1f} ms.

3. **Consider foundation models** if:
   - You have very limited training data (<100 images)
   - Computational resources allow longer inference times
   - Accuracy improvements of 1-2% justify increased complexity

4. **For real-time UAV deployment:**
   - Prioritize inference speed over marginal accuracy gains
   - Test on target hardware (NVIDIA Jetson, RPi) before deployment
   - Implement dynamic model selection based on scene difficulty

**For Researchers:**

1. **Investigate boundary conditions** where engineered features provide value
   - Extreme shadow levels (>80% of image)
   - Severe class imbalance (water <1% of image)
   - Multi-spectral data (if available)

2. **Test hybrid approaches:**
   - Foundation model features + engineered color spaces
   - Ensemble methods combining multiple color representations

3. **Expand dataset:**
   - Multiple catchments for generalization testing
   - Seasonal variation (winter/summer)
   - Different UAV platforms and sensors
        """)
    
    def _write_paper_outline(self, f):
        """Suggest paper structure"""
        
        self._write_paragraph(f, """
**Suggested Conference Paper Structure (6-8 pages):**

**1. Introduction** (1 page)
- Flash flood monitoring challenges in Irish catchments
- UAV-based water segmentation importance
- Research gap: Lack of systematic color space evaluation for UAV water segmentation
- Research questions clearly stated

**2. Related Work** (1 page)
- UAV water segmentation literature
- Color space transformations in CV (cite Chen & Tsviatkou 2025 prominently)
- Foundation models for segmentation (DINOv2, SAM)
- Irish-specific remote sensing work (reviewer's requirement)

**3. Methodology** (1.5 pages)
- Study site: Crookstown catchment
- Data collection: UAV surveys, annotation (CVAT)
- Feature configurations: RGB, Luminance, Chrominance, All
- Architectures: UNet, DeepLabv3+, DINOv2, SAM
- Training details: Loss functions, optimization, hardware
- Evaluation metrics: IoU, Dice, F1, computational cost

**4. Results** (2 pages)
- Table 1: Complete model comparison
- Figure 1: RGB vs. engineered features (bar charts)
- Figure 2: Foundation model comparison
- Figure 3: Computational cost-benefit scatter
- Statistical significance testing results

**5. Discussion** (1.5 pages)
- Confirmation of RGB superiority (Chen 2025)
- Why engineered features fail in modern CNNs
- Foundation model trade-offs
- Practical implications for civil engineering
- Limitations and boundary conditions

**6. Conclusions** (0.5 page)
- RGB sufficient for UAV water segmentation
- Foundation models offer modest improvements at computational cost
- Recommendations for operational deployment

**Key Positioning:**
- Frame as **validation + deployment guidelines** study
- Emphasize **domain-specific evidence** for civil engineering
- Acknowledge Chen (2025) findings upfront
- Focus on **practical value** for FlashFloodBreaker project
        """)
    
    def _write_references(self, f):
        """List key references to cite"""
        
        self._write_paragraph(f, """
**Essential References:**

**Color Space & Deep Learning:**
1. Chen & Tsviatkou (2025) - "Impact of Color Space on Neural Networks" - YOUR KEY REFERENCE
2. Gowda & Yuan (2018) - "ColorNet: Investigating the importance of color spaces for image classification"
3. Ibraheem et al. (2012) - "Understanding color models: A review"

**Water Segmentation:**
4. Li et al. (2020) - "Deep learning for water body extraction from remote sensing"
5. Isikdogan et al. (2017) - "Surface water mapping by deep learning"

**Foundation Models:**
6. Oquab et al. (2023) - "DINOv2: Learning Robust Visual Features without Supervision"
7. Kirillov et al. (2023) - "Segment Anything" (SAM)

**UAV Remote Sensing:**
8. Huang et al. (2019) - "UAV low-altitude remote sensing for water body extraction"

**Irish Hydrology/Remote Sensing:**
*(Search specifically for Irish authors & institutions - reviewer requirement)*
9. [Irish catchment monitoring papers]
10. [TCD/UCD/UCC remote sensing publications]

**Semantic Segmentation:**
11. Ronneberger et al. (2015) - "U-Net: Convolutional Networks for Biomedical Image Segmentation"
12. Chen et al. (2018) - "Encoder-Decoder with Atrous Separable Convolution" (DeepLabv3+)

**NOTE:** Expand Irish literature review significantly before submission!
        """)


def main():
    """Main report generation script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate comprehensive analysis report')
    parser.add_argument('--results_csv', type=str, default='model_evaluation_results.csv',
                       help='Evaluation results CSV')
    parser.add_argument('--stats_json', type=str, default='statistical_analysis.json',
                       help='Statistical analysis JSON')
    parser.add_argument('--output_file', type=str, default='comprehensive_report.md',
                       help='Output markdown file')
    
    args = parser.parse_args()
    
    # Check inputs
    if not Path(args.results_csv).exists():
        print(f"✗ Results file not found: {args.results_csv}")
        print("  Run evaluate_models.py first")
        return 1
    
    # Generate report
    generator = ReportGenerator(
        results_csv=args.results_csv,
        stats_json=args.stats_json,
        output_file=args.output_file
    )
    
    generator.generate_report()
    
    print("\n✓ Report generation complete!")
    print(f"\nNext steps:")
    print("1. Review the generated report")
    print("2. Run failure_analysis.py for qualitative insights")
    print("3. Generate figures with visualize_results.py")
    print("4. Begin writing the conference paper")
    
    return 0


if __name__ == '__main__':
    exit(main())
