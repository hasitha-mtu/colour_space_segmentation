"""
Example Usage: Testing Your DeepLabv3+ Models
=============================================
Quick start examples for testing RGB vs All Channels models
"""

# =============================================================================
# EXAMPLE 1: Test Single Model (RGB)
# =============================================================================

# Command line:
"""
python test_model_performance.py --model_path experiments/results/baseline/deeplabv3plus/rgb/checkpoints/best_model.pth --test_image_dir dataset/test/images --test_mask_dir dataset/test/masks --feature_config rgb --save_predictions --output_dir experiments/test_results/deeplabv3plus/rgb
"""

# Expected output:
"""
==================================================================
Model Performance Testing
==================================================================
Model: best_model.pth
Feature config: rgb
Device: cuda
Test images: dataset/test/images
Test masks: dataset/test/masks
==================================================================

Loading model...
  Detected standard architecture: 3 input channels
  Model loaded successfully!
Model input channels: 3

Loading test data...
  Loaded 77 test samples

Running inference...
Testing: 100%|████████████████████| 77/77 [00:15<00:00,  4.89it/s]

Calculating statistics...

==================================================================
TEST RESULTS
==================================================================
Model: experiments/results/.../best_model.pth
Feature config: rgb
Number of samples: 77
==================================================================

PER-METRIC STATISTICS:
----------------------------------------------------------------------

IoU:
  Mean:   0.5234 ± 0.1456
  Median: 0.5401
  Range:  [0.2145, 0.7823]
  IQR:    [0.4321, 0.6234]

[... more metrics ...]

Results saved to: test_results/rgb/test_results.json

✓ Testing complete!
"""


# =============================================================================
# EXAMPLE 2: Test Single Model (All Channels)
# =============================================================================

# Command line:
"""
python test_model_performance.py --model_path experiments/results/baseline/deeplabv3plus/all/checkpoints/best_model.pth --test_image_dir dataset/test/images --test_mask_dir dataset/test/masks --feature_config all --save_predictions --output_dir experiments/test_results/deeplabv3plus/all_channels
"""

# Expected output:
"""
Loading model...
  Detected channel adapter architecture: 10 → 3 channels
  Using custom DeepLabV3Plus (channel adapter architecture)
  Model loaded successfully!
Model input channels: 10

[... testing proceeds ...]
"""


# =============================================================================
# EXAMPLE 3: Compare RGB vs All Channels
# =============================================================================

# Command line:
"""
python compare_models.py --test_image_dir dataset/test/images --test_mask_dir dataset/test/masks --models experiments/results/baseline/deeplabv3plus/rgb/checkpoints/best_model.pth experiments/results/baseline/deeplabv3plus/all/checkpoints/best_model.pth --configs rgb all --names "RGB (3ch)" "All Channels (10ch)" --output_dir experiments/test_results/deeplabv3plus/comparison_results
"""

# Expected output:
"""
==================================================================
COMPARING 2 MODELS
==================================================================

Added model: RGB (3ch)
Added model: All Channels (10ch)

Testing: RGB (3ch)
----------------------------------------------------------------------
Loading model...
  Detected standard architecture: 3 input channels
[... testing ...]

  Summary:
    IoU:       0.5234 ± 0.1456
    Dice:      0.6123 ± 0.1234
    F1:        0.6123 ± 0.1234
    Precision: 0.6234 ± 0.1345
    Recall:    0.6012 ± 0.1456

Testing: All Channels (10ch)
----------------------------------------------------------------------
Loading model...
  Detected channel adapter: 10 → 3 channels
[... testing ...]

  Summary:
    IoU:       0.5567 ± 0.1312
    Dice:      0.6401 ± 0.1189
    F1:        0.6401 ± 0.1189
    Precision: 0.6567 ± 0.1234
    Recall:    0.6234 ± 0.1345

==================================================================
MODEL COMPARISON TABLE
==================================================================

Model                IoU              Dice             F1               Precision        Recall
RGB (3ch)            0.5234 ± 0.1456  0.6123 ± 0.1234  0.6123 ± 0.1234  0.6234 ± 0.1345  0.6012 ± 0.1456
All Channels (10ch)  0.5567 ± 0.1312  0.6401 ± 0.1189  0.6401 ± 0.1189  0.6567 ± 0.1234  0.6234 ± 0.1345

----------------------------------------------------------------------
BEST MODEL PER METRIC:
----------------------------------------------------------------------
  IoU         : All Channels (10ch)  (0.5567)
  Dice        : All Channels (10ch)  (0.6401)
  F1          : All Channels (10ch)  (0.6401)
  Precision   : All Channels (10ch)  (0.6567)
  Recall      : All Channels (10ch)  (0.6234)
==================================================================

Creating comparison plots...
  Saved comparison plot to: comparison_results/comparison_plot.png
  Saved distribution plot to: comparison_results/distribution_comparison.png

Comparison results saved to: comparison_results/comparison_results.json
Exported to CSV: comparison_results/comparison_results.csv

✓ Comparison complete!
"""


# =============================================================================
# EXAMPLE 4: Python API Usage
# =============================================================================

# You can also use the scripts programmatically:

from test_model_performance import ModelTester

# Create tester
tester = ModelTester(
    model_path='experiments/results/baseline/deeplabv3plus/rgb/checkpoints/best_model.pth',
    test_image_dir='dataset/test/images',
    test_mask_dir='dataset/test/masks',
    feature_config='rgb',
    device='cuda',
    save_predictions=True,
    output_dir='test_results/rgb'
)

# Run testing
results = tester.test()

# Access results programmatically
print(f"Mean IoU: {results['statistics']['IoU']['mean']:.4f}")
print(f"Mean Dice: {results['statistics']['Dice']['mean']:.4f}")

# Save results
tester.print_results(results)
tester.save_results(results)


# =============================================================================
# EXAMPLE 5: Compare Multiple Configurations
# =============================================================================

from compare_models import ModelComparator

# Create comparator
comparator = ModelComparator(
    test_image_dir='dataset/test/images',
    test_mask_dir='dataset/test/masks',
    output_dir='comparison_results'
)

# Add models
comparator.add_model(
    model_path='path/to/rgb_model.pth',
    feature_config='rgb',
    name='RGB Only'
)

comparator.add_model(
    model_path='path/to/luminance_model.pth',
    feature_config='luminance',
    name='Luminance Features'
)

comparator.add_model(
    model_path='path/to/chrominance_model.pth',
    feature_config='chrominance',
    name='Chrominance Features'
)

comparator.add_model(
    model_path='path/to/all_model.pth',
    feature_config='all',
    name='All Features'
)

# Run comparison
comparison_results = comparator.run_comparison(device='cuda')

# Print results
comparator.print_comparison_table()

# Create visualizations
comparator.create_comparison_plots()

# Save results
comparator.save_comparison()
comparator.export_to_csv()


# =============================================================================
# EXAMPLE 6: Batch Testing Script
# =============================================================================

# Save this as test_all_models.sh:
"""
#!/bin/bash

# Test all trained models

echo "Testing RGB model..."
python test_model_performance.py \
    --model_path experiments/results/baseline/deeplabv3plus/rgb/checkpoints/best_model.pth \
    --test_image_dir dataset/test/images \
    --test_mask_dir dataset/test/masks \
    --feature_config rgb \
    --save_predictions \
    --output_dir test_results/rgb

echo "Testing Luminance model..."
python test_model_performance.py \
    --model_path experiments/results/baseline/deeplabv3plus/luminance/checkpoints/best_model.pth \
    --test_image_dir dataset/test/images \
    --test_mask_dir dataset/test/masks \
    --feature_config luminance \
    --save_predictions \
    --output_dir test_results/luminance

echo "Testing Chrominance model..."
python test_model_performance.py \
    --model_path experiments/results/baseline/deeplabv3plus/chrominance/checkpoints/best_model.pth \
    --test_image_dir dataset/test/images \
    --test_mask_dir dataset/test/masks \
    --feature_config chrominance \
    --save_predictions \
    --output_dir test_results/chrominance

echo "Testing All Channels model..."
python test_model_performance.py \
    --model_path experiments/results/baseline/deeplabv3plus/all/checkpoints/best_model.pth \
    --test_image_dir dataset/test/images \
    --test_mask_dir dataset/test/masks \
    --feature_config all \
    --save_predictions \
    --output_dir test_results/all

echo "Comparing all models..."
python compare_models.py \
    --test_image_dir dataset/test/images \
    --test_mask_dir dataset/test/masks \
    --models \
        experiments/results/baseline/deeplabv3plus/rgb/checkpoints/best_model.pth \
        experiments/results/baseline/deeplabv3plus/luminance/checkpoints/best_model.pth \
        experiments/results/baseline/deeplabv3plus/chrominance/checkpoints/best_model.pth \
        experiments/results/baseline/deeplabv3plus/all/checkpoints/best_model.pth \
    --configs rgb luminance chrominance all \
    --names "RGB" "Luminance" "Chrominance" "All Features" \
    --output_dir comparison_results

echo "Done! Check results in test_results/ and comparison_results/"
"""

# Make executable: chmod +x test_all_models.sh
# Run: ./test_all_models.sh


# =============================================================================
# NOTES
# =============================================================================

"""
File Organization After Testing:

test_results/
├── rgb/
│   ├── test_results.json
│   ├── predictions/
│   │   ├── sample_001_pred.png
│   │   └── ...
│   └── probabilities/
│       ├── sample_001_prob.png
│       └── ...
├── all_channels/
│   └── ...
└── ...

comparison_results/
├── comparison_results.json
├── comparison_results.csv
├── comparison_plot.png
├── distribution_comparison.png
├── RGB (3ch)/
│   └── test_results.json
└── All Channels (10ch)/
    └── test_results.json
"""
