#!/usr/bin/env python3
"""
Master Analysis Script
======================
Runs complete analysis pipeline:
1. Model evaluation
2. Statistical analysis
3. Visualization
4. Failure analysis (optional)
5. Report generation

Usage:
    python run_all_analysis.py [--skip-failure-analysis]
"""

import subprocess
import sys
from pathlib import Path
import argparse
from datetime import datetime


class ANSIColors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text):
    """Print formatted header"""
    print(f"\n{ANSIColors.HEADER}{'=' * 80}{ANSIColors.ENDC}")
    print(f"{ANSIColors.HEADER}{ANSIColors.BOLD}{text}{ANSIColors.ENDC}")
    print(f"{ANSIColors.HEADER}{'=' * 80}{ANSIColors.ENDC}\n")


def print_step(step_num, total_steps, description):
    """Print step indicator"""
    print(f"\n{ANSIColors.OKBLUE}{ANSIColors.BOLD}[Step {step_num}/{total_steps}]{ANSIColors.ENDC} {description}")


def print_success(message):
    """Print success message"""
    print(f"{ANSIColors.OKGREEN}✓ {message}{ANSIColors.ENDC}")


def print_error(message):
    """Print error message"""
    print(f"{ANSIColors.FAIL}✗ {message}{ANSIColors.ENDC}")


def print_warning(message):
    """Print warning message"""
    print(f"{ANSIColors.WARNING}⚠ {message}{ANSIColors.ENDC}")


def run_command(cmd, description):
    """
    Run shell command and handle errors
    
    Args:
        cmd: Command list for subprocess
        description: Description of what command does
    
    Returns:
        True if successful, False otherwise
    """
    print(f"\n  Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print_success(f"{description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"{description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print_error(f"Command not found: {cmd[0]}")
        print(f"  Make sure the script exists: {cmd[0]}")
        return False


def check_prerequisites():
    """Check that required files and directories exist"""
    print_header("Checking Prerequisites")
    
    all_ok = True
    
    # Check if scripts exist
    scripts = [
        'evaluate_models.py',
        'statistical_analysis.py',
        'visualize_results.py',
        'failure_analysis.py',
        'generate_report.py'
    ]
    
    print("Checking analysis scripts...")
    for script in scripts:
        script_path = Path(script)
        if script_path.exists():
            print_success(f"Found: {script}")
        else:
            print_error(f"Missing: {script}")
            all_ok = False
    
    # Check if trained models exist
    print("\nChecking for trained models...")
    results_dir = Path('experiments/results')
    
    if not results_dir.exists():
        print_error(f"Results directory not found: {results_dir}")
        print("  Have you trained your models?")
        all_ok = False
    else:
        # Check for at least some models
        checkpoint_dirs = list(results_dir.rglob('checkpoints'))
        
        if not checkpoint_dirs:
            print_error("No checkpoint directories found")
            print("  Train models before running analysis")
            all_ok = False
        else:
            # Count models
            n_models = sum(1 for d in checkpoint_dirs if (d / 'best_model.pth').exists())
            
            if n_models == 0:
                print_error("No trained models (best_model.pth) found")
                all_ok = False
            else:
                print_success(f"Found {n_models} trained models")
    
    # Check data directory
    print("\nChecking data directory...")
    data_dir = Path('data')
    if not data_dir.exists():
        print_error(f"Data directory not found: {data_dir}")
        all_ok = False
    else:
        print_success(f"Data directory exists: {data_dir}")
    
    if not all_ok:
        print("\n" + ANSIColors.FAIL + "Prerequisites check failed!" + ANSIColors.ENDC)
        print("\nPlease fix the issues above before running analysis.")
        return False
    
    print("\n" + ANSIColors.OKGREEN + "✓ All prerequisites satisfied" + ANSIColors.ENDC)
    return True


def main():
    """Main analysis pipeline"""
    
    parser = argparse.ArgumentParser(
        description='Run complete analysis pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full analysis
  python run_all_analysis.py
  
  # Skip failure analysis (faster)
  python run_all_analysis.py --skip-failure-analysis
  
  # Use CPU instead of GPU
  python run_all_analysis.py --device cpu
        """
    )
    
    parser.add_argument('--skip-failure-analysis', action='store_true',
                       help='Skip failure analysis (saves time)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu, default: auto-detect)')
    parser.add_argument('--results-dir', type=str, default='experiments/results',
                       help='Results directory')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Data directory')
    
    args = parser.parse_args()
    
    # Print banner
    print("\n" + ANSIColors.HEADER + "=" * 80 + ANSIColors.ENDC)
    print(ANSIColors.HEADER + ANSIColors.BOLD + 
          "UAV WATER SEGMENTATION ANALYSIS PIPELINE" + ANSIColors.ENDC)
    print(ANSIColors.HEADER + "=" * 80 + ANSIColors.ENDC)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {args.device if args.device else 'auto-detect'}")
    print(f"Skip failure analysis: {args.skip_failure_analysis}")
    
    # Check prerequisites
    if not check_prerequisites():
        return 1
    
    # Determine total steps
    total_steps = 4 if args.skip_failure_analysis else 5
    current_step = 0
    
    # Track success
    all_successful = True
    
    # Step 1: Model Evaluation
    current_step += 1
    print_step(current_step, total_steps, "Model Evaluation")
    print("Computing comprehensive metrics for all trained models...")
    
    cmd = [
        sys.executable, 'evaluate_models.py',
        '--results_dir', args.results_dir,
        '--data_dir', args.data_dir
    ]
    if args.device:
        cmd.extend(['--device', args.device])
    
    if not run_command(cmd, "Model evaluation"):
        all_successful = False
    
    # Check if results were generated
    if not Path('model_evaluation_results.csv').exists():
        print_error("Expected output file not found: model_evaluation_results.csv")
        print("  Cannot continue without evaluation results")
        return 1
    
    # Step 2: Statistical Analysis
    current_step += 1
    print_step(current_step, total_steps, "Statistical Analysis")
    print("Testing statistical significance of performance differences...")
    
    cmd = [
        sys.executable, 'statistical_analysis.py',
        '--results_csv', 'model_evaluation_results.csv',
        '--output_json', 'statistical_analysis.json'
    ]
    
    if not run_command(cmd, "Statistical analysis"):
        all_successful = False
    
    # Step 3: Visualization
    current_step += 1
    print_step(current_step, total_steps, "Visualization")
    print("Generating publication-quality figures...")
    
    cmd = [
        sys.executable, 'visualize_results.py',
        '--results_csv', 'model_evaluation_results.csv',
        '--output_dir', 'figures'
    ]
    
    if not run_command(cmd, "Visualization"):
        all_successful = False
    
    # Step 4: Failure Analysis (optional)
    if not args.skip_failure_analysis:
        current_step += 1
        print_step(current_step, total_steps, "Failure Analysis (Optional)")
        print("Analyzing model failures and generating qualitative visualizations...")
        print_warning("This step may take 30-60 minutes")
        
        cmd = [
            sys.executable, 'failure_analysis.py',
            '--data_dir', args.data_dir,
            '--output_dir', 'failure_analysis'
        ]
        if args.device:
            cmd.extend(['--device', args.device])
        
        if not run_command(cmd, "Failure analysis"):
            all_successful = False
            print_warning("Failure analysis failed, but continuing with report generation")
    else:
        print_warning("Skipping failure analysis (use without --skip-failure-analysis to include)")
    
    # Step 5: Report Generation
    current_step += 1
    print_step(current_step, total_steps, "Report Generation")
    print("Generating comprehensive analysis report...")
    
    cmd = [
        sys.executable, 'generate_report.py',
        '--results_csv', 'model_evaluation_results.csv',
        '--stats_json', 'statistical_analysis.json',
        '--output_file', 'comprehensive_report.md'
    ]
    
    if not run_command(cmd, "Report generation"):
        all_successful = False
    
    # Final summary
    print_header("Analysis Complete")
    
    if all_successful:
        print(ANSIColors.OKGREEN + ANSIColors.BOLD + "✓ ALL ANALYSES COMPLETED SUCCESSFULLY!" + ANSIColors.ENDC)
    else:
        print(ANSIColors.WARNING + "⚠ Some analyses encountered errors (see above)" + ANSIColors.ENDC)
    
    print("\n" + ANSIColors.BOLD + "Generated Files:" + ANSIColors.ENDC)
    print("─" * 80)
    
    # List generated files
    generated_files = [
        ('model_evaluation_results.csv', 'Complete evaluation metrics for all models'),
        ('statistical_analysis.json', 'Statistical significance test results'),
        ('figures/', 'Publication-quality figures (PNG + LaTeX table)'),
        ('comprehensive_report.md', 'Complete analysis report for paper writing')
    ]
    
    if not args.skip_failure_analysis:
        generated_files.append(('failure_analysis/', 'Qualitative visualizations and failure mode analysis'))
    
    for filepath, description in generated_files:
        path = Path(filepath)
        if path.exists():
            print_success(f"{filepath:40} - {description}")
        else:
            print_warning(f"{filepath:40} - NOT FOUND")
    
    print("\n" + ANSIColors.BOLD + "Next Steps:" + ANSIColors.ENDC)
    print("─" * 80)
    print("1. Read comprehensive_report.md for detailed findings")
    print("2. Review figures/ directory for publication figures")
    print("3. Check statistical_analysis.json for significance tests")
    
    if not args.skip_failure_analysis:
        print("4. Examine failure_analysis/ for qualitative insights")
    
    print("\n" + ANSIColors.BOLD + "For Paper Writing:" + ANSIColors.ENDC)
    print("─" * 80)
    print("• Use figures/results_table.tex for LaTeX table")
    print("• Reference figures/fig*.png in your paper")
    print("• Base Discussion section on comprehensive_report.md")
    print("• Cite Chen & Tsviatkou (2025) for RGB findings")
    print("• Expand Irish literature review (reviewer requirement!)")
    
    print(f"\n{ANSIColors.OKGREEN}Complete analysis available in: comprehensive_report.md{ANSIColors.ENDC}")
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return 0 if all_successful else 1


if __name__ == '__main__':
    try:
        exit_code = main()
    except KeyboardInterrupt:
        print(f"\n\n{ANSIColors.WARNING}Analysis interrupted by user{ANSIColors.ENDC}")
        exit_code = 130
    except Exception as e:
        print(f"\n{ANSIColors.FAIL}Unexpected error: {e}{ANSIColors.ENDC}")
        import traceback
        traceback.print_exc()
        exit_code = 1
    
    sys.exit(exit_code)
