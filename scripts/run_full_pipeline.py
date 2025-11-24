"""
Master Pipeline Script

Runs the complete end-to-end workflow:
1. Data download/ingestion
2. Feature engineering
3. Regime detection (all models)
4. Visualization
5. Risk analysis
6. Backtesting
"""

import sys
from pathlib import Path
import subprocess
import time

def run_step(step_name, script_path, description):
    """Run a pipeline step"""
    print("\n" + "="*70)
    print(f"STEP: {step_name}")
    print("="*70)
    print(f"{description}")
    print("-"*70)
    
    script_path = Path(script_path)
    if not script_path.exists():
        print(f"[SKIP] Script not found: {script_path}")
        return False
    
    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print(f"\n[SUCCESS] {step_name} completed")
            return True
        else:
            print(f"\n[ERROR] {step_name} failed with return code {result.returncode}")
            return False
    
    except Exception as e:
        print(f"\n[ERROR] {step_name} failed: {e}")
        return False

def main():
    """Run complete pipeline"""
    print("="*70)
    print("MARKET REGIME DETECTION - FULL PIPELINE")
    print("="*70)
    print("\nThis will run the complete workflow:")
    print("  1. Feature Engineering")
    print("  2. Regime Detection (K-Means, GMM, HMM)")
    print("  3. Regime Visualizations")
    print("  4. Risk Analysis")
    print("  5. Backtesting")
    print("\n" + "="*70)
    
    # Auto-continue when run non-interactively
    import sys
    if sys.stdin.isatty():
        # Interactive mode - ask for confirmation
        response = input("\nContinue? (y/n): ")
        if response.lower() != 'y':
            print("Pipeline cancelled.")
            return
    else:
        # Non-interactive mode - auto-continue
        print("\n[Non-interactive mode] Starting pipeline...")
    
    start_time = time.time()
    results = {}
    
    # Step 1: Feature Engineering
    results['features'] = run_step(
        "Feature Engineering",
        "src/data_processing/features.py",
        "Creating features from raw price data"
    )
    
    # Step 2: Clustering Models
    results['clustering'] = run_step(
        "Clustering Models (K-Means, GMM)",
        "src/models/clustering.py",
        "Detecting regimes using K-Means and GMM"
    )
    
    # Step 3: HMM Model
    results['hmm'] = run_step(
        "Hidden Markov Model",
        "src/models/hmm_model.py",
        "Detecting regimes using HMM"
    )
    
    # Step 4: Visualizations
    results['visualizations'] = run_step(
        "Regime Visualizations",
        "scripts/visualize_regimes.py",
        "Creating regime visualization plots"
    )
    
    # Step 5: Risk Analysis
    results['risk'] = run_step(
        "Risk Analytics",
        "src/risk_analytics/metrics.py",
        "Computing risk metrics"
    )
    
    # Step 6: Backtesting
    results['backtest'] = run_step(
        "Backtesting",
        "scripts/run_backtest.py",
        "Running backtests with regime-aware strategies"
    )
    
    # Summary
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("PIPELINE SUMMARY")
    print("="*70)
    
    for step, success in results.items():
        status = "[SUCCESS]" if success else "[SKIP/FAILED]"
        print(f"  {status} {step}")
    
    success_count = sum(results.values())
    total_steps = len(results)
    
    print(f"\nCompleted: {success_count}/{total_steps} steps")
    print(f"Time elapsed: {elapsed_time/60:.1f} minutes")
    
    if success_count == total_steps:
        print("\n" + "="*70)
        print("PIPELINE COMPLETE!")
        print("="*70)
        print("\nNext steps:")
        print("  - Review visualizations in figures/regimes/ and figures/backtest/")
        print("  - Check backtest results in data/processed/backtest_summary.csv")
        print("  - Launch interactive dashboard: python app/gradio_app.py")
    else:
        print(f"\nPipeline completed with {total_steps - success_count} skipped/failed steps.")
        print("Review errors above and rerun failed steps individually.")

if __name__ == "__main__":
    main()

