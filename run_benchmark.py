# Controlled CPU Performance Benchmark for NMT Training
import os
import sys
import subprocess
from datetime import datetime

def main():
    """
    Run the controlled 4-test benchmark comparing batch sizes and sequence lengths
    """
    print("=" * 70)
    print("CONTROLLED CPU BENCHMARK - 4 TEST COMPARISON")
    print("=" * 70)
    
    # Check if required files exist
    required_files = [
        'data/processed/clean_train_en.txt',
        'data/processed/clean_train_hi.txt', 
        'data/processed/tokenizer/sp_model.model'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\nPlease run data preprocessing and tokenizer training first.")
        return
    
    print("This benchmark will run 4 controlled tests:")
    print("1. Baseline: batch_size=16, max_length=256")
    print("2. Batch comparison: batch_size=8, max_length=256 (only batch size changed)")
    print("3. Sequence comparison: batch_size=16, max_length=128 (only max length changed)")  
    print("4. Combined: batch_size=8, max_length=128 (both optimizations)")
    print()
    print("Each test uses:")
    print("- Same 200 sample subset (reduced for speed)")
    print("- Same 93.3M parameter model")
    print("- Same timing methodology")
    print("- Only ONE variable changes per test")
    print("- Loads samples once, reuses for all tests")
    print()
    print("Expected time: 5-10 minutes total (vs previous 60+ minutes)")
    print("Benchmark will exit after testing, no full training will start.")
    print()
    
    response = input("Run controlled benchmark? [y/N]: ").strip().lower()
    if response not in ['y', 'yes']:
        print("Benchmark cancelled.")
        return
    
    # Build benchmark command
    cmd = [
        sys.executable, 'src/training/train_nmt.py',
        '--benchmark_mode',
        '--speed_test_samples', '200',  # Reduced sample size
        '--checkpoint_dir', f'benchmark_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    ]
    
    print("Starting controlled benchmark...")
    print("Command:", ' '.join(cmd[2:]))  # Show args only
    print()
    
    # Run benchmark
    try:
        subprocess.run(cmd, check=True)
        print("\n" + "=" * 70)
        print("BENCHMARK COMPLETE")
        print("=" * 70)
        print("Results show measured performance, not promises.")
        print("Choose configuration based on actual measured speed.")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Benchmark failed with exit code {e.returncode}")
    except KeyboardInterrupt:
        print("\n❌ Benchmark interrupted by user")
    except Exception as e:
        print(f"❌ Benchmark error: {e}")

if __name__ == "__main__":
    main()