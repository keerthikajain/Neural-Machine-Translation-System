# CPU Performance Speed Test for NMT Model
import os
import sys
import subprocess
from datetime import datetime

def main():
    """
    Run various speed tests to optimize CPU training performance
    """
    print("=" * 70)
    print("SAARTHI NMT SPEED TEST - CPU OPTIMIZATION")
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
    
    print("This script will run several speed tests:")
    print("1. Analyze sequence length distribution (max_length impact)")
    print("2. Test batch size 8 vs 16 performance") 
    print("3. Test max_length 128 vs 256 performance")
    print("4. Measure baseline performance with current settings")
    print()
    
    response = input("Run speed tests? [y/N]: ").strip().lower()
    if response not in ['y', 'yes']:
        print("Speed tests cancelled.")
        return
    
    test_dir = f'speed_tests_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    
    # Test configurations
    tests = [
        {
            'name': 'Sequence Length Analysis', 
            'args': ['--analyze_lengths']
        },
        {
            'name': 'Batch Size 8 Test',
            'args': ['--speed_test_only', '--speed_test_samples', '1000', 
                    '--test_batch_size', '8', '--checkpoint_dir', f'{test_dir}/batch8']
        },
        {
            'name': 'Batch Size 16 Test',
            'args': ['--speed_test_only', '--speed_test_samples', '1000',
                    '--test_batch_size', '16', '--checkpoint_dir', f'{test_dir}/batch16']
        },
        {
            'name': 'Max Length 128 Test (Batch 16)', 
            'args': ['--speed_test_only', '--speed_test_samples', '1000',
                    '--test_batch_size', '16', '--test_max_length', '128', 
                    '--checkpoint_dir', f'{test_dir}/len128']
        },
        {
            'name': 'Current Settings Baseline',
            'args': ['--speed_test_only', '--speed_test_samples', '1000',
                    '--checkpoint_dir', f'{test_dir}/baseline']
        }
    ]
    
    print(f"Running {len(tests)} tests...")
    print("=" * 50)
    
    for i, test in enumerate(tests, 1):
        print(f"\n[{i}/{len(tests)}] {test['name']}")
        print("-" * 40)
        
        # Build command
        cmd = [sys.executable, 'src/training/train_nmt.py'] + test['args']
        
        print(f"Command: {' '.join(cmd[2:])}")  # Show args only
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
        except subprocess.CalledProcessError as e:
            print(f"❌ Test failed with exit code {e.returncode}")
        except KeyboardInterrupt:
            print("❌ Test interrupted by user")
            break
        except Exception as e:
            print(f"❌ Test error: {e}")
    
    print("\n" + "=" * 50)
    print("SPEED TEST SUMMARY")
    print("=" * 50)
    print("Review the results above to choose optimal settings:")
    print("• Compare processing speeds (samples/sec)")
    print("• Check sequence length truncation impact") 
    print("• Consider batch size vs memory tradeoffs")
    print("• Factor in training time estimates")
    print()
    print("Recommended next steps:")
    print("1. Choose best batch size (8 or 16)")
    print("2. Decide on max_length (128 for speed, 256 for accuracy)")
    print("3. Run actual training with optimized settings")
    print(f"4. Checkpoints from tests saved in: {test_dir}/")

if __name__ == "__main__":
    main()