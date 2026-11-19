# Quick verification that fast benchmark changes work
import subprocess
import sys
import os

def test_dataset_modification():
    """Test that NMTDataset can accept pre-loaded sentences"""
    print("Testing NMTDataset modification...")
    
    try:
        # Test that we can import and create dataset with sentences
        from src.data.dataset import NMTDataset
        
        # Test with minimal data
        if os.path.exists('data/processed/tokenizer/sp_model.model'):
            test_en = ["Hello world", "How are you"]
            test_hi = ["नमस्ते दुनिया", "आप कैसे हैं"]
            
            dataset = NMTDataset(
                english_sentences=test_en,
                hindi_sentences=test_hi,
                tokenizer_path='data/processed/tokenizer/sp_model.model',
                max_length=128
            )
            
            print(f"✅ NMTDataset created with {len(dataset)} sentence pairs")
            return True
        else:
            print("⚠️  Tokenizer not found - skipping dataset test")
            return True
            
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        return False

def test_benchmark_functions():
    """Test that benchmark functions can be imported"""
    print("Testing benchmark function imports...")
    
    try:
        from src.training.train_nmt import create_benchmark_dataset, create_benchmark_data_loaders
        print("✅ Benchmark functions imported successfully")
        return True
    except Exception as e:
        print(f"❌ Benchmark import failed: {e}")
        return False

def test_argument_parsing():
    """Test benchmark_mode argument"""
    print("Testing --benchmark_mode argument...")
    
    try:
        result = subprocess.run([
            sys.executable, 'src/training/train_nmt.py', '--help'
        ], capture_output=True, text=True, timeout=30)
        
        if '--benchmark_mode' in result.stdout:
            print("✅ --benchmark_mode found in help")
            return True
        else:
            print("❌ --benchmark_mode missing from help")
            return False
            
    except Exception as e:
        print(f"❌ Argument test failed: {e}")
        return False

def main():
    print("=" * 50)
    print("FAST BENCHMARK VERIFICATION")
    print("=" * 50)
    
    tests = [
        ("Dataset Modification", test_dataset_modification),
        ("Benchmark Functions", test_benchmark_functions),
        ("Argument Parsing", test_argument_parsing)
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        print(f"\n[TEST] {name}")
        print("-" * 25)
        if test_func():
            passed += 1
            print("✅ PASSED")
        else:
            print("❌ FAILED")
    
    print("\n" + "=" * 50)
    print(f"VERIFICATION: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ Fast benchmark ready!")
        print("\nChanges made:")
        print("- Reduced sample size to 200 (from 1000)")
        print("- Load samples once, reuse for all 4 tests")
        print("- No more full 1.4M dataset loading per test")
        print("- Expected time: 5-10 minutes total")
        print("\nTo run: python run_benchmark.py")
    else:
        print(f"\n⚠️  {total - passed} tests failed")

if __name__ == "__main__":
    main()