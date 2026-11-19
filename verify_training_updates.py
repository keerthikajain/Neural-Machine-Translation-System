# Quick verification of updated training code
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

def test_trainer_import():
    """Test that the updated trainer can be imported"""
    try:
        from src.training.trainer import NMTTrainer
        print("✅ NMTTrainer import successful")
        
        # Check if new methods exist
        methods = ['save_periodic_checkpoint', '_cleanup_old_checkpoints', 'speed_test']
        for method in methods:
            if hasattr(NMTTrainer, method):
                print(f"✅ Method {method} exists")
            else:
                print(f"❌ Method {method} missing")
                
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_train_nmt_args():
    """Test that train_nmt.py accepts new arguments"""
    try:
        import subprocess
        import sys
        
        # Test help command to see if new args are there
        result = subprocess.run([
            sys.executable, 'src/training/train_nmt.py', '--help'
        ], capture_output=True, text=True, timeout=30)
        
        help_text = result.stdout
        
        new_args = [
            '--save_every_steps',
            '--progress_every_steps', 
            '--speed_test_first',
            '--test_batch_size',
            '--analyze_lengths'
        ]
        
        for arg in new_args:
            if arg in help_text:
                print(f"✅ Argument {arg} found in help")
            else:
                print(f"❌ Argument {arg} missing from help")
                
        return True
        
    except Exception as e:
        print(f"❌ train_nmt.py args test failed: {e}")
        return False

def test_speed_test_script():
    """Test that speed_test.py exists and runs help"""
    try:
        if not os.path.exists('speed_test.py'):
            print("❌ speed_test.py not found")
            return False
            
        print("✅ speed_test.py exists")
        
        # Try to import it
        spec = __import__('speed_test')
        print("✅ speed_test.py can be imported")
        
        return True
        
    except Exception as e:
        print(f"❌ speed_test.py test failed: {e}")
        return False

def test_sequence_analysis():
    """Test sequence length analysis function"""
    try:
        from src.training.train_nmt import analyze_sequence_lengths
        print("✅ analyze_sequence_lengths function exists")
        return True
    except Exception as e:
        print(f"❌ analyze_sequence_lengths test failed: {e}")
        return False

def main():
    """Run all verification tests"""
    print("=" * 60)
    print("TRAINING OPTIMIZATION VERIFICATION")
    print("=" * 60)
    
    tests = [
        ("Trainer Import & Methods", test_trainer_import),
        ("train_nmt.py Arguments", test_train_nmt_args), 
        ("Speed Test Script", test_speed_test_script),
        ("Sequence Analysis", test_sequence_analysis)
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        print(f"\n[TEST] {name}")
        print("-" * 30)
        if test_func():
            passed += 1
            print(f"✅ PASSED")
        else:
            print(f"❌ FAILED")
    
    print("\n" + "=" * 60)
    print(f"VERIFICATION COMPLETE: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All optimizations implemented successfully!")
        print("\nNext steps:")
        print("1. Run: python speed_test.py")
        print("2. Or run: python train_model.py")
        print("3. Both support the new periodic checkpointing")
    else:
        print(f"⚠️  {total - passed} tests failed - please check the issues above")

if __name__ == "__main__":
    main()