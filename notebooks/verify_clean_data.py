# Verify the cleaned dataset was saved correctly
import os

print("Verifying cleaned dataset files...")

# Check if files exist
en_file = 'data/processed/clean_train_en.txt'
hi_file = 'data/processed/clean_train_hi.txt'

if os.path.exists(en_file) and os.path.exists(hi_file):
    print("✓ Both files exist")
    
    # Count lines in each file
    with open(en_file, 'r', encoding='utf-8') as f:
        en_lines = sum(1 for line in f)
    
    with open(hi_file, 'r', encoding='utf-8') as f:
        hi_lines = sum(1 for line in f)
    
    print(f"English lines: {en_lines:,}")
    print(f"Hindi lines: {hi_lines:,}")
    
    if en_lines == hi_lines:
        print("✓ Line counts match")
    else:
        print("✗ Line counts don't match!")
    
    # Check file sizes
    en_size = os.path.getsize(en_file) / (1024*1024)  # MB
    hi_size = os.path.getsize(hi_file) / (1024*1024)  # MB
    
    print(f"English file size: {en_size:.1f} MB")
    print(f"Hindi file size: {hi_size:.1f} MB")
    
    # Show first few lines
    print("\nFirst 3 examples:")
    with open(en_file, 'r', encoding='utf-8') as f_en, \
         open(hi_file, 'r', encoding='utf-8') as f_hi:
        for i in range(3):
            en_line = f_en.readline().strip()
            hi_line = f_hi.readline().strip()
            print(f"{i+1}. EN: {en_line}")
            print(f"   HI: {hi_line}")
            print()
            
else:
    print("✗ Files missing!")
    print(f"EN file exists: {os.path.exists(en_file)}")
    print(f"HI file exists: {os.path.exists(hi_file)}")