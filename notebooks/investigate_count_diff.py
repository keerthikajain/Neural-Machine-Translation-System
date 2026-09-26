# Investigate the counting discrepancy between the two scripts

print("=== INVESTIGATING COUNT DISCREPANCY ===")
print()

# Method 1: Same as verify_clean_data.py (line counting)
print("METHOD 1 - Line counting (like verify_clean_data.py):")
with open('data/processed/clean_train_en.txt', 'r', encoding='utf-8') as f:
    en_lines = sum(1 for line in f)

with open('data/processed/clean_train_hi.txt', 'r', encoding='utf-8') as f:
    hi_lines = sum(1 for line in f)

print(f"English lines: {en_lines:,}")
print(f"Hindi lines: {hi_lines:,}")

# Method 2: Same as check_overlap.py (set building with zip)
print("\nMETHOD 2 - Set building (like check_overlap.py):")
train_pairs = set()
pair_count = 0
with open('data/processed/clean_train_en.txt', 'r', encoding='utf-8') as f_en, \
     open('data/processed/clean_train_hi.txt', 'r', encoding='utf-8') as f_hi:
    for en_line, hi_line in zip(f_en, f_hi):
        pair = (en_line.strip(), hi_line.strip())
        train_pairs.add(pair)
        pair_count += 1

print(f"Pairs processed: {pair_count:,}")
print(f"Unique pairs in set: {len(train_pairs):,}")
print(f"Duplicates found during set building: {pair_count - len(train_pairs):,}")

print(f"\nDISCREPANCY ANALYSIS:")
print(f"Line count method: {en_lines:,}")
print(f"Set building method: {len(train_pairs):,}")
print(f"Difference: {en_lines - len(train_pairs):,}")

# Check if there are empty lines
print(f"\nCHECKING FOR EMPTY LINES:")
empty_en_lines = 0
empty_hi_lines = 0
with open('data/processed/clean_train_en.txt', 'r', encoding='utf-8') as f_en, \
     open('data/processed/clean_train_hi.txt', 'r', encoding='utf-8') as f_hi:
    for line_num, (en_line, hi_line) in enumerate(zip(f_en, f_hi), 1):
        if en_line.strip() == '':
            empty_en_lines += 1
            if empty_en_lines <= 3:
                print(f"Empty EN line at {line_num}")
        if hi_line.strip() == '':
            empty_hi_lines += 1
            if empty_hi_lines <= 3:
                print(f"Empty HI line at {line_num}")

print(f"Empty English lines: {empty_en_lines}")
print(f"Empty Hindi lines: {empty_hi_lines}")

print(f"\nCONCLUSION:")
if empty_en_lines + empty_hi_lines > 0:
    print("Empty lines found - these get filtered out during set building")
elif pair_count - len(train_pairs) > 0:
    print("Duplicate pairs found during set building - were not fully cleaned")
else:
    print("Unknown cause - need deeper investigation")