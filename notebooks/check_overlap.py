# Check overlap between cleaned training and validation/test sets
from datasets import load_dataset

print("Loading original dataset...")
dataset = load_dataset("cfilt/iitb-english-hindi")

print("Loading cleaned training data...")
# Load cleaned training pairs
train_pairs = set()
with open('data/processed/clean_train_en.txt', 'r', encoding='utf-8') as f_en, \
     open('data/processed/clean_train_hi.txt', 'r', encoding='utf-8') as f_hi:
    for en_line, hi_line in zip(f_en, f_hi):
        pair = (en_line.strip(), hi_line.strip())
        train_pairs.add(pair)

print(f"Cleaned training pairs loaded: {len(train_pairs):,}")

print("\nChecking validation set overlap...")
val_overlaps = []
for i, example in enumerate(dataset['validation']):
    pair = (example['translation']['en'], example['translation']['hi'])
    if pair in train_pairs:
        val_overlaps.append((i, pair))

print(f"Validation examples: {len(dataset['validation'])}")
print(f"Overlaps with training: {len(val_overlaps)}")

print("\nChecking test set overlap...")
test_overlaps = []
for i, example in enumerate(dataset['test']):
    pair = (example['translation']['en'], example['translation']['hi'])
    if pair in train_pairs:
        test_overlaps.append((i, pair))

print(f"Test examples: {len(dataset['test'])}")
print(f"Overlaps with training: {len(test_overlaps)}")

# Show examples if any overlaps found
if val_overlaps:
    print(f"\nFirst few validation overlaps:")
    for i, (idx, pair) in enumerate(val_overlaps[:3]):
        print(f"{i+1}. EN: {pair[0]}")
        print(f"   HI: {pair[1]}")

if test_overlaps:
    print(f"\nFirst few test overlaps:")
    for i, (idx, pair) in enumerate(test_overlaps[:3]):
        print(f"{i+1}. EN: {pair[0]}")
        print(f"   HI: {pair[1]}")

print(f"\nSUMMARY:")
print(f"✓ Clean training: {len(train_pairs):,} pairs")
print(f"{'✓' if len(val_overlaps) == 0 else '✗'} Validation overlap: {len(val_overlaps)}/{len(dataset['validation'])}")
print(f"{'✓' if len(test_overlaps) == 0 else '✗'} Test overlap: {len(test_overlaps)}/{len(dataset['test'])}")