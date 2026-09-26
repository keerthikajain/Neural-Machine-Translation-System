# Check for duplicate translation pairs
from datasets import load_dataset

dataset = load_dataset("cfilt/iitb-english-hindi")

print("Checking for duplicate translation pairs...")

seen_pairs = set()
duplicates = []
total_checked = 0

# Check first 10,000 examples
sample_size = min(10000, len(dataset['train']))

for i in range(sample_size):
    example = dataset['train'][i]['translation']
    
    # Create a pair tuple (English, Hindi)
    pair = (example['en'], example['hi'])
    
    if pair in seen_pairs:
        duplicates.append((i, pair))
    else:
        seen_pairs.add(pair)
    
    total_checked += 1

print(f"\nResults from {total_checked:,} examples:")
print(f"Unique pairs: {len(seen_pairs):,}")
print(f"Duplicate pairs found: {len(duplicates)}")

if duplicates:
    print(f"\nFirst few duplicates:")
    for i, (idx, pair) in enumerate(duplicates[:3]):
        print(f"{i+1}. Index {idx}: EN='{pair[0]}' HI='{pair[1]}'")
else:
    print("\nNo duplicates found in sample!")

# Calculate duplicate percentage
if total_checked > 0:
    unique_rate = len(seen_pairs) / total_checked * 100
    print(f"\nUnique rate: {unique_rate:.1f}%")