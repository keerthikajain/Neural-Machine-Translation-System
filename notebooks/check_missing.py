# Check for missing or empty translations
from datasets import load_dataset

dataset = load_dataset("cfilt/iitb-english-hindi")

print("Checking for missing/empty translations in training data...")

missing_english = 0
missing_hindi = 0  
empty_english = 0
empty_hindi = 0
total_checked = 0

# Check first 10,000 examples (to avoid long processing)
sample_size = min(10000, len(dataset['train']))

for i in range(sample_size):
    example = dataset['train'][i]['translation']
    
    # Check if keys exist
    if 'en' not in example:
        missing_english += 1
    elif not example['en'] or example['en'].strip() == '':
        empty_english += 1
        
    if 'hi' not in example:
        missing_hindi += 1
    elif not example['hi'] or example['hi'].strip() == '':
        empty_hindi += 1
    
    total_checked += 1

print(f"\nResults from {total_checked:,} examples:")
print(f"Missing English: {missing_english}")
print(f"Empty English: {empty_english}")
print(f"Missing Hindi: {missing_hindi}")
print(f"Empty Hindi: {empty_hindi}")

# Show examples of any problems found
if empty_english > 0 or empty_hindi > 0:
    print("\nFirst few problematic examples:")
    count = 0
    for i in range(sample_size):
        if count >= 3:
            break
        example = dataset['train'][i]['translation']
        if (example.get('en', '').strip() == '' or 
            example.get('hi', '').strip() == ''):
            print(f"Example {i}: {example}")
            count += 1