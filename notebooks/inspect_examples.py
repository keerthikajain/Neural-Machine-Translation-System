# Day 3: Inspect Dataset Examples
# Goal: See how English-Hindi pairs are actually stored

from datasets import load_dataset

print("Loading the IIT Bombay English-Hindi dataset...")
dataset = load_dataset("cfilt/iitb-english-hindi")

print("\nDataset structure:")
print(dataset)

print("\nLooking at the first 3 examples from training data:")
print("=" * 50)

for i in range(3):
    example = dataset['train'][i]
    print(f"\nExample {i+1}:")
    print(f"Raw data: {example}")
    print(f"Translation field: {example['translation']}")
    print("-" * 30)

print("\nNow let's see what's inside the 'translation' field:")
print("=" * 50)

# Look at the structure of the translation field
first_translation = dataset['train'][0]['translation']
print(f"Type of translation field: {type(first_translation)}")
print(f"Translation content: {first_translation}")

# If it's a dictionary, let's see the keys
if isinstance(first_translation, dict):
    print(f"Keys in translation: {list(first_translation.keys())}")