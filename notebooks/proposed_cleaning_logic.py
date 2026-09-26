# PROPOSED CLEANING LOGIC - DO NOT RUN YET
# This shows the exact logic to be implemented

import re

def normalize_text(text):
    """
    Normalize text for consistent deduplication
    1. Strip leading/trailing whitespace
    2. Collapse multiple whitespace into single spaces
    3. Preserve capitalization
    """
    if not text:
        return ""
    
    # Strip leading/trailing whitespace
    normalized = text.strip()
    
    # Collapse multiple whitespace (spaces, tabs, newlines) into single spaces
    normalized = re.sub(r'\s+', ' ', normalized)
    
    return normalized

def is_valid_translation(en_text, hi_text):
    """
    Check if translation pair is valid (not empty after normalization)
    """
    normalized_en = normalize_text(en_text)
    normalized_hi = normalize_text(hi_text)
    
    # Both must be non-empty after normalization
    return len(normalized_en) > 0 and len(normalized_hi) > 0

# PROPOSED CLEANING PIPELINE:
print("=== PROPOSED CLEANING PIPELINE ===")
print()

print("STEP 1: Load dataset")
print("dataset = load_dataset('cfilt/iitb-english-hindi')")
print()

print("STEP 2: Process each example with validation")
print("for example in dataset['train']:")
print("    en_raw = example['translation']['en']")
print("    hi_raw = example['translation']['hi']")
print()
print("    # Check if valid (not empty after normalization)")
print("    if not is_valid_translation(en_raw, hi_raw):")
print("        empty_pairs_skipped += 1")
print("        continue")
print()
print("    # Normalize for deduplication key")
print("    en_normalized = normalize_text(en_raw)")
print("    hi_normalized = normalize_text(hi_raw)")
print("    pair_key = (en_normalized, hi_normalized)")
print()
print("    # Check for duplicates using normalized pair")
print("    if pair_key not in seen_pairs:")
print("        seen_pairs.add(pair_key)")
print("        # Save ORIGINAL text (not normalized) to preserve formatting")
print("        clean_examples.append({")
print("            'en': en_raw,  # Original text")
print("            'hi': hi_raw   # Original text")  
print("        })")
print("    else:")
print("        duplicates_removed += 1")
print()

print("STEP 3: Save with same normalization applied")
print("# Apply normalization when saving to ensure consistency")
print("for example in clean_examples:")
print("    en_clean = normalize_text(example['en'])")
print("    hi_clean = normalize_text(example['hi'])")
print("    f_en.write(en_clean + '\\n')")
print("    f_hi.write(hi_clean + '\\n')")
print()

print("=== EXAMPLE NORMALIZATION ===")
print()

# Show examples of what normalization does
test_cases = [
    ("  Hello world  ", "नमस्ते दुनिया"),
    ("Multiple    spaces   here", "कई     स्थान    यहाँ"),
    ("Tab\tseparated\ttext", "टैब\tअलग\tपाठ"),
    ("Line\nbreaks\ninside", "लाइन\nब्रेक\nअंदर"),
    ("", "वैध नहीं"),  # Empty English
    ("Valid text", ""),  # Empty Hindi
    ("  ", "   "),  # Whitespace only
]

print("INPUT -> NORMALIZED:")
for en, hi in test_cases:
    en_norm = normalize_text(en)
    hi_norm = normalize_text(hi)
    valid = is_valid_translation(en, hi)
    
    print(f"EN: '{en}' -> '{en_norm}' (valid: {valid})")
    print(f"HI: '{hi}' -> '{hi_norm}'")
    print()

print("=== KEY BENEFITS ===")
print("✓ Removes empty/whitespace-only pairs")
print("✓ Consistent whitespace handling")
print("✓ True deduplication (same normalization everywhere)")
print("✓ Preserves original capitalization")
print("✓ Clean saved files from the start")
print("✓ Same logic used in verification scripts")