"""
Day 2: IIT Bombay Dataset Exploration
Understanding what a parallel corpus looks like
"""

# For now, let's create some example data to understand the concept
# We'll replace this with real dataset loading later

def demonstrate_parallel_corpus():
    """
    Show what a parallel corpus looks like with examples
    """
    
    # Example English-Hindi parallel sentences
    # These are the type of sentence pairs in the IIT Bombay corpus
    sample_parallel_data = [
        {
            "english": "I am going home.",
            "hindi": "मैं घर जा रहा हूँ।",
            "explanation": "Simple present continuous sentence"
        },
        {
            "english": "The weather is nice today.",
            "hindi": "आज मौसम अच्छा है।",
            "explanation": "Common weather description"
        },
        {
            "english": "What is your name?",
            "hindi": "आपका नाम क्या है?",
            "explanation": "Basic question format"
        },
        {
            "english": "I love learning new languages.",
            "hindi": "मुझे नई भाषाएँ सीखना पसंद है।",
            "explanation": "Expression of preference"
        },
        {
            "english": "The book is on the table.",
            "hindi": "किताब मेज़ पर है।",
            "explanation": "Spatial relationship"
        }
    ]
    
    print("=" * 60)
    print("WHAT IS A PARALLEL CORPUS?")
    print("=" * 60)
    print("A parallel corpus contains sentence pairs where:")
    print("- Each English sentence has its Hindi translation")
    print("- The AI model learns from these examples")
    print("- Pattern: English → Hindi")
    print()
    
    print("SAMPLE PARALLEL SENTENCE PAIRS:")
    print("-" * 60)
    
    for i, pair in enumerate(sample_parallel_data, 1):
        print(f"{i}. English: {pair['english']}")
        print(f"   Hindi:   {pair['hindi']}")
        print(f"   Note:    {pair['explanation']}")
        print()
    
    print("=" * 60)
    print("WHAT THE AI MODEL LEARNS:")
    print("=" * 60)
    print("From these examples, the model discovers:")
    print("• 'I am' → 'मैं' (subject patterns)")
    print("• 'going' → 'जा रहा' (verb patterns)")  
    print("• 'home' → 'घर' (object patterns)")
    print("• Word order differences between languages")
    print("• Grammar structures and rules")
    print()
    
    return sample_parallel_data

def explain_iitb_dataset():
    """
    Explain the IIT Bombay dataset specifications
    """
    
    print("=" * 60)
    print("IIT BOMBAY ENGLISH-HINDI PARALLEL CORPUS")
    print("=" * 60)
    print("📊 DATASET STATISTICS:")
    print("• Total sentence pairs: 1.49 million")
    print("• Source: IIT Bombay Language Technology Center")
    print("• License: Creative Commons (free for education)")
    print("• Quality: Research-grade, manually verified")
    print()
    
    print("🎯 WHY THIS DATASET IS PERFECT FOR US:")
    print("• Large enough: 1.49M examples for good AI training")
    print("• High quality: Created by language researchers")
    print("• Academic use: Free for student projects")
    print("• Accessible: Available through Hugging Face")
    print("• Hindi focus: Matches our project requirements")
    print()
    
    print("📁 WHAT WE'LL GET:")
    print("• English sentences (input)")
    print("• Corresponding Hindi sentences (target)")
    print("• Pre-split train/validation/test sets")
    print("• Clean, preprocessed text")
    print()

def next_steps():
    """
    Explain what we'll do next with the dataset
    """
    
    print("=" * 60)
    print("WHAT WE'LL DO WITH THIS DATASET:")
    print("=" * 60)
    print("1. LOAD: Download the dataset using Hugging Face")
    print("2. INSPECT: Look at real English-Hindi sentence pairs")  
    print("3. UNDERSTAND: Check data quality and structure")
    print("4. CLEAN: Remove empty or problematic sentences")
    print("5. SPLIT: Organize into train/validation/test sets")
    print("6. TOKENIZE: Convert sentences to tokens (Week 1 final step)")
    print()
    
    print("🚫 WHAT WE'RE NOT DOING YET:")
    print("• Building the Transformer model (Week 2)")
    print("• Training the AI (Week 2)")
    print("• Creating the web app (Week 4)")
    print()

if __name__ == "__main__":
    print("🎓 NEURAL MACHINE TRANSLATION - DAY 2")
    print("Understanding Parallel Corpus and IIT Bombay Dataset")
    print()
    
    # Show what a parallel corpus looks like
    sample_data = demonstrate_parallel_corpus()
    
    # Explain our specific dataset
    explain_iitb_dataset()
    
    # Show next steps
    next_steps()
    
    print("=" * 60)
    print("✅ DAY 2 LEARNING CHECKPOINT")
    print("=" * 60)
    print("You should now understand:")
    print("• What a parallel corpus is")
    print("• Why IIT Bombay dataset is suitable")
    print("• How English-Hindi pairs help AI learn translation")
    print("• What we'll do with the data this week")
    print("=" * 60)