# Evaluation utilities for Neural Machine Translation
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sentencepiece as smp
from typing import List, Dict, Tuple, Optional
import json
import time
from collections import defaultdict

class NMTEvaluator:
    """
    Evaluator for Neural Machine Translation model
    Handles inference, BLEU score calculation, and translation quality assessment
    """
    
    def __init__(self, model: nn.Module, tokenizer_path: str, device: torch.device,
                 max_length: int = 256, beam_size: int = 1):
        """
        Args:
            model: Trained Transformer model
            tokenizer_path: Path to SentencePiece tokenizer
            device: Device for inference
            max_length: Maximum generation length
            beam_size: Beam search size (1 for greedy decoding)
        """
        self.model = model.to(device)
        self.device = device
        self.max_length = max_length
        self.beam_size = beam_size
        
        # Load tokenizer
        self.tokenizer = smp.SentencePieceProcessor()
        self.tokenizer.load(tokenizer_path)
        
        # Special tokens
        self.pad_id = self.tokenizer.pad_id()
        self.bos_id = self.tokenizer.bos_id()  
        self.eos_id = self.tokenizer.eos_id()
        self.unk_id = self.tokenizer.unk_id()
        
        print(f"Evaluator initialized:")
        print(f"  Device: {device}")
        print(f"  Vocabulary size: {self.tokenizer.vocab_size()}")
        print(f"  Max length: {max_length}")
        print(f"  Beam size: {beam_size}")
    
    def translate_sentence(self, source_text: str) -> str:
        """
        Translate a single sentence
        
        Args:
            source_text: English input sentence
            
        Returns:
            str: Hindi translation
        """
        self.model.eval()
        
        with torch.no_grad():
            # Tokenize source
            src_tokens = self.tokenizer.encode_as_ids(source_text)
            src_tokens = [self.bos_id] + src_tokens
            src_tensor = torch.tensor([src_tokens], dtype=torch.long).to(self.device)
            
            # Encode source
            encoder_output, src_mask = self.model.encode(src_tensor)
            
            # Generate translation (greedy decoding for simplicity)
            if self.beam_size == 1:
                translation = self._greedy_decode(encoder_output, src_mask)
            else:
                translation = self._beam_search(encoder_output, src_mask)
            
            # Decode tokens to text
            translation_text = self.tokenizer.decode_ids(translation)
            
            return translation_text
    
    def _greedy_decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor) -> List[int]:
        """
        Greedy decoding for translation generation
        
        Args:
            encoder_output: Encoder output [1, src_len, d_model]
            src_mask: Source mask [1, 1, 1, src_len]
            
        Returns:
            List[int]: Generated token IDs
        """
        batch_size = encoder_output.size(0)
        
        # Start with BOS token
        generated = torch.tensor([[self.bos_id]], dtype=torch.long).to(self.device)
        
        for _ in range(self.max_length):
            # Get next token logits
            logits = self.model.decode_step(generated, encoder_output, src_mask)
            
            # Get last token logits and predict next token
            next_token_logits = logits[0, -1, :]  # [vocab_size]
            next_token = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)  # [1, 1]
            
            # Stop if EOS generated
            if next_token.item() == self.eos_id:
                break
                
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
        
        # Return as list (excluding BOS)
        return generated[0, 1:].tolist()
    
    def _beam_search(self, encoder_output: torch.Tensor, src_mask: torch.Tensor) -> List[int]:
        """
        Beam search decoding (simplified implementation)
        
        Args:
            encoder_output: Encoder output [1, src_len, d_model]
            src_mask: Source mask [1, 1, 1, src_len]
            
        Returns:
            List[int]: Generated token IDs
        """
        # For now, fallback to greedy decode
        # Full beam search implementation is more complex
        return self._greedy_decode(encoder_output, src_mask)
    
    def evaluate_dataset(self, data_loader: DataLoader, max_samples: Optional[int] = None) -> Dict:
        """
        Evaluate model on a dataset
        
        Args:
            data_loader: DataLoader with source-target pairs
            max_samples: Limit evaluation to first N samples
            
        Returns:
            Dict: Evaluation results
        """
        self.model.eval()
        
        translations = []
        references = []
        source_sentences = []
        
        total_samples = 0
        start_time = time.time()
        
        print("Starting evaluation...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                # Get source and reference texts
                batch_sources = batch['english_texts']
                batch_references = batch['hindi_texts']
                
                for src_text, ref_text in zip(batch_sources, batch_references):
                    if max_samples and total_samples >= max_samples:
                        break
                    
                    # Translate
                    translation = self.translate_sentence(src_text)
                    
                    # Store results
                    source_sentences.append(src_text)
                    translations.append(translation)
                    references.append(ref_text)
                    
                    total_samples += 1
                    
                    # Progress report
                    if total_samples % 100 == 0:
                        elapsed = time.time() - start_time
                        print(f"  Translated {total_samples} sentences in {elapsed:.1f}s")
                
                if max_samples and total_samples >= max_samples:
                    break
        
        # Calculate metrics
        print("Calculating metrics...")
        
        # Simple BLEU calculation (character-level for simplicity)
        bleu_scores = []
        for trans, ref in zip(translations, references):
            bleu = self._calculate_simple_bleu(trans, ref)
            bleu_scores.append(bleu)
        
        avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
        
        # Translation length statistics
        trans_lengths = [len(t.split()) for t in translations]
        ref_lengths = [len(r.split()) for r in references]
        
        results = {
            'num_samples': total_samples,
            'avg_bleu_score': avg_bleu,
            'avg_translation_length': sum(trans_lengths) / len(trans_lengths) if trans_lengths else 0,
            'avg_reference_length': sum(ref_lengths) / len(ref_lengths) if ref_lengths else 0,
            'evaluation_time': time.time() - start_time,
            'samples': [
                {
                    'source': src,
                    'translation': trans,
                    'reference': ref,
                    'bleu': bleu
                }
                for src, trans, ref, bleu in zip(
                    source_sentences[:10],  # First 10 examples
                    translations[:10],
                    references[:10],
                    bleu_scores[:10]
                )
            ]
        }
        
        return results
    
    def _calculate_simple_bleu(self, translation: str, reference: str) -> float:
        """
        Simple BLEU score calculation (character-level n-grams)
        
        Args:
            translation: Generated translation
            reference: Reference translation
            
        Returns:
            float: Approximate BLEU score
        """
        # Tokenize into characters for multilingual compatibility
        trans_chars = list(translation.replace(' ', ''))
        ref_chars = list(reference.replace(' ', ''))
        
        if len(trans_chars) == 0 or len(ref_chars) == 0:
            return 0.0
        
        # Calculate precision for different n-gram sizes
        precisions = []
        
        for n in range(1, 5):  # 1-gram to 4-gram
            if len(trans_chars) < n or len(ref_chars) < n:
                precisions.append(0.0)
                continue
            
            # Generate n-grams
            trans_ngrams = [''.join(trans_chars[i:i+n]) for i in range(len(trans_chars)-n+1)]
            ref_ngrams = [''.join(ref_chars[i:i+n]) for i in range(len(ref_chars)-n+1)]
            
            # Count matches
            trans_counts = defaultdict(int)
            ref_counts = defaultdict(int)
            
            for ngram in trans_ngrams:
                trans_counts[ngram] += 1
            for ngram in ref_ngrams:
                ref_counts[ngram] += 1
            
            # Calculate precision
            matches = 0
            for ngram, count in trans_counts.items():
                matches += min(count, ref_counts[ngram])
            
            precision = matches / len(trans_ngrams) if trans_ngrams else 0.0
            precisions.append(precision)
        
        # Geometric mean of precisions
        if all(p > 0 for p in precisions):
            bleu = (precisions[0] * precisions[1] * precisions[2] * precisions[3]) ** 0.25
        else:
            bleu = 0.0
        
        # Brevity penalty
        bp = min(1.0, len(trans_chars) / len(ref_chars)) if ref_chars else 0.0
        
        return bleu * bp
    
    def interactive_translate(self):
        """
        Interactive translation for manual testing
        """
        print("Interactive Translation Mode")
        print("Enter English sentences to translate (or 'quit' to exit):")
        print("-" * 50)
        
        while True:
            try:
                source = input("\nEnglish: ").strip()
                
                if source.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not source:
                    continue
                
                start_time = time.time()
                translation = self.translate_sentence(source)
                elapsed = time.time() - start_time
                
                print(f"Hindi: {translation}")
                print(f"Time: {elapsed:.2f}s")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Translation error: {e}")

def load_model_for_evaluation(checkpoint_path: str, vocab_size: int = 32000, 
                            d_model: int = 512, n_heads: int = 8, n_layers: int = 6) -> nn.Module:
    """
    Load trained model from checkpoint for evaluation
    
    Args:
        checkpoint_path: Path to model checkpoint
        vocab_size: Vocabulary size
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of layers
        
    Returns:
        nn.Module: Loaded model
    """
    # Import here to avoid circular imports
    from src.model import create_transformer_model
    
    # Create model
    model = create_transformer_model(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Training epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Training step: {checkpoint.get('step', 'unknown')}")
    
    return model

if __name__ == "__main__":
    # Test evaluation setup
    print("Testing evaluator setup...")
    
    # This would normally load a trained model
    from src.model import create_transformer_model
    
    device = torch.device('cpu')
    model = create_transformer_model(vocab_size=32000, d_model=128, n_heads=4, n_layers=2)
    
    evaluator = NMTEvaluator(
        model=model,
        tokenizer_path='data/processed/tokenizer/sp_model.model',
        device=device,
        max_length=50
    )
    
    # Test translation
    test_sentence = "Hello world"
    translation = evaluator.translate_sentence(test_sentence)
    
    print(f"Test translation:")
    print(f"  English: {test_sentence}")
    print(f"  Hindi: {translation}")
    
    print("Evaluator setup successful!")