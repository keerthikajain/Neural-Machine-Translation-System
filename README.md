# Neural Machine Translation System

A complete English-to-Hindi Neural Machine Translation system built from scratch using Transformer architecture with PyTorch.

## 🎯 Project Overview

This project demonstrates the **complete AI/ML pipeline** for building a production-ready translation system:

- **Dataset Processing**: IIT Bombay English-Hindi Parallel Corpus (1.4M pairs)
- **Tokenization**: SentencePiece BPE with 32K vocabulary
- **Architecture**: Transformer encoder-decoder (93.3M parameters)  
- **Training Pipeline**: Complete training framework with verification
- **Evaluation**: BLEU scoring and translation quality assessment
- **Documentation**: Comprehensive learning journey documentation

## ✅ Current Status

**🎉 COMPLETED: Training Pipeline Ready**

- ✅ **Dataset Preparation**: 1,412,400 clean English-Hindi sentence pairs
- ✅ **Tokenization**: SentencePiece trained and verified (32K vocabulary)
- ✅ **Transformer Architecture**: Complete encoder-decoder implementation
- ✅ **Training Pipeline**: AdamW optimizer, warmup scheduling, checkpointing
- ✅ **Verification**: 28/28 comprehensive tests passed
- ✅ **Documentation**: Complete learning journey documented

**🚀 READY FOR**: Model training execution

## 🏗️ Architecture

**Transformer Model**:
- **Size**: 93.3M parameters (~356MB)
- **Architecture**: 6-layer encoder + 6-layer decoder
- **Attention**: 8-head multi-head attention  
- **Embedding**: 512-dimensional with positional encoding
- **Vocabulary**: Shared 32K SentencePiece (English + Hindi)

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
PyTorch 2.0+
8GB+ RAM (for training)
```

### Setup
```bash
# Clone repository
git clone https://github.com/keerthikajain/Neural-Machine-Translation-System.git
cd Neural-Machine-Translation-System

# Install dependencies
pip install -r requirements.txt

# Download and prepare dataset (automatic)
python -c "from src.data.dataset import create_validation_files; create_validation_files()"
```

### Training
```bash
# Start training with student-friendly defaults
python train_model.py
```

### Verification
```bash
# Run comprehensive verification tests
python notebooks/training_pipeline_verification.py

# Test architecture components
python notebooks/comprehensive_verification.py
```

## 📊 Dataset Information

- **Source**: IIT Bombay English-Hindi Parallel Corpus
- **Training**: 1,412,400 sentence pairs
- **Validation**: 520 sentence pairs
- **Test**: 2,507 sentence pairs
- **Languages**: English → Hindi
- **Quality**: Cleaned, deduplicated, no leakage

## 🔬 Verification Results

**Training Pipeline**: ✅ **28/28 tests passed**
- Dataset loading and tokenization ✅
- Model architecture and attention ✅  
- Training loop and optimization ✅
- Loss function and gradient flow ✅
- Checkpointing and evaluation ✅

**Architecture Verification**: ✅ **All components verified**
- Cross-attention handles different sequence lengths
- Gradient flow through all 93.3M parameters
- Real tokenizer integration working

## 📚 Documentation

- **[Project Learning Log](docs/PROJECT_LEARNING_LOG.md)**: Complete development journey
- **[Transformer Verification](docs/TRANSFORMER_VERIFICATION.md)**: Architecture verification report  
- **[Training Pipeline Verification](docs/TRAINING_PIPELINE_VERIFICATION.md)**: Pre-training safety audit

## 🛠️ Technology Stack

- **Deep Learning**: PyTorch, Custom Transformer implementation
- **NLP**: SentencePiece tokenization, BLEU evaluation
- **Data Processing**: Custom dataset pipeline, Unicode normalization  
- **Training**: AdamW optimizer, warmup scheduling, gradient clipping
- **Verification**: Comprehensive testing framework
- **Documentation**: Markdown with detailed implementation notes

## 📁 Project Structure

```
src/
├── model/                  # Transformer architecture
│   ├── transformer.py     # Complete model implementation
│   ├── attention.py       # Multi-head attention + FFN
│   ├── encoder.py         # Transformer encoder  
│   ├── decoder.py         # Transformer decoder
│   └── embeddings.py      # Token + positional embeddings
├── training/               # Training framework
│   ├── trainer.py         # Training utilities and loops
│   └── train_nmt.py       # Main training script
├── data/                   # Dataset processing
│   └── dataset.py         # PyTorch dataset and data loaders
└── evaluation/             # Evaluation and inference
    └── evaluator.py        # Translation and BLEU scoring

notebooks/                  # Verification and testing
├── comprehensive_verification.py      # Architecture testing
├── training_pipeline_verification.py  # Pre-training audit
└── test_training_pipeline.py         # Integration testing

docs/                       # Complete documentation
├── PROJECT_LEARNING_LOG.md           # Learning journey
├── TRANSFORMER_VERIFICATION.md        # Architecture verification  
└── TRAINING_PIPELINE_VERIFICATION.md  # Training safety audit

train_model.py             # Simple training launcher
```

## 🎓 Learning Outcomes

This project demonstrates:

**Deep Learning Engineering**:
- Transformer architecture implementation from scratch
- Multi-head attention and encoder-decoder mechanisms
- Custom training loops with proper optimization
- Gradient flow verification and debugging

**ML Pipeline Engineering**:
- Large dataset processing and cleaning
- Tokenization and vocabulary management  
- Training pipeline design and verification
- Checkpointing and model persistence

**Software Engineering**:
- Modular code architecture and testing
- Comprehensive documentation and verification
- Git workflow and project management
- Production-ready code standards

## 🔍 Key Implementation Highlights

- **Cross-attention handling**: Supports different encoder/decoder sequence lengths
- **Proper masking**: Causal masking for decoder + padding mask semantics
- **Label smoothing**: Cross-entropy loss with smoothing for better generalization
- **Learning rate scheduling**: Transformer-style warmup + inverse square root decay
- **Memory efficiency**: Dynamic padding and batch processing
- **Comprehensive testing**: 28-test verification suite ensuring technical correctness

## 📈 Expected Training Performance

**On CPU**: ~2-3 hours per epoch (5 epochs recommended)  
**Memory usage**: ~1GB during training  
**Model size**: 356MB (manageable for student hardware)

---

**🏆 This project showcases a complete, production-ready neural machine translation pipeline with comprehensive verification and documentation - demonstrating advanced ML engineering skills.**