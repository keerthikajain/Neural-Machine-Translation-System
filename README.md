# Neural Machine Translation System

An English-to-Hindi Neural Machine Translation system built using Transformer architecture for educational purposes.

## Project Overview

This project demonstrates the complete AI/ML pipeline for building a translation system:
- Dataset preparation using IIT Bombay English-Hindi Parallel Corpus
- Tokenization with SentencePiece
- Transformer-based model development
- Model training and evaluation
- Simple deployment with FastAPI and React

## Current Status

🔄 **Week 1**: Project setup and data preparation
- [x] Project structure setup
- [ ] Dataset acquisition and exploration
- [ ] Data cleaning and preprocessing
- [ ] Tokenization setup

## Technology Stack

- **AI/ML**: PyTorch, Hugging Face Transformers
- **Backend**: FastAPI, Uvicorn
- **Frontend**: React, JavaScript, Tailwind CSS
- **Data**: Pandas, NumPy
- **Evaluation**: sacreBLEU, chrF
- **Tools**: Git, VS Code, Jupyter

## Project Structure

```
neural-machine-translation/
├── data/                    # Dataset storage
│   ├── raw/                # Original dataset files
│   └── processed/          # Cleaned and preprocessed data
├── notebooks/              # Jupyter notebooks for exploration
├── src/                    # Source code
│   ├── data/              # Data processing modules
│   ├── tokenization/      # Tokenization utilities
│   ├── model/             # Model architecture
│   ├── training/          # Training scripts
│   └── evaluation/        # Evaluation utilities
├── app/                   # Application deployment
│   ├── backend/           # FastAPI backend
│   └── frontend/          # React frontend
├── tests/                 # Unit tests
└── docs/                  # Documentation
```

## Getting Started

1. Set up virtual environment
2. Install dependencies
3. Download IIT Bombay dataset
4. Explore the data

## Learning Objectives

By building this project, you will learn:
- How Neural Machine Translation works
- Transformer architecture and attention mechanisms
- Data preprocessing for NLP tasks
- Model training and evaluation techniques
- Deployment of ML models

---

**Note**: This is a student project focused on learning. The goal is understanding each component rather than building a production-ready system.