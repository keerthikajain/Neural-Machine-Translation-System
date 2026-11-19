# Saarthi NMT - Credit-Efficient Completion Plan

## Current Project State (as of November 2026)

### Completed (CPU, $0 cost)
- Dataset acquisition and cleaning (1.4M pairs)
- SentencePiece tokenizer (32K vocab)
- Transformer architecture (Compact A: 256-dim/4-layer + Full: 512-dim/6-layer)
- Training pipeline with checkpointing and resume support
- Evaluation framework (BLEU, greedy decode)
- Diagnostic experiments (3 tiny overfit runs, 100 samples each)

### Current Blocker
- Model achieves 100% EOS-position accuracy but 0% non-EOS accuracy
- Model only predicts EOS tokens regardless of input
- Root cause not yet identified (no gradient-flow, label-smoothing, or target-placement bug found)
- Learning-rate experiment (3 configs on 100 samples) is ready to run on CPU

### Remaining Work
1. Diagnose and fix the EOS-only prediction problem
2. Train on full 1.4M dataset (requires GPU)
3. Evaluate with BLEU/chrF on test set
4. Implement inference system with beam search
5. Build FastAPI backend
6. Build React frontend
7. Deploy and host

---

## Compute Resource Assessment

### Local CPU (Current)
- **Hardware**: Intel i7, 4 physical cores, Windows 11
- **GPU**: None (CUDA not available)
- **Speed**: ~2.0 samples/sec (Compact A, batch 16, max_len 256)
- **Full epoch estimate**: 199+ hours per epoch on full dataset
- **Suitable for**: Diagnostics, unit tests, small experiments (100-200 samples), code development

### GPU Options (Ranked by Credit Efficiency)

| Option | GPU | $/hour | Speed (est.) | Epoch time | 5-epoch cost | Notes |
|--------|-----|--------|-------------|------------|-------------|-------|
| Kaggle (free) | T4 x2 | $0 | ~50 samples/sec | ~7.8 hrs | $0 | 30 hrs/week limit, may need 2 weeks |
| Google Colab Pro | T4/P100 | ~$10/mo | ~50 samples/sec | ~7.8 hrs | $10/mo | 12+ hrs/day GPU, reliable |
| Google Colab Pro+ | A100 | ~$50/mo | ~200 samples/sec | ~2 hrs | $50/mo | Best throughput per dollar |
| Lambda Labs | A100 40GB | $1.50/hr | ~200 samples/sec | ~2 hrs | ~$15 | No time limits, dedicated |
| AWS g4dn.xlarge (spot) | T4 | ~$0.75/hr | ~50 samples/sec | ~7.8 hrs | ~$30 | May be preempted |
| AWS g4dn.xlarge (on-demand) | T4 | ~$0.74/hr | ~50 samples/sec | ~7.8 hrs | ~$30 | Stable, no preemption |
| Vast.ai (community) | RTX 3090 | ~$0.30/hr | ~100 samples/sec | ~3.9 hrs | ~$6 | Cheapest, less reliable |

### Recommendation
1. **Phase 1 (Diagnostic)**: Local CPU - $0
2. **Phase 2 (Full Training)**: Kaggle free GPU or Lambda Labs A100
3. **Phase 3 (Evaluation/Deployment)**: Local CPU for inference, minimal cloud for hosting

---

## Credit-Efficient Execution Plan

### Phase 1: Fix EOS-Only Prediction (CPU, $0, ~1 hour)

**Goal**: Confirm model can learn non-EOS Hindi tokens on 100 samples.

**Steps**:
1. Run the 3-config learning-rate experiment on CPU (~15 min)
   ```
   python experiments/lr_experiment.py
   ```
2. Analyze results:
   - Does any LR config produce non-zero non-EOS accuracy?
   - Does loss decrease meaningfully?
   - Is first-token accuracy above 0%?
3. If LR experiment does not fix the issue:
   - Try overfitting on 10 samples with very high LR (0.01) for 50 epochs
   - Inspect gradient norms per layer
   - Check if decoder cross-attention is actually attending to encoder output
4. **Gate**: Do not proceed to GPU training until non-EOS accuracy > 0% on tiny dataset

**Credit cost**: $0
**Time**: ~1 hour on CPU

### Phase 2: Full Training on GPU (~$6-$30 depending on choice)

**Prerequisites**: Phase 1 must show the model can learn non-EOS tokens.

**Goal**: Train the model on full 1.4M dataset to produce real Hindi translations.

**Architecture decision**:
- Start with Compact A (256-dim, 4 layers) - faster training, smaller checkpoints
- Only upgrade to full 512-dim model if Compact A shows meaningful learning and time/budget allows

**Training configuration**:
- Batch size: 32-64 (GPU allows larger batches)
- Max length: 128 (sufficient for 99% of sentences based on analysis)
- Mixed precision: Enabled (fp16) - ~2x speedup, ~50% memory reduction
- Checkpointing: Every 1000 steps
- Resume support: Load from latest checkpoint if interrupted

**Execution plan**:
1. Upload dataset and code to GPU instance
2. Run training with monitoring:
   ```
   python src/training/train_nmt.py --batch_size 32 --max_length 128 --num_epochs 5
   ```
3. Monitor validation loss every epoch
4. **Stopping criteria**:
   - Stop if validation loss has not improved for 2 consecutive epochs
   - Stop if non-EOS token accuracy does not improve for 3 epochs
   - Stop if estimated total time exceeds 2x the initial estimate

**Estimated cost by platform**:

| Platform | Est. time | Est. cost | Risk |
|----------|-----------|-----------|------|
| Kaggle free T4 | ~40 hrs (5+ epochs) | $0 | May hit weekly limit, need 2 sessions |
| Lambda A100 | ~10 hrs | ~$15 | Low risk, dedicated |
| Colab Pro T4 | ~40 hrs over 5 days | ~$10 (1 month) | Session disconnects possible |
| Vast.ai RTX 3090 | ~20 hrs | ~$6 | Instance may disappear |

**Credit cost**: $0-$15
**Time**: 10-40 hours depending on GPU choice

### Phase 3: Evaluation (CPU, $0, ~2 hours)

**Goal**: Measure translation quality on the 2,507 test samples.

**Steps**:
1. Load best checkpoint from Phase 2
2. Run greedy decoding on test set
3. Calculate BLEU score and chrF score
4. Generate translation samples for manual inspection
5. Save evaluation results

**Implementation**:
- Use existing `src/evaluation/evaluator.py`
- Run on CPU (inference is much faster than training)
- Save results to `experiments/results/evaluation/`

**Credit cost**: $0
**Time**: ~2 hours on CPU

### Phase 4: Inference System (CPU, $0, ~3 hours)

**Goal**: Build a translation inference module with beam search.

**Steps**:
1. Implement beam search decoding (beam_size=4)
2. Add length normalization penalty
3. Implement batch translation for efficiency
4. Test on sample sentences
5. Save inference module to `src/inference/`

**Credit cost**: $0
**Time**: ~3 hours on CPU

### Phase 5: FastAPI Backend (CPU, $0, ~4 hours)

**Goal**: Build REST API for translation.

**Steps**:
1. Create FastAPI app with /translate endpoint
2. Load trained model checkpoint
3. Accept English text, return Hindi translation
4. Add request validation and error handling
5. Test with curl/Postman
6. Save to `app/backend/`

**Credit cost**: $0
**Time**: ~4 hours on CPU

### Phase 6: React Frontend (CPU, $0, ~8 hours)

**Goal**: Build user interface for translation.

**Steps**:
1. Create React app with translation interface
2. Connect to FastAPI backend
3. Implement features per feature_breakdown.md:
   - Translation input/output
   - Quick phrases
   - Translation history (local storage)
   - Settings
4. Test end-to-end
5. Save to `app/frontend/`

**Credit cost**: $0
**Time**: ~8 hours on CPU

### Phase 7: Deployment and Hosting (~$0-$5/month)

**Goal**: Deploy the working system.

**Options**:
- **Backend hosting**: Render free tier, Railway ($5/mo), or local deployment
- **Frontend hosting**: Vercel free tier, Netlify free tier
- **Model file**: ~366 MB (Compact A) - can be bundled with backend

**Recommended**:
- Frontend: Vercel free tier ($0)
- Backend: Render free tier ($0) or Railway ($5/mo)
- Model: Load from checkpoint file at startup

**Credit cost**: $0-$5/month
**Time**: ~2 hours

---

## Total Estimated Credit Consumption

| Phase | Platform | Cost | Time |
|-------|----------|------|------|
| 1. Fix EOS issue | CPU | $0 | ~1 hr |
| 2. Full training | GPU (Kaggle/Lambda) | $0-$15 | 10-40 hrs |
| 3. Evaluation | CPU | $0 | ~2 hrs |
| 4. Inference system | CPU | $0 | ~3 hrs |
| 5. FastAPI backend | CPU | $0 | ~4 hrs |
| 6. React frontend | CPU | $0 | ~8 hrs |
| 7. Deployment | Cloud free tier | $0-$5/mo | ~2 hrs |
| **Total** | | **$0-$20** | **~30-60 hrs** |

## Credit-Saving Rules

1. **Never run GPU experiments that don't provide new information**
   - If a diagnostic can run on CPU in <30 min, use CPU
   - If an experiment repeats a previous result, skip it
2. **Gate GPU usage behind CPU validation**
   - Model must learn non-EOS tokens on 100 samples (CPU) before any GPU training
3. **Use spot/preemptible instances where possible**
   - Checkpoint every 1000 steps so preemption does not waste progress
4. **Use mixed precision (fp16) on GPU**
   - ~2x speedup, ~50% memory savings, same accuracy
5. **Clear stopping criteria to avoid burning credits**
   - Stop if val loss plateaus for 2 epochs
   - Stop if non-EOS accuracy does not improve for 3 epochs
6. **Reserve credits for phases that actually need GPU**
   - Phases 3-7 can all run on CPU
   - Only Phase 2 (full training) requires GPU
7. **Prefer free tiers first**
   - Kaggle: 30 hrs/week free GPU
   - Colab: limited free GPU
   - Vercel/Render: free hosting tiers

## Immediate Next Steps (No Credits Required)

1. Run the learning-rate experiment on CPU:
   ```
   python experiments/lr_experiment.py
   ```
2. Analyze whether any configuration achieves >0% non-EOS accuracy
3. If yes: proceed to plan GPU training
4. If no: run deeper diagnostics (gradient flow, attention weights, target inspection)
5. All of the above costs $0 on local CPU

## Decision Gates

| Gate | Condition | Action if fail |
|------|-----------|----------------|
| Gate 1 | Non-EOS accuracy > 0% on 100 samples (CPU) | Do not start GPU training; run more diagnostics |
| Gate 2 | Non-EOS accuracy > 5% after 1 GPU epoch | Stop if not improving; investigate |
| Gate 3 | Val loss improvement < 0.01 for 2 epochs | Stop training; use best checkpoint |
| Gate 4 | BLEU > 0 on test set | If BLEU = 0, model is not producing valid translations; retrain |
| Gate 5 | End-to-end API test passes | Fix inference pipeline before deployment |

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| GPU instance preempted | Checkpoint every 1000 steps, resume from latest |
| Model doesn't learn on GPU either | CPU diagnostics must pass first (Gate 1) |
| Credits run out mid-training | Use free Kaggle first; Lambda as paid backup |
| Training too slow on T4 | Switch to A100 if budget allows; reduce max_length to 128 |
| BLEU score is very low | Train more epochs or increase model capacity |
| Deployment costs too much | Use free tiers; keep model on CPU for inference |