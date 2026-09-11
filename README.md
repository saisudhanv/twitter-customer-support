# Hiver Support Agent - README

A production-quality AI customer support system that classifies intents, retrieves similar cases, and decides whether to auto-reply or escalate.

## Quick Start

### 1. Setup Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Copy configuration template
cp .env.example .env

# (Optional) Add API keys if using LLM features
# Edit .env and add OPENAI_API_KEY if you have one
```

### 2. Download Data
```bash
# Phase 1: Inspect and download raw dataset from Kaggle
# (First time only - downloads to ~/.cache/kagglehub)
python -m scripts.inspect_dataset
```

### 3. Build Conversations & Intents
```bash
# Phase 2-3: Extract conversation pairs from tweets
python -m scripts.prepare_data

# Phase 4: Discover intent taxonomy and label conversations
python -m scripts.discover_intents

# Phase 5: Create golden evaluation set
python -m scripts.build_golden_set
```

### 4. Run Evaluation
```bash
# Phase 6: Evaluate system vs baselines (requires manually labeled golden_set.csv)
python -m scripts.run_evaluation

# Results saved to: artifacts/metrics/evaluation_results.csv
```

### Quick Mode
The current evaluation script reads the already-generated processed and golden CSV files; changing `DATA_SAMPLE_SIZE` or `EVAL_SAMPLE_SIZE` alone does not reduce this run. To create a smaller experiment, regenerate the processed and golden files with smaller sample sizes before evaluating.

```bash
DATA_SAMPLE_SIZE=100 python -m scripts.prepare_data
python -m scripts.discover_intents
EVAL_SAMPLE_SIZE=50 python -m scripts.build_golden_set
python -m scripts.run_evaluation
```

## Project Structure

```
hiver-assignment/
├── src/
│   ├── config.py                 # Configuration management
│   ├── schemas.py                # Pydantic data models
│   ├── data_loader.py            # Kaggle dataset download
│   ├── conversation_builder.py   # Tweet → conversation conversion
│   ├── intent_taxonomy.py        # Intent classification
│   ├── baselines.py              # Trivial + Simple baselines
│   ├── retrieval.py              # TF-IDF similarity search
│   ├── pipeline.py               # End-to-end pipeline
│   └── metrics.py                # Evaluation metrics
│
├── scripts/
│   ├── inspect_dataset.py        # Phase 1-2
│   ├── prepare_data.py           # Phase 3
│   ├── discover_intents.py       # Phase 4
│   ├── build_golden_set.py       # Phase 5
│   └── run_evaluation.py         # Phase 6
│
├── data/
│   ├── raw/                      # Downloaded from Kaggle
│   ├── processed/                # Extracted conversations
│   └── golden/                   # Golden evaluation set
│
├── artifacts/
│   └── metrics/                  # Evaluation results
│
├── .env.example                  # Configuration template
├── requirements.txt              # Python dependencies
├── REPORT.md                     # Results & methodology
├── DECISIONS.md                  # Engineering decisions
├── PROGRESS.md                   # Phase-by-phase log
└── README.md                     # This file
```

## Key Components

### Intent Classification
Classifies customer messages into 8 intent categories:
- order_tracking (34%) - "where is my order?"
- general_inquiry (38.6%) - "how do I...?"
- refund_return (4%) - "I want to return"
- product_info (3.6%) - "is it compatible?"
- account_access (1.8%) - "I can't login"
- quality_defect (1.8%) - "it's broken"
- payment_issue (1%) - "my card was declined"
- other (15.2%) - ambiguous

### Escalation Policy
Decides between AUTO_HANDLE and ESCALATE based on:
- Intent type (payment issues → escalate)
- Model confidence (< 60% → escalate)
- Risk keywords (fraud, urgent, lawsuit → escalate)

### Retrieval System
Finds similar customer messages from historical cases using TF-IDF similarity.

## Results

### Performance

These are the results from the manually labeled `golden_set.csv`, after removing all golden examples from the training and retrieval data. The evaluator normalizes granular human labels into the eight canonical intent IDs.

| Metric | Trivial | Simple (TF-IDF) | AI System |
|--------|---------|-----------------|-----------|
| Intent Accuracy | 23.5% | 39.5% | **43.5%** |
| Intent Macro F1 | 4.8% | 14.0% | **28.1%** |
| Escalation Accuracy | 53.5% | 52.0% | **47.5%** |
| False Auto-Handle Rate | 0% | **81.3%** ❌ | **19.6%** |

**Key Insight**: The system is safer than the simple policy but still auto-handles 19.6% of cases that human labels marked for escalation. It is not production-ready; improving escalation recall and intent classification is the next priority.

### Dataset
- **Source**: 2,811,774 customer support tweets from Kaggle
- **Brand**: AmazonHelp (169,840 conversations, 71,049 unique customers)
- **Training**: 500 conversation pairs
- **Evaluation**: 200 stratified examples (all intent types represented)

## Configuration

Edit `.env` to customize:

```bash
# LLM Provider (openai, anthropic, ollama)
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4-turbo

# Data Settings
DATA_SAMPLE_SIZE=500              # Conversations to sample
EVAL_SAMPLE_SIZE=200              # Golden examples
RANDOM_SEED=42                    # Reproducibility

# Embeddings & Retrieval
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RETRIEVAL_TOP_K=3

# Caching
USE_LLM_CACHE=true
USE_EMBEDDING_CACHE=true
CACHE_DIR=artifacts/cache

# Debugging
DEBUG=false
VERBOSE=true
```

## Testing

### Run Tests
```bash
pytest tests/ -v
```

### Test Coverage
The test directory currently contains no test modules. The evaluation command has been validated, but automated unit and integration coverage still needs to be added for conversation construction, intent classification, retrieval leakage prevention, escalation policy, schema validation, and the end-to-end pipeline.

## Reproducibility

All results are reproducible with fixed random seeds:

```python
import numpy as np
np.random.seed(42)  # Fixed in config.py
```

To reproduce exact results:
```bash
RANDOM_SEED=42 python -m scripts.run_evaluation
```

All intermediate data and results are saved:
- `data/processed/conversations_with_intents.csv` - Training data with labels
- `data/golden/golden_candidates.csv` - Unlabeled labeling template
- `data/golden/golden_set.csv` - Human-labeled evaluation set
- `artifacts/metrics/evaluation_results.csv` - Performance results

## Production Deployment

### Monitoring
Track these metrics continuously:
- Intent prediction confidence distribution
- Escalation rate (target: 15-20%)
- False auto-handle rate (should stay < 2%)
- Customer satisfaction on auto-handled cases

### Cost Estimation
- Per-message cost: ~$0.01-0.05 (depends on LLM)
- 1M messages/month: $10,000-50,000 (rough estimate)
- Caching reduces costs by 30-50% (repeated queries)

### Scaling
- Current: 100-1000 messages/day (Python, single-threaded)
- Production: Requires async job queue + load balancer
- Estimated throughput: 10,000+ messages/day with proper infrastructure

## Common Issues

### Issue: Dataset won't download
```
FileNotFoundError: Kaggle dataset not found
```
**Solution**: Requires `kagglehub` API. See https://github.com/Kaggle/kagglehub

### Issue: Slow first run
**Reason**: First evaluation indexes 500 cases (5-10 seconds)
**Solution**: Results are cached - subsequent runs are faster

### Issue: Intent classification too conservative
**Reason**: Keyword-based approach is interpretable but can miss paraphrases
**Solution**: Add LLM layer (planned PHASE 7)

## Documentation

- **REPORT.md** - Executive summary, results, methodology
- **DECISIONS.md** - Engineering decisions & rationale
- **PROGRESS.md** - Detailed phase-by-phase execution log
- **src/schemas.py** - All data model definitions

## Architecture Diagram

```
Customer Message
    ↓
┌─────────────────────────────────┐
│  Intent Classification          │
│  (Keyword Matching + TF-IDF)    │
│  → intent_id, confidence        │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  Similarity Retrieval           │
│  (TF-IDF Search)                │
│  → top-3 similar cases          │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  Reply Generation               │
│  (Grounded in Retrieved Cases)  │
│  → contextual response          │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  Escalation Policy              │
│  (Rules: intent + confidence +  │
│   keywords)                     │
│  → AUTO_HANDLE or ESCALATE      │
└─────────────────────────────────┘
    ↓
Structured JSON Output
```

## Next Steps (Future Work)

1. **LLM Integration** - Add GPT-4 for generation layer
2. **Human Evaluation** - Manually label golden set for ground truth
3. **Multilingual Support** - Handle non-English messages
4. **Sentiment Analysis** - Detect angry/sarcastic customers
5. **A/B Testing** - Compare vs human support baseline
6. **Fine-tuning** - Domain-specific models if needed

## Citation

If you use this project in research or production:

```bibtex
@project{hiver_support_agent_2024,
  title={Hiver Customer Support Agent},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/hiver-support}
}
```

## License

MIT License - See LICENSE file for details

---

**Questions?** See DECISIONS.md for design rationale.  
**Issues?** Check PROGRESS.md for phase-by-phase debugging hints.
