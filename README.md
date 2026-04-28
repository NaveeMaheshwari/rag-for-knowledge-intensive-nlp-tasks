# RAG for Knowledge-Intensive NLP

End-to-end re-implementation of Lewis et al. (2020), Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (NeurIPS), for open-domain question answering. Built as a personal learning project to develop hands-on understanding of each RAG component — retrieval, indexing, marginalized generation, and joint fine-tuning.

## What's implemented

- **Retrieval**: DPR question and context encoders (`facebook/dpr`)
- **Indexing**: HNSW FAISS index over chunked TriviaQA passages (100-word chunks, inner-product similarity on L2-normalized embeddings)
- **Generation**: BART-Large generator with **RAG-token** decoding — per-token mixture over the top-k retrieved passages, weighted by retrieval probabilities
- **Training**: joint fine-tuning of DPR-Q and BART with retrieval-marginalized cross-entropy loss

## Repo layout

```
.
├── config.py               # central config: training, eval, indexing hyperparameters
├── build_index.py          # chunk corpus → DPR-encode → FAISS HNSW index
├── train_ragtoken.py       # joint DPR-Q + BART fine-tuning (RAG-token)
├── evaluate.py             # exact-match accuracy on test set
└── slides.pdf              # project slides + results
```

## Setup

```bash
pip install -r requirements.txt
```

Pre-trained checkpoints (placed under "models/"):
- DPR question encoder: "facebook/dpr-question_encoder-single-nq-base"
- DPR context encoder: "facebook/dpr-ctx_encoder-single-nq-base"
- Generator: "facebook/bart-large"

Dataset: [TriviaQA](https://huggingface.co/datasets/mandarjoshi/trivia_qa) (rc / wikipedia config).

## Usage

**1. Build the FAISS index over the retrieval corpus**

```bash
python build_index.py \
    --corpus_path data/TriviaQA/train/retrieval_corpus.jsonl \
    --output_dir TriviaQA/ \
    --model_path models/cxt_encoder
```

**2. Train (joint DPR-Q + BART, RAG-token)**

```bash
python train_ragtoken.py \
    --qa_file TriviaQA/train/qa_pairs.jsonl \
    --index_file TriviaQA/train_index.faiss \
    --passages_file TriviaQA/train_index.txt
```

Checkpoints, loss curves, and accuracy logs are written under the directories configured in `config.py` (`token_checkpoints/`, `token_loss_log/`, etc.).

**3. Evaluate (exact match)**

```bash
python evaluate.py
```

## Differences from the paper

This is a re-implementation under a constrained compute and data budget, so a gap from the paper's numbers is expected. Specifically:

- **Corpus**: paper uses the December 2018 Wikipedia dump (~21M passages); this work uses the TriviaQA training-split corpus only.
- **Datasets**: paper jointly trains/evaluates on NQ + TriviaQA + WebQuestions + CuratedTREC; this work uses TriviaQA alone.
- **Scale**: ~138K training QA pairs and ~17K test pairs here, with mini-batch SGD instead of the paper's full-scale training setup.
- **Variant**: only RAG-token is implemented (paper reports both RAG-sequence and RAG-token).

## Stack

PyTorch · HuggingFace Transformers · FAISS · sentence-transformers

## Reference

Lewis, P., Perez, E., Piktus, A., et al. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020. [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
