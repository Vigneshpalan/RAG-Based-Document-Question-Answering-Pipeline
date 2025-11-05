# RAG-Based Document Question Answering Pipeline

## Overview

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline for extracting and answering questions from technical PDF documents. It combines **semantic search**, **cross-encoder reranking**, and **LLM-based answer generation** with grounding evaluation to ensure that answers are **grounded in source documents**, reducing hallucinations and improving reliability.

---

## Features

- **Document Ingestion:** Extract text from PDFs using PyMuPDF.
- **Chunking:** Split text into overlapping chunks (350 words per chunk, 150-word overlap) to preserve context.
- **Embedding:** Encode chunks using a Sentence Transformer (`all-MiniLM-L6-v2`).
- **Indexing:** Store embeddings in **FAISS** for fast similarity search.
- **Retrieval & Reranking:** Retrieve top-`k` relevant chunks (e.g., `k=3`) and rerank using a Cross-Encoder model.
- **LLM Answer Generation:** Generate concise, context-based answers using **Flan-T5**. Prompt constraints prevent hallucinations.
- **Evaluation:**
  - **Grounding Precision:** Measures how well generated answers are supported by retrieved chunks using a semantic similarity threshold (e.g., 0.7).
  - **Retrieval Recall:** Measures how well relevant chunks are retrieved.

---

## Pipeline Flow

1. **Load PDFs** → Extract raw text.
2. **Chunking** → Split text into overlapping segments.
3. **Embedding** → Convert chunks into vector representations.
4. **Indexing** → Store embeddings in FAISS.
5. **Query Retrieval** → Retrieve top-`k` relevant chunks.
6. **Cross-Encoder Reranking** → Rank chunks by relevance.
7. **LLM Answer Generation** → Generate answers from top-ranked chunks only.
8. **Evaluation** → Compute grounding precision and retrieval recall.

---

## Installation

```bash
pip install PyMuPDF
pip install sentence-transformers
pip install faiss-cpu
pip install transformers
pip install torch
```

---

## Usage

1. Set the path to your PDF documents:

```python
DOCS_DIR = "path_to_pdf_folder"
```

2. Load documents, chunk, and build embeddings:



3. Retrieve answers for a query:

```python
query = "What is the purpose of the cyclone separator?"
answer = generate_answer(query, k=3)
print(answer)
```

4. Evaluate the system:

```python
test_questions = ["Question 1", "Question 2", ...]
evaluate_generated_answer_support(test_questions, k=3, threshold=0.7)
evaluate_retrieval_support(test_questions, k=3)
```

---

## Metrics

- **Grounding Precision@0.7 (max):** 0.93  
- **Grounding Precision@0.7 (avg):** 0.71  
- **Retrieval Recall@3:** 0.93  

> These metrics indicate that answers are generally well-supported by retrieved chunks and retrieval is effective.

---

## Notes

- Optimal performance depends on **chunk size**, **k (top retrieved chunks)**, and **similarity threshold**.
- LLM answer quality depends on **domain knowledge**; domain-specific models can improve precision.
- The pipeline can be extended to other document types or larger corpora.

---

## License

This project is licensed under the **MIT License**.

