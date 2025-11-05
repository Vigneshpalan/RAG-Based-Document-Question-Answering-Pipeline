# Project Notes: RAG-Based Document QA Pipeline

## 1. Design Trade-offs

- **Chunk Size vs Context:**  
  Using 350-word chunks with 150-word overlap balances context retention with embedding efficiency. Larger chunks preserve more context but increase embedding size and retrieval latency.

- **Embedding Model Selection:**  
  `all-MiniLM-L6-v2` offers a good trade-off between speed and semantic quality. Domain-specific models could improve precision but may increase resource requirements.

- **Cross-Encoder Re-ranking:**  
  Improves retrieval precision but adds additional computation. Useful when top-k retrieval alone is noisy.

- **LLM Choice:**  
  Flan-T5 provides concise, controlled outputs. Larger models could reduce hallucinations but require more compute and memory.

---

## 2. Retrieval Strategy

- **FAISS Index:**  
  Stores embeddings for fast approximate nearest neighbor search. Scales well to large document corpora.

- **Top-k Retrieval:**  
  Default `k=3` retrieves the most semantically relevant chunks. Can be tuned based on desired precision/recall trade-off.

- **Re-ranking with Cross-Encoder:**  
  Ranks the top-k chunks by relevance score to improve grounding of generated answers.

- **Grounding Evaluation:**  
  Measures how well the LLM answer aligns with retrieved chunks using semantic similarity thresholds (e.g., 0.7).

---

## 3. Guardrails

- LLM is **restricted to retrieved chunks** to reduce hallucinations.  
- Prompts instruct the model to cite documents and avoid generating information not present in the context.  
- Adaptive semantic thresholds ensure only high-confidence answers are counted in metrics.

---

## 4. Scaling Plan

- **Document Corpus:**  
  Can scale to hundreds or thousands of PDFs using FAISS for efficient vector search.  
- **Chunking & Indexing:**  
  Parallel processing can be used to embed large corpora.  
- **Distributed Retrieval:**  
  FAISS supports GPU acceleration and sharding to handle very large datasets.  
- **Model Serving:**  
  LLM inference can be scaled using batching or cloud endpoints for higher throughput.

---

## 5. Future Considerations

- Incorporate **domain-specific embeddings** to improve precision.  
- Add **iterative retrieval loops** where the model can request additional context.  
- Integrate **feedback loop** from users to refine retrieval and answer quality over time.  
