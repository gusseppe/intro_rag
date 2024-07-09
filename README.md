# Solution

This repository contains the implementation of an assistant for the sagemaker documentation.

## Demo

<video width="600" controls>
  <source src="img/app.webm" type="video/webm">
  Your browser does not support the video tag.
</video>

## Approach

```python
                   +-----------+          
                   | __start__ |          
                   +-----------+          
                         *                
                         *                
                         *                
                 +--------------+         
                 | check_misuse |         
                 +--------------+         
                 ...            ...       
               ..                  ..     
             ..                      ..   
+-------------------+                  .. 
| generate_response |                ..   
+-------------------+              ..     
                 ***            ...       
                    **        ..          
                      **    ..            
               +------------------+       
               | get_final_answer |       
               +------------------+       
                         *                
                         *                
                         *                
                    +---------+           
                    | __end__ |           
                    +---------+           

```

## Benchmark

![Final RAG](img/benchmark.png)

## Files and Structure

- **thought_process.ipynb**: `It contains all the explanations. Please, look at the notebook for a detailed explanation`.
- **app.py**: GUI for the RAG system.
- **backend.py**: Backend that contains the graph and the logic for the RAG system.
- **qa_list.json**: Contains the initial ground truth list of questions and answers.
- **qa_list_answers_***: Answers for each approach (naive RAG, no RAG, RAG with chain-of-thought, and RAG with chain-of-thought and re-ranking).
- **qa_list_shuffled.json**: Shuffled list of questions and answers used as the ground truth testing dataset.



## Detailed Methodology

### Initial Setup and Baseline

1. **Understand the Data**: All data are markdown files with mixed content types including explanations, JSON, YAML, etc.
2. **Baseline Solution**: Implemented a naive RAG approach to establish a baseline.
3. **Evaluation**: Created a `ground truth testing dataset` and used accuracy as the `evaluation metric`.

### Naive RAG Implementation

- **Embedding Model**: Used HuggingFace's `BAAI/bge-small-en model` for creating embeddings.
- **Document Splitting**: Split documents into chunks of 1000 tokens with a 200-token overlap.
- **Vector Store**: Created a vector store using Chroma.

### Chain-of-Thought (CoT) Enhancement

- **System Prompt**: Developed a detailed system prompt to guide the model through a `chain-of-thought` process.
- **Improved Accuracy**: Observed significant improvement in accuracy with the chain-of-thought approach.

### Re-Ranking Mechanism

- **Cross-Encoder Reranker**: Implemented a reranker using HuggingFace's `BAAI/bge-reranker-base model`.
- **Contextual Compression Retriever**: Enhanced the retriever to use the reranker for better document relevance.

### Final Solution

- **Evaluation**: Achieved high accuracy with the combined `RAG + CoT + Reranker approach`.
- **Future Work**: Plans to explore further enhancements such as reflection, hybrid search, and integration with `cognitive architectures`.

## How to Run

1. **Install Dependencies**: Ensure all necessary libraries are installed.
```sh
   pip install -r requirements.txt
   streamlit run app.py
```

## Future work

- create an image for the backend and the frontend.
- create an image for the vector store.
- create an image to deploy Ollama for offline inference.
- orchestrate the application using docker-compose.
