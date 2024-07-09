# Solution 



This repository contains the implementation of an assistant for the sagemaker documentation.
Check the [thought_process.ipynb](thought_process.ipynb) for a detailed explanation.

## Demo


[![Streamlit App Demo](https://img.youtube.com/vi/5Imka0GCNGI/0.jpg)](https://www.youtube.com/watch?v=5Imka0GCNGI)

## Approach

The following approach uses RAG + CoT + Reranker implemented with Langgraph.

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

## Files

- **thought_process.ipynb**: `It contains all the explanations. Please, look at the notebook for a detailed explanation`.
- **app.py**: GUI for the RAG system.
- **backend.py**: Backend that contains the graph and the logic for the RAG system.
- **qa_list.json**: Contains the initial ground truth list of questions and answers.
- **qa_list_answers_***: Answers for each approach (naive RAG, no RAG, RAG with chain-of-thought, and RAG with chain-of-thought and re-ranking).
- **qa_list_shuffled.json**: final shuffled list of Q&A used as the ground truth testing dataset.



## Methodology 

### Setup and baseline

1. **Data**: All data are markdown files with mixed content types including explanations, JSON, YAML, etc.
2. **Baseline solution**: Implemented a naive RAG approach to establish a baseline.
3. **Evaluation**: Created a `ground truth testing dataset` and used accuracy as the `evaluation metric` similar to MMLU paper.

### Naive RAG implementation

- **Embedding**: Used HuggingFace's `BAAI/bge-small-en model` for creating embeddings.
- **Chunking**: Split documents into chunks of 1000 tokens with a 200-token overlap.
- **Vector store**: A vector store using Chroma. 

### Chain-of-Thought (CoT) 

- **System prompt**: Naive RAG + `chain-of-thought` process (examples of step by step reasoning).
- **Improved accuracy**: Better accuracy than Naive RAG.

### Re-Ranking 

- **Cross-Encoder reranker**: Implemented a reranker using HuggingFace's `BAAI/bge-reranker-base model`.
- **Contextual compression retriever**: Enhanced the retriever to use the reranker for better document relevance.

### Final solution

- **Evaluation**: Achieved high accuracy with the combined `RAG + CoT + Reranker approach`. Refactor for langgraph.
- **Future work**: Explore further enhancements such as reflection, hybrid search, graphrag and a custom `cognitive architecture for RAG`.

## How to run

1. **Install dependencies**: Ensure all necessary libraries are installed.
```sh
   pip install -r requirements.txt
   streamlit run app.py
```

## Disclaimer

- For a production-ready system, further testing and optimization are required.
- The system is designed for the Sagemaker documentation and may not generalize to other domains.
- The accuracy metric may not be representative of real-world performance.

## TODO

- add a function to update the vectore store when a new document is added or updated.
- apply SOTA best practices for the RAG system (https://arxiv.org/pdf/2407.01219)
- create fewshot examples to finetune the rag prompts using dspy. Consider human in the loop to help to get examples.
- test with more rag metrics like faithfulness, relevance, etc.
- create an docker image for the backend and the frontend.
- create an image for the vector store.
- create an image to deploy Ollama for offline inference.
- orchestrate the application using docker-compose.
