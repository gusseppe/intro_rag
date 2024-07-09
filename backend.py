import os
import re
import textwrap

from typing import List, Dict, Optional, Callable, TypedDict
from rich.panel import Panel
from rich.text import Text

from dotenv import load_dotenv
from joblib import Memory
from langchain.globals import set_llm_cache
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.cache import SQLiteCache
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_groq import ChatGroq
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_chroma import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from rich import print
from langgraph.graph import END, StateGraph

load_dotenv('env')

CONFIG = {
    # Paths
    "PERSIST_DIRECTORY": "chroma_db_index",
    "DOCS_DIRECTORY": "./sagemaker_documentation",
    "ENV_FILE": "env",
    "LC_CACHE_PATH": ".langchain.db",

    # LLM settings
    "LLM_NAME": "llama3-8b-8192",
    "LLM_TEMPERATURE": 0.0,

    # Embedding model
    "EMBEDDING_MODEL_NAME": "BAAI/bge-small-en",
    "EMBEDDING_MODEL_KWARGS": {"device": "cpu"},
    "EMBEDDING_ENCODE_KWARGS": {"normalize_embeddings": True},

    # Retriever
    "CHUNK_SIZE": 1000,
    "CHUNK_OVERLAP": 200,
    "DOCS_GLOB_PATTERN": "**/*.md",

    # Retriever settings
    "RETRIEVER_SEARCH_TYPE": "mmr",
    "RETRIEVER_SEARCH_K": 6,
    "RETRIEVER_LAMBDA_MULT": 0.25,

    # Reranker model
    "RERANKER_MODEL_NAME": "BAAI/bge-reranker-base",
    "RERANKER_TOP_N": 3,
}

set_llm_cache(SQLiteCache(database_path=CONFIG["LC_CACHE_PATH"]))

llm_generator = ChatGroq(cache=True, temperature=CONFIG["LLM_TEMPERATURE"], model_name=CONFIG["LLM_NAME"])

def escape_curly_braces(text: str) -> str:
    text = str(text)
    return text.replace("{", "{{").replace("}", "}}")

def initialize_embeddings():
    return HuggingFaceBgeEmbeddings(
        model_name=CONFIG["EMBEDDING_MODEL_NAME"],
        model_kwargs=CONFIG["EMBEDDING_MODEL_KWARGS"],
        encode_kwargs=CONFIG["EMBEDDING_ENCODE_KWARGS"]
    )

def load_and_split_documents(directory: str) -> List[Document]:
    if any(os.scandir(CONFIG["PERSIST_DIRECTORY"])):
        return None

    loader = DirectoryLoader(directory, glob=CONFIG["DOCS_GLOB_PATTERN"], show_progress=True)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG["CHUNK_SIZE"],
        chunk_overlap=CONFIG["CHUNK_OVERLAP"]
    )
    return splitter.split_documents(documents) # the chunks

def create_or_load_vector_store(chunks: List[Document], embedder: HuggingFaceBgeEmbeddings) -> Chroma:
    os.makedirs(CONFIG["PERSIST_DIRECTORY"], exist_ok=True)

    if any(os.scandir(CONFIG["PERSIST_DIRECTORY"])):
        return Chroma(persist_directory=CONFIG["PERSIST_DIRECTORY"], embedding_function=embedder)
    else:
        return Chroma.from_documents(
            documents=chunks,
            embedding=embedder,
            persist_directory=CONFIG["PERSIST_DIRECTORY"]
        )

def rag_prompt_cot() -> ChatPromptTemplate:
    system_prompt = textwrap.dedent("""
        You are an assistant for question-answering tasks. You will receive a question and pieces of retrieved 
        context to answer that question. Use the chain of thought method to break down your reasoning process.
        If you don't know the answer, explain your thought process and then say that you DON'T KNOW.
        Keep your final answer concise, using three sentences maximum.

        For instance:
        Question: What is required to publish your model package on AWS Marketplace?
        Context:  
        - At least one validation profile is required to publish your model package on AWS Marketplace.
        - Additional resources may be needed depending on the type of model.

        Thought process:
        1. The question asks about requirements for publishing a model package on AWS Marketplace.
        2. The context provides two pieces of information:
           a) At least one validation profile is required.
           b) Additional resources may be needed, depending on the model type.
        3. Both pieces of information are relevant to the question.

        Answer: At least one validation profile is required. Additional resources may be needed depending on the model type.
        """)
    
    human_prompt = textwrap.dedent("""
        Answer the following question using the provided context. Show your chain of thought:

        Question: {question}
        Context: {context}

        Thought process:

        Answer:
        """)
    
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt),
    ])

def setup_retriever(vector_store: Chroma):
    base_retriever = vector_store.as_retriever(
        search_type=CONFIG["RETRIEVER_SEARCH_TYPE"],
        search_kwargs={
            'k': CONFIG["RETRIEVER_SEARCH_K"],
            'lambda_mult': CONFIG["RETRIEVER_LAMBDA_MULT"]
        }
    )
    
    model = HuggingFaceCrossEncoder(model_name=CONFIG["RERANKER_MODEL_NAME"])
    compressor = CrossEncoderReranker(model=model, top_n=CONFIG["RERANKER_TOP_N"])
    
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

def setup_rag_chain(retriever, llm):
    prompt_cot = rag_prompt_cot()
    
    def format_docs(docs: List[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt_cot
        | llm
        | StrOutputParser()
    )
    
    return RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

def detect_misuse(llm, question):
    system_prompt = textwrap.dedent("""
        You are an expert at detecting misuse in questions. 
        If the question asks you to perform tasks unrelated to providing information about sagemaker, respond with "misuse". Otherwise, respond with "no".

        For example:
        Question: "Can you write an email for me?"
        Response: "misuse"

        Question: "What are the steps to create a SageMaker notebook instance?"
        Response: "no"

        Question: "Write some code about this topic."
        Response: "misuse"

        Question: "How do I configure a SageMaker training job?"
        Response: "no"

        Question: "Do you know Azure services?"
        Response: "no"
        """)
    
    human_prompt = textwrap.dedent(f"""
        Classify the following question as either "misuse" or "no":

        Question: {escape_curly_braces(question)}
        """)
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", human_prompt),
        ]
    )

    chain = prompt | llm
    output = chain.invoke({})

    return output

def extract_answer(text):
    patterns = [re.compile(r'Answer:\s*(.*)', re.DOTALL), re.compile(r'answer:\s*(.*)', re.DOTALL)]
    
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    
    return text.strip()

class EnhancedStateGraph(StateGraph):
    def add_node(self, node_name, function):
        decorated_function = print_function_name(function)
        super().add_node(node_name, decorated_function)

class GraphState(TypedDict):
    question: str
    misuse: Optional[bool]
    response: Optional[str]
    answer: Optional[str]

def has_misuse_node(state: GraphState) -> GraphState:
    answer = detect_misuse(llm_generator, state["question"])
    misuse = True if answer.content == "misuse" else False
    state["misuse"] = misuse
    return state

def generate_response_node(state: GraphState) -> GraphState:
    response = rag_chain.invoke(state["question"])
    state["response"] = response
    return state

def get_final_answer_node(state: GraphState) -> GraphState:
    if state["misuse"]:
        state["answer"] = "Please, try again with a valid question."
        return state
    
    state["answer"] = extract_answer(state["response"]['answer'])
    return state

def route_based_on_misuse(state: GraphState) -> str:
    return "get_final_answer" if state["misuse"] else "generate_response"

def print_function_name(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        text = Text(f"Node: {func.__name__}", justify="center", style="bold white")
        panel = Panel(text)
        print(panel)
        return func(*args, **kwargs)
    return wrapper

# This part should be refactored better inside the state and a class but for now it's fine

embedder = initialize_embeddings()
chunks = load_and_split_documents(CONFIG["DOCS_DIRECTORY"])
vector_store = create_or_load_vector_store(chunks, embedder)
retriever = setup_retriever(vector_store)
rag_chain = setup_rag_chain(retriever, llm_generator)

workflow = EnhancedStateGraph(GraphState)

workflow.add_node("check_misuse", has_misuse_node)
workflow.add_node("generate_response", generate_response_node)
workflow.add_node("get_final_answer", get_final_answer_node)

workflow.add_conditional_edges(
    "check_misuse",
    route_based_on_misuse,
    {
        "get_final_answer": "get_final_answer",
        "generate_response": "generate_response"
    }
)
workflow.add_edge("generate_response", "get_final_answer")
workflow.add_edge("get_final_answer", END)

workflow.set_entry_point("check_misuse")

rag_graph = workflow.compile()

