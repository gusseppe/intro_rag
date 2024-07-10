import streamlit as st
import glob
import plotly.express as px
import pandas as pd

from typing import TypedDict, Optional
from backend import RAG

from streamlit_option_menu import option_menu


def read_markdown_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        return "Markdown file not found."
    except Exception as e:
        return f"Error reading markdown file: {str(e)}"
    
def main():
    st.set_page_config(page_title="SageMaker Documentation RAG", layout="wide")

    rag = RAG(debug=False)
    rag_graph = rag.rag_graph

    class GraphState(TypedDict):
        question: str
        misuse: Optional[bool]
        response: Optional[str]
        answer: Optional[str]

    def clear_chat_history():
        st.session_state.messages = []
    
    # Initialize session state variables
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'state' not in st.session_state:
        st.session_state.state = {}

    # Sidebar for navigation
    with st.sidebar:
        selected = option_menu(
            menu_title="Navigation",
            options=["Home", "Chat", "Debug"],
            icons=["house", "chat-dots", "bug"],
            menu_icon="book",
            styles={
                "container": {"padding": "5!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "25px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "#02ab21"},
            }
        )

    if selected == "Home":
        st.title("🏠 Welcome to SageMaker Documentation RAG")
        
        markdown_files = glob.glob("./sagemaker_documentation/*.md")

        st.write(f"""
        ## 🚀 How to Use the SageMaker Q&A System
        
        1. Navigate to the 'Chat' and type your question about SageMaker.
        2. If you want to start fresh, use the 'Clear Chat History' button.
        3. Check the 'Debug' section for detailed information about the state of the system.
        
        ## 📊 System Statistics
        
        - **Number of Documents**: **`{len(markdown_files)}`** markdown documents about SageMaker documentation.

        """)
        filenames = [file.split('/')[-1] for file in markdown_files]
        # In this part, I gave all the filename to a LLM to provide the most improtant topics
        # This is the result:
        data = {
            "Topic": [
                "Amazon SageMaker Toolkits and Integrations",
                "SageMaker Job Definitions and Configuration",
                "SageMaker Endpoints and Models",
                "SageMaker Pipelines and Workflow",
                "SageMaker Security and Compliance",
                "SageMaker Features and Feature Groups",
                "SageMaker Monitoring and Resources",
                "SageMaker Projects and Templates",
                "SageMaker Marketplace",
                "SageMaker Notebooks",
                "SageMaker Roles and Permissions",
                "SageMaker RL and Training",
                "Miscellaneous",
                # "Others"
            ],
            "Count": [10, 36, 32, 8, 5, 6, 11, 14, 8, 6, 6, 8, 9]
            # "Count": [10, 36, 32, 8, 5, 6, 11, 14, 8, 6, 6, 8, 9, 177]
        }

        df = pd.DataFrame(data)

        # Create the plot
        fig = px.bar(df, x="Topic", y="Count", title="Most important topics",
                    labels={"Count": "Number of Files", "Topic": "Topics"},
                    color="Count")

        # Display the plot using streamlit
        st.plotly_chart(fig)

        # st.write(f"""
        # Feel free to explore and ask questions about any SageMaker-related topic. Our RAG system is here to help!
        # """)

    elif selected == "Chat":
        st.title("💬 Chat with SageMaker RAG")
        

        if st.button("Clear Chat History"):
            clear_chat_history()
        
 
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
            
            # If there are sources then show them
            if "sources" in message and message["sources"]:
                st.write("📚 Reference Sources:")
                cols = st.columns(min(len(message["sources"]), 4))  # Up to 4 columns
                for idx, (source, col) in enumerate(zip(message["sources"], cols), 1):
                    with col:
                        with st.popover(f"Source {idx}", use_container_width=True):
                            st.markdown("**Metadata:**")
                            for key, value in source.metadata.items():
                                st.markdown(f"- **{key}**: {value}")
                            
                            st.markdown("**Full Markdown Content:**")
                            if 'source' in source.metadata:
                                markdown_content = read_markdown_file(source.metadata['source'])
                                st.markdown(markdown_content)
                            else:
                                st.code(source.page_content, language="markdown")
        
        # Chat input
        if prompt := st.chat_input("What would you like to know about Amazon SageMaker?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                # Use rag_graph to get the answer
                with st.spinner("Thinking... 🤔"):
                    initial_state = GraphState(question=prompt)
                    for output in rag_graph.stream(initial_state):
                        for node_name, state in output.items():
                            if node_name == 'get_final_answer' and 'answer' in state:
                                full_response = state['answer']
                                st.session_state.state = state  # Save the state
                                message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
 
                sources = []
                
     
                if ('response' in st.session_state.state and 
                    st.session_state.state['response'] is not None and 
                    'context' in st.session_state.state['response']):
                    sources = st.session_state.state['response']['context']
                    
                    # Display sources
                    st.write("📚 Reference Sources:")
                    cols = st.columns(min(len(sources), 4))  # Up to 4 columns
                    for idx, (source, col) in enumerate(zip(sources, cols), 1):
                        with col:
                            with st.popover(f"Source {idx}", use_container_width=True):
                                st.markdown("**Metadata:**")
                                for key, value in source.metadata.items():
                                    st.markdown(f"- **{key}**: {value}")
                                
                                st.markdown("**Full markdown Content:**")
                                if 'source' in source.metadata:
                                    markdown_content = read_markdown_file(source.metadata['source'])
                                    st.markdown(markdown_content)
                                else:
                                    st.code(source.page_content, language="markdown")
                

                elif 'misuse' in st.session_state.state and st.session_state.state['misuse']:
                    st.warning("This question is not related to Amazon SageMaker. Please ask a SageMaker-specific question.")
                
  
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response,
                    "sources": sources
                })

    elif selected == "Debug":
        st.title("🐛 Debug Information")
        

        if 'state' in st.session_state and st.session_state.state:
            with st.container(border=True, height=600):
                # pass
                # st.text_area("State:", value=str(st.session_state.state), height=300)
                # st.code(repr(st.session_state.state), language="python")
                st.write(st.session_state.state)
                # st.markdown(f"```\n{st.session_state.state}\n```")
        else:
            st.write("No state information available.")

if __name__ == "__main__":
    main()
