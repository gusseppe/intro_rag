import streamlit as st
from typing import TypedDict, Optional
from backend import RAG

from streamlit_option_menu import option_menu


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
        
        st.write("""
        ## 🚀 How to Use the SageMaker Q&A System
        
        1. Navigate to the 'Chat' and type your question about SageMaker.
        2. If you want to start fresh, use the 'Clear Chat History' button.
        3. Check the 'Debug' section for detailed information about the state of the system.
        
        ## 📊 System Statistics
        
        - **Number of Documents**: 336 markdown documents about SageMaker documentation
        - **Topics Covered**: The documentation includes information about:
          * Kubernetes with SageMaker
          * Rust SageMaker
          * Marketplace
          * SageMaker Notebook
          * And many more SageMaker-related topics!
        
        Feel free to explore and ask questions about any SageMaker-related topic. Our RAG system is here to help!
        """)

    elif selected == "Chat":
        st.title("💬 Chat with SageMaker RAG")
        
        # Button to clear chat history
        if st.button("Clear Chat History"):
            clear_chat_history()
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
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
            st.session_state.messages.append({"role": "assistant", "content": full_response})

    elif selected == "Debug":
        st.title("🐛 Debug Information")
        
        # Print the state in markdown
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
