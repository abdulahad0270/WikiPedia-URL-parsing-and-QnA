import os
from dotenv import load_dotenv
import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from streamlit_extras.add_vertical_space import add_vertical_space

# Load the API key from the .env file
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')  # Replace with your actual API key

# Function to initialize session state
def init_session_state():
    return st.session_state.setdefault('messages', [])

# Function to update chat history in the session state
def update_chat_history(user_message, assistant_message):
    st.session_state.messages.append({"role": "user", "content": user_message})
    st.session_state.messages.append({"role": "assistant", "content": assistant_message})

# Initialize session state
init_session_state()

# Front-end chat set up
st.title('Wikipedia QnA')
url = st.text_input("Enter the URL")

if url:
    try:
        # For pulling the information from the web
        response = requests.get(url)  # Web requesting
        if response.status_code == 200:
            # Extracting HTML content from response
            soup = BeautifulSoup(response.content, 'html.parser')
            page_content = soup.get_text()  # Retrieves the text from the parsed HTML.

            # Using Language Models for Chatbot Interaction
            embeddings = OpenAIEmbeddings(api_key=openai_api_key)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
            chunks = text_splitter.split_text(text=page_content)
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)  # Vector store for the text chunks using embeddings.

    # Error handling using try and except
    except requests.RequestException as e:
        st.error(f"Request Error: {e}")
    except Exception as e:
        st.error(f"Error: {e}")
else:
    pass
    #st.error("")

# Sidebar contents
with st.sidebar:
    st.title('Wikipedia QnA')
    if st.button("New Chat"):
        st.session_state.clear()
        st.experimental_rerun()

    # Chat interface
    st.header("URL Chatting WebApp")
    query = st.text_input("Ask a question about the content from the URL")

    if query and url:
        chat_history = st.session_state.get("messages", [])
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        llm = ChatOpenAI(api_key=openai_api_key)
        CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template("Your custom prompt template here.")

        # Retrieval chain to generate a response to the user's query.
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm,
            VectorStore.as_retriever(),
            condense_question_prompt=CUSTOM_QUESTION_PROMPT,
            memory=memory,
        )

        # Generates a response based on the user's query using the language model and the content from the URL.
        response = conversation_chain({"question": query, "chat_history": chat_history})

        # The assistant's response is then displayed in a separate expandable section
        with st.expander("Assistant's Response"):
            st.text(response["answer"])

        # Update chat history
        update_chat_history(query, response["answer"])

    # Display chat history
    st.header("Chat History")
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'**User**: {message["content"]}')
        else:
            st.markdown(f'**Assistant**: {message["content"]}')

with st.sidebar:
    #st.title("PDF Chat Bot")
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
 
    ''')
    add_vertical_space(1)
    st.write('Made by Abdul Ahad')