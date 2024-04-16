import os
from dotenv import load_dotenv
import pickle
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import time


ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
DB_DIR: str = os.path.join(ABS_PATH, "db")
st.set_page_config(page_title="Chat with PDF")

# model choices
MODEL_CHOICES = {
  "Llama2": "llama2",
  "Mistral": "mistral",
  "Gemma:7b": "gemma:7b"
}

# Sidebar
with st.sidebar:
  st.title('🤖 LLM Chat App')
  st.markdown('''
  ## About
  This app is an LLM-powered chatbot built using:
  - [Streamlit](https://streamlit.io/)
  - [LangChain](https://python.langchain.com/)
  - [Ollama](https://ollama.com/) 
  - [Llama2](https://llama.meta.com/) LLM model
  - [Mistral](https://mistral.ai/) LLM model
  - [Gemma](https://ollama.com/library/gemma) LLM model
  ''')

  # Model selection
  model_selection = st.selectbox("Choose your model", list(MODEL_CHOICES.keys()))

  add_vertical_space(5)
  st.write('Made by Group 20 - SY | CS6493')



# Load the model with callbackmanager
def load_model(model_name):
  llm = Ollama(
      model=model_name,
      verbose=True,
      callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
  )

  return llm

# Set up retrieval chian
def setup_retrieval_chain(llm, vectorstore):
    # create a retriever chain with conversation history
    history_aware_prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up to get information relevant to the conversation")
    ])
    retriever = vectorstore.as_retriever()
    history_aware_retriever = create_history_aware_retriever(llm, retriever, history_aware_prompt)
    
    # prompt for document retrieval and answer generation
    answer_prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer user's questions based on below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    document_chain = create_stuff_documents_chain(llm, answer_prompt)
    retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)
    
    return retrieval_chain


def main():
  st.title("Chat with PDF 💬")

  #Upload a pdf
  pdf = st.file_uploader("Upload your PDF", type='pdf')

  # Initialize session state for chat history
  # if "messages" not in st.session_state.keys():
  #   # print(f"key: {st.session_state.keys()}")
  #   st.session_state["messages"] = [
  #     {"role": "assistant", "content": "Hello there! Please upload your PDF and start chatting."}
  #   ]
  
  # Display all messages
  # for message in st.session_state["messages"]:
  #   with st.chat_message(message["role"]):
  #     st.write(message["content"])

  # Default directory
  if not os.path.exists(DB_DIR):
    os.makedirs(DB_DIR)

  # extract data / context from PDF
  if pdf is not None:
    # Initialize session state for chat history
    if 'last_uploaded_file' not in st.session_state or st.session_state['last_uploaded_file'] != pdf.name:
      # clear existing session state
      keys = list(st.session_state.keys())
      print(keys)
      for key in keys:
          del st.session_state[key]
        
      st.session_state['messages'] = [
        {
          "role": "assistant", 
          "content": "Hello there! Please upload your PDF and start chatting.",
        }
      ]
      st.session_state['last_uploaded_file'] = pdf.name  

    if "messages" not in st.session_state:
      st.session_state["messages"] = [
        {
          "role": "assistant",
          "content": "Hello there! Please upload your PDF and start chatting."
        }
      ]

    # Display all messages
    for message in st.session_state["messages"]:
      with st.chat_message(message["role"]):
        st.write(message["content"])

    # read pdf and extract text   
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
      text += page.extract_text()

    # split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000,
      chunk_overlap=200,
      length_function=len
    )
    chunks = text_splitter.split_text(text=text)

    # embeddings
    store_name = pdf.name[:-4]
    # st.write(f'{store_name}')
    file_path = os.path.join(DB_DIR, f"{store_name}.pkl")
    model_name = MODEL_CHOICES[model_selection]
    if os.path.exists(file_path):
      with open(file_path, "rb") as f:
        vectorStore = pickle.load(f)
    else:
      with st.spinner("Loading..."):
        # embeddings = OllamaEmbeddings(model="llama2")
        # embeddings = OllamaEmbeddings(model="gemma:7b")
        embeddings = OllamaEmbeddings(model=model_name)
        vectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        # vectorStore = FAISS.from_documents(chunks, embedding=embeddings)
        with open(file_path, "wb") as f:
          pickle.dump(vectorStore, f)

    # Load the selected model
    model_name = MODEL_CHOICES[model_selection]
    llm = load_model(model_name)
    retrieval_chain = setup_retrieval_chain(llm, vectorStore)

    # User query
    query = st.chat_input("Ask questions about your PDF file...")

    if query:
      print('history: ', st.session_state["messages"])
      print("=========================================")
      # Update chat history with the user's message
      st.session_state["messages"].append({"role": "user", "content": query})
      with st.chat_message("user"):
        st.write(query)
      # st.session_state["chat_history"].append(HumanMessage(content=query))

      if st.session_state["messages"][-1]["role"] != "assistant":
        with st.chat_message("assistant"):
          with st.spinner("Loading..."):
            #start timer
            start_time = time.time() 

            # Use retrieval chain to get response based on chat history
            response = retrieval_chain.invoke({
              "chat_history": st.session_state["messages"],
              "input": query
            })
            ai_response = response['answer']
            st.write(ai_response)

            #end timer
            end_time = time.time()
            processing_time = end_time - start_time
            st.write(f'Response time: {processing_time:.2f} seconds')
            # st.session_state["message"].append(AIMessage(content=response['answer']))

          new_ai_message = {"role": "assistant", "content": ai_response}
          st.session_state["messages"].append(new_ai_message)


if __name__ == '__main__':
  main()