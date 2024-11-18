# app.py
from flask import Flask, render_template, request, jsonify
import json
from langchain_community.document_loaders.csv_loader import CSVLoader
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores import FAISS
import time
app = Flask(__name__)

os.environ["OPENAI_API_KEY"] = ""
llm = ChatOpenAI(model="gpt-4-turbo")

embeddings = OpenAIEmbeddings()

# Determine embedding dimension
embedding_dim = len(embeddings.embed_query(" "))

# Create a FAISS index with L2 (Euclidean) distance
index = faiss.IndexFlatL2(embedding_dim)

# Initialize FAISS vector store
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore({}),  # Initialize with empty docstore
    index_to_docstore_id={}
)

# Load the vector store from a local file
vector_store = FAISS.load_local("faiss_index1", embeddings, allow_dangerous_deserialization=True)

retriever = vector_store.as_retriever()

# Set up system prompt
system_prompt = (
    "You are an plicy chatbot We provided you policy you have to ans from it  "
    "Use the following pieces of retrieved context to answer with post link for example for more deatils (link) and always give link of realted information always"
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Create the question-answer chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)



LOG_FILE_PATH = "chat_logs.json"

def load_logs():
    """Load chat logs from the file."""
    if os.path.exists(LOG_FILE_PATH):
        with open(LOG_FILE_PATH, "r") as f:
            return json.load(f)
    return []

def save_logs():
    """Save chat logs to the file."""
    with open(LOG_FILE_PATH, "w") as f:
        json.dump(chat_logs, f)

# Initialize chat_logs by loading from the file
chat_logs = load_logs()

@app.route('/')
def index():
    return render_template('index.html')

from datetime import datetime

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    try:
        # Get the chatbot's response
        answer = rag_chain.invoke({"input": user_input})['answer']
        answer = answer.strip()  # Clean up any extra spaces/newlines
        
        # Log the user query and response with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        chat_logs.append({'timestamp': timestamp, 'user_query': user_input, 'chatbot_response': answer})
        
        # Return the response
        return jsonify({'response': answer})
    
    except Exception as e:
        return jsonify({'response': "Sorry, something went wrong. Please try again later."})

@app.route('/logs')
def view_logs():
    # Display the logged interactions
    return render_template('logs.html', chat_logs=chat_logs)
if __name__ == '__main__':
    app.run(debug=True)