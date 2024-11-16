# app.py
from flask import Flask, render_template, request, jsonify

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
    "Use the following pieces of retrieved context to answer with post link for example for more deatils (link)"
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    try:
        # Add a delay to simulate "typing" if needed
        answer = rag_chain.invoke({"input": user_input})['answer']
        return jsonify({'response': answer.strip()})
    except Exception as e:
        return jsonify({'response': "Sorry, something went wrong. Please try again later."})

if __name__ == '__main__':
    app.run(debug=True)