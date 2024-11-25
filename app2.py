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
from datetime import datetime
from flask_cors import CORS
import random

app = Flask(__name__)
CORS(app)

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
    docstore=InMemoryDocstore({}),
    index_to_docstore_id={}
)

# Load the vector store from a local file
vector_store = FAISS.load_local("faiss_index1", embeddings, allow_dangerous_deserialization=True)
retriever = vector_store.as_retriever()

# Set up system prompt for main response
system_prompt = (
    "You are a policy chatbot. We provided you policy you have to ans from it  "
    "Use the following pieces of retrieved context to answer with post link for example for more details (link) and always give link of related information always"
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
    if os.path.exists(LOG_FILE_PATH):
        with open(LOG_FILE_PATH, "r") as f:
            return json.load(f)
    return []

def save_logs():
    with open(LOG_FILE_PATH, "w") as f:
        json.dump(chat_logs, f)

def generate_random_questions():
    """Generate 3 random questions from the knowledge base."""
    # Get random documents from the vector store
    random_docs = vector_store.similarity_search("", k=3)
    
    # Create a prompt to generate questions from the random documents
    random_question_prompt = ChatPromptTemplate.from_messages([
        ("system", """Given the following policy document, generate one specific question under 15 to 10  words  that could be asked about its content.
        Make the question natural and conversational. Don't reference the document directly in the question."""),
        ("human", "{document}")
    ])
    
    questions = []
    for doc in random_docs:
        response = llm.invoke(random_question_prompt.format(document=doc.page_content))
        questions.append(response.content.strip())
    
    # If we somehow get fewer than 3 questions, add generic ones
    while len(questions) < 3:
        questions.append("Would you like to learn more about our policies?")
    
    return questions[:3]

chat_logs = load_logs()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    try:
        # Get the chatbot's response
        answer = rag_chain.invoke({"input": user_input})['answer']
        answer = answer.strip()
        
        # Generate random questions from the knowledge base
        random_questions = generate_random_questions()
        
        # Log the interaction
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        chat_logs.append({
            'timestamp': timestamp,
            'user_query': user_input,
            'chatbot_response': answer,
            'suggested_questions': random_questions
        })
        
        # Return both the response and random questions
        return jsonify({
            'response': answer,
            'question1': random_questions[0],
            'question2': random_questions[1],
            'question3': random_questions[2]
        })
    
    except Exception as e:
        return jsonify({
            'response': "Sorry, something went wrong. Please try again later.",
            'question1': "Would you like to try asking your question differently?",
            'question2': "Would you like to learn more about our policies?",
            'question3': "Can I help you find specific information?"
        })

@app.route('/logs')
def view_logs():
    return render_template('logs.html', chat_logs=chat_logs)

if __name__ == '__main__':
    app.run(debug=True)