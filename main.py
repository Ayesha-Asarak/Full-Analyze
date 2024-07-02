
from configparser import ConfigParser
from pymongo import MongoClient
import google.generativeai as genai
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pymongo.server_api import ServerApi
from urllib.parse import quote_plus
from bson import json_util
from fastapi.encoders import jsonable_encoder
import textwrap
from pydantic import BaseModel
from typing import List, Dict
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
from sentence_transformers import SentenceTransformer, util
from fastapi import WebSocket
import asyncio
import json
from bson.objectid import ObjectId
from bson import ObjectId
# Initialize FastAPI app
app = FastAPI()

# Read configuration
config = ConfigParser()
config.read("credentials.ini")
api_key = config["API_KEY"]["google_api_key"]

# MongoDB connection
username = quote_plus('ayesha')
password = quote_plus('pTPivwignr4obw2U')
cluster = 'cluster0.mhaksto.mongodb.net'
uri = f'mongodb+srv://{username}:{password}@{cluster}/?retryWrites=true&w=majority&appName=Cluster0&ssl=true'
client = MongoClient(uri, server_api=ServerApi('1'))
db = client['AGENTSMTHS']
collection = db['support_chats']
group_collections = db["group_collections"]
representative_questions_collection = db["representative_questions"]

class RephraseRequest(BaseModel):
    question: str
    
def to_markdown(text):
    text = text.replace("•", "  *")
    indented_text = textwrap.indent(text, "\n ", predicate=lambda _: True)
    return indented_text

# Configure generative AI model
genai.configure(api_key=api_key)
model_gemini_pro = genai.GenerativeModel("gemini-pro")

# Safety settings for content generation
safety_settings = [
    {"category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

@app.get("/fetch_data")
async def fetch_all_data():
    documents = list(collection.find())  # Fetch all documents from the collection
    # Convert ObjectId to string for JSON serialization
    for doc in documents:
        doc['_id'] = str(doc['_id'])  # Convert ObjectId to string
    return {"data": jsonable_encoder(documents)}

@app.get("/fetch_user_messages")
async def fetch_user_messages():
    documents = list(collection.find())  # Fetch all documents from the collection
    
    user_messages = []
    for doc in documents:
        if 'messages' in doc:
            # Filter user messages from the messages array
            user_messages.extend([msg for msg in doc['messages'] if msg.get('sender') == 'user'])

    return {"user_messages": jsonable_encoder(user_messages)}
# Load SBERT model for sentence embeddings
sbert_model = SentenceTransformer('all-mpnet-base-v2')
# Function to encode text using SBERT model
def encode_text_sbert(text):
    return sbert_model.encode([text])[0]


# Function to calculate cosine similarity between two vectors
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def encode_text_sbert(text):
    return sbert_model.encode([text])[0]

# Filter out unwanted messages like greetings and thank-yous
def filter_unwanted_messages(messages):
    unwanted_phrases = [
        "hello", "hi", "hey", "greetings", "good morning", "good afternoon",
        "good evening", "thank you", "thanks", "bye", "goodbye"
    ]
    filtered_messages = [
        message for message in messages
        if not any(phrase in message['message'].lower() for phrase in unwanted_phrases)
    ]
    return filtered_messages

# Function to cluster messages using SBERT embeddings and cosine similarity
def cluster_messages_sbert(messages, existing_clusters=None, threshold=0.75):
    clusters = existing_clusters if existing_clusters else []
    remaining_messages = messages.copy()
    
    current_date = datetime.utcnow().isoformat()
    status = "Unread"

    while remaining_messages:
        seed_message = remaining_messages.pop(0)['message']
        seed_embedding = encode_text_sbert(seed_message)

        # Find the most suitable cluster or create a new one
        cluster_found = False
        for cluster in clusters:
            centroid = np.mean([encode_text_sbert(msg['message']) for msg in cluster['messages']], axis=0)
            similarity = cosine_similarity([seed_embedding], [centroid])[0][0]
            if similarity >= threshold:
                cluster['messages'].append({"_id": str(len(cluster['messages']) + 1), "message": seed_message})
                cluster_found = True
                break
        
        if not cluster_found:
            new_cluster = {
                "created_date": current_date,
                "status": status,
                "messages": [{"_id": "1", "message": seed_message}]
            }
            clusters.append(new_cluster)
#
    return clusters

# Global variable to save clusters
saved_clusters = None

# Endpoint to fetch and save clustered messages
@app.get("/clustered-messages-sbert")
async def get_clustered_messages_sbert():
    global saved_clusters
    
    # Check if clusters are already saved globally or in database
    if saved_clusters:
        return {"clusters": saved_clusters}
    
    # Fetch all user messages from the collection
    documents = list(collection.find())
    user_messages = [msg for doc in documents for msg in doc.get('messages', []) if msg.get('sender') == 'user']
    
    if not user_messages:
        return {"message": "No user messages found"}
    
    # Filter out unwanted messages
    user_messages = filter_unwanted_messages(user_messages)
    
    # Load existing clusters from database
    existing_clusters = []
    existing_clusters_docs = list(group_collections.find())
    for cluster_doc in existing_clusters_docs:
        existing_clusters.append({
            "created_date": cluster_doc.get("created_date", datetime.utcnow().isoformat()),  # Default to current date if missing
            "status": cluster_doc.get("status", "Unread"),  # Default to "Unread" if missing
            "messages": cluster_doc.get("messages", [])  # Default to an empty list if missing
        })
    
    # Cluster user messages using SBERT embeddings and cosine similarity
    clustered_messages = cluster_messages_sbert(user_messages, existing_clusters)
    
    # Save updated clusters in the database
    group_collections.delete_many({})  # Clear existing clusters
    for idx, cluster in enumerate(clustered_messages):
        group_collections.insert_one({
            "cluster_id": idx,
            "created_date": cluster["created_date"],
            "status": cluster["status"],
            "messages": cluster["messages"]
        })
    
    # Prepare clusters for response serialization
    serialized_clusters = []
    for cluster in clustered_messages:
        serialized_cluster = {
            "created_date": cluster["created_date"],
            "status": cluster["status"],
            "messages": [{"_id": msg["_id"], "message": msg["message"]} for msg in cluster["messages"]]
        }
        serialized_clusters.append(serialized_cluster)
    
    # Update saved_clusters variable
    saved_clusters = serialized_clusters
    
    return {"clusters": serialized_clusters}



model = SentenceTransformer('paraphrase-MiniLM-L6-v2')






greeting_phrases = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
thank_you_phrases = ["thank you", "thanks", "thank you so much", "thanks a lot"]

def is_greeting_or_thank_you(message):
    message_lower = message.lower()
    for phrase in greeting_phrases + thank_you_phrases:
        if phrase in message_lower:
            return True
    return False

def get_representative_question(cluster):
    if not cluster:
        return None
    messages = [msg['message'] for msg in cluster]
    embeddings = model.encode(messages)
    
    # Compute pairwise cosine similarity
    cosine_matrix = util.pytorch_cos_sim(embeddings, embeddings).numpy()
    
    # Sum of similarities for each message
    similarity_sum = np.sum(cosine_matrix, axis=1)
    
    # Index of the message with the highest sum of similarities
    representative_idx = np.argmax(similarity_sum)
    
    return cluster[representative_idx]['message']


# Function to rephrase a question using the generative AI model
def rephrase_question(question, model_gemini_pro):
    prompt = f"Rephrase the following question as instruction to admin from system to check and solve the user's problem in one sentence more accurately: {question}"
    response = model_gemini_pro.generate_content(prompt)
    return response.text.strip()

@app.get("/rephrased-representative-questions")
async def get_rephrased_representative_questions():
    # Fetch all clusters from the group_collections collection
    clusters_cursor = group_collections.find()
    clusters = list(clusters_cursor)
    
    if not clusters:
        raise HTTPException(status_code=404, detail="No clusters found in the group_collections collection.")
    
    representative_questions = []
    rephrased_questions = []
    for cluster_document in clusters:
        cluster = cluster_document.get("messages", [])
        # Skip clusters that are primarily greetings or thank you messages
        if all(is_greeting_or_thank_you(msg['message']) for msg in cluster):
            continue
        rep_question = get_representative_question(cluster)
        if rep_question:
            representative_questions.append(rep_question)
            rephrased_question = rephrase_question(rep_question, model_gemini_pro)
            rephrased_questions.append(rephrased_question)
    
    return {"rephrased_representative_questions": rephrased_questions}
class RepresentativeQuestionsResponse(BaseModel):
    representative_questions: List[str]








# Set to track sent notifications
sent_notifications = set()


@app.websocket("/notifications")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    while True:
        # Fetch all clusters from the database
        clusters_cursor = group_collections.find()
        clusters = list(clusters_cursor)

        if not clusters:
            await websocket.send_json({"message": "No clusters found"})
            continue

        rephrased_questions = []
        cluster_ids = []

        for cluster_document in clusters:
            cluster_id = str(cluster_document["_id"])

            # Check if notification for this cluster has already been sent or cluster is marked as read
            if cluster_id in sent_notifications:
                continue
            
            cluster_status = cluster_document.get("status", "")
            if cluster_status == "Read":
                sent_notifications.add(cluster_id)  # Add to set even if it's read to prevent future notifications
                continue

            cluster = cluster_document.get("messages", [])
            if all(is_greeting_or_thank_you(msg['message']) for msg in cluster):
                continue
            
            rep_question = get_representative_question(cluster)
            
            if rep_question:
                rephrased_question = rephrase_question(rep_question, model_gemini_pro)
                rephrased_questions.append(rephrased_question)
                cluster_ids.append(cluster_id)
            else:
                # Create a new cluster with unread status
                new_cluster = {
                    "messages": cluster,
                    "status": "Unread"
                }
                insert_result = group_collections.insert_one(new_cluster)
                cluster_ids.append(str(insert_result.inserted_id))  # Save the new cluster ID

        if not rephrased_questions:
            await websocket.send_json({"message": "No new notifications found"})
        else:
            for rephrased_question, cluster_id in zip(rephrased_questions, cluster_ids):
                notification = {
                    "question": rephrased_question,
                    "buttons": ["Done", "Ignore"],
                    "cluster_id": cluster_id
                }
                await websocket.send_json(notification)
                sent_notifications.add(cluster_id)

        data = await websocket.receive_json()
        action = data.get("action")
        cluster_id = data.get("cluster_id")

        if action in ["Done", "Ignore"] and cluster_id:
            update_query = {"_id": ObjectId(cluster_id)}
            update_fields = {"$set": {"status": "Read"}} 
            update_result = group_collections.update_one(update_query, update_fields)

            if update_result.modified_count == 1:
                await websocket.send_json({"message": f"Cluster {cluster_id} status updated"})
            else:
                await websocket.send_json({"message": f"Failed to update status for cluster {cluster_id}"})
        else:
            await websocket.send_json({"message": "Invalid action or cluster ID"})

    await websocket.close()