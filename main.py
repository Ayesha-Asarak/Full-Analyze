# from configparser import ConfigParser
# from pymongo import MongoClient
# import google.generativeai as genai
# from fastapi import FastAPI, BackgroundTasks, HTTPException
# from pymongo.server_api import ServerApi
# from urllib.parse import quote_plus
# from bson import json_util
# from fastapi.encoders import jsonable_encoder
# import textwrap
# # import to_markdown

# # Initialize FastAPI app
# app = FastAPI()

# # Read configuration
# config = ConfigParser()
# config.read("credentials.ini")
# api_key = config["API_KEY"]["google_api_key"]

# # MongoDB connection
# username = quote_plus('ayesha')
# password = quote_plus('pTPivwignr4obw2U')
# cluster = 'cluster0.mhaksto.mongodb.net'
# uri = f'mongodb+srv://{username}:{password}@{cluster}/?retryWrites=true&w=majority&appName=Cluster0&ssl=true'
# client = MongoClient(uri, server_api=ServerApi('1'))
# db = client['AGENTSMTHS']
# collection = db['support_chats']
# def to_markdown(text):
#     text = text.replace("•", "  *")
#     indented_text = textwrap.indent(text, "\n ", predicate=lambda _: True)
#     return indented_text

# # Configure generative AI model
# genai.configure(api_key=api_key)
# model_gemini_pro = genai.GenerativeModel("gemini-pro")

# # Safety settings for content generation
# safety_settings = [
#     {"category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_NONE"},
#     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
#     {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
#     {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
#     {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
# ]

# @app.get("/fetch_data")
# async def fetch_all_data():
#     documents = list(collection.find())  # Fetch all documents from the collection
#     # Convert ObjectId to string for JSON serialization
#     for doc in documents:
#         doc['_id'] = str(doc['_id'])  # Convert ObjectId to string
#     return {"data": jsonable_encoder(documents)}

# # FastAPI endpoint to generate PPT slides for a specific topic
# @app.post("/summary/{topic}")
# async def summary(topic: str, background_tasks: BackgroundTasks):
#     if not topic:
#         # Fetch the first document to use its topic as the default
#         first_document = collection.find_one()
#         if first_document:
#             topic = first_document.get("topic", "default_topic")
#         else:
#             topic = "default_topic"

#     # Fetch document based on the provided or default topic
#     document = collection.find_one({"topic": topic})
#     if document:
#         # Assuming "content" field exists in your MongoDB documents
#         prompt = document.get("content", f"I want a main intention of the user in one sentence {topic}")
#     else:
#         prompt = f"I want a main intention of the user in one sentence {topic}"

#     # Generate content using the generative AI model
#     response = model_gemini_pro.generate_content(prompt, safety_settings=safety_settings)
#     summary = to_markdown(response.text)
    
#     # Here you can add code to save or process the slide_titles asynchronously if needed

#     return {"summary of chat": summary}
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
sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
# Function to encode text using SBERT model
def encode_text_sbert(text):
    return sbert_model.encode([text])[0]
# def encode_text_sbert(text):
#     return sbert_model.encode([text])[0]

# Function to calculate cosine similarity between two vectors
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Function to cluster messages using SBERT embeddings and cosine similarity
def cluster_messages_sbert(messages, existing_clusters=None, threshold=0.75):
    clusters = existing_clusters if existing_clusters else []
    remaining_messages = messages.copy()

    while remaining_messages:
        seed_message = remaining_messages.pop(0)['message']
        seed_embedding = encode_text_sbert(seed_message)

        # Find the most suitable cluster or create a new one
        cluster_found = False
        for cluster in clusters:
            centroid = np.mean([encode_text_sbert(msg['message']) for msg in cluster], axis=0)
            similarity = cosine_similarity(seed_embedding, centroid)
            if similarity >= threshold:
                cluster.append({"_id": str(len(cluster) + 1), "message": seed_message})
                cluster_found = True
                break
        
        if not cluster_found:
            clusters.append([{"_id": "1", "message": seed_message}])

    return clusters

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
    
    # Load existing clusters from database
    existing_clusters = []
    existing_clusters_docs = list(group_collections.find())
    for cluster_doc in existing_clusters_docs:
        existing_clusters.append(cluster_doc['messages'])
    
    # Cluster user messages using SBERT embeddings and cosine similarity
    clustered_messages = cluster_messages_sbert(user_messages, existing_clusters)
    
    # Save updated clusters in the database
    group_collections.delete_many({})  # Clear existing clusters
    for idx, cluster in enumerate(clustered_messages):
        group_collections.insert_one({"cluster_id": idx, "messages": cluster})
    
    # Prepare clusters for response serialization
    serialized_clusters = []
    for cluster in clustered_messages:
        serialized_cluster = [{"_id": msg["_id"], "message": msg["message"]} for msg in cluster]
        serialized_clusters.append(serialized_cluster)
    
    # Update saved_clusters variable
    saved_clusters = serialized_clusters
    
    return {"clusters": serialized_clusters}
# # Function to calculate cosine similarity between two vectors
# def cosine_similarity(vec1, vec2):
#     return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# #     return clusters
# def cluster_messages_sbert(messages, threshold=0.75):
#     # Check if clusters already exist in the database
#     existing_clusters = list(group_collections.find())
    
#     if existing_clusters:
#         clusters = [cluster_document["messages"] for cluster_document in existing_clusters]
#         return clusters
    
#     # Perform clustering if clusters don't exist
#     clusters = []
#     remaining_messages = messages.copy()
    
#     while remaining_messages:
#         seed_message = remaining_messages.pop(0)['message']
#         seed_embedding = encode_text_sbert(seed_message)
        
#         # Find the most suitable cluster or create a new one
#         cluster_found = False
#         for cluster in clusters:
#             centroid = np.mean([encode_text_sbert(msg['message']) for msg in cluster], axis=0)
#             similarity = cosine_similarity(seed_embedding, centroid)
#             if similarity >= threshold:
#                 cluster.append({"_id": str(len(cluster) + 1), "message": seed_message})
#                 cluster_found = True
#                 break
        
#         if not cluster_found:
#             clusters.append([{"_id": "1", "message": seed_message}])
    
#     # Save clusters in database for future use
#     for idx, cluster in enumerate(clusters):
#         group_collections.insert_one({"cluster_id": idx, "messages": cluster})
    
#     return clusters
# saved_clusters = None
# # Endpoint to fetch and save clustered messages
# @app.get("/clustered-messages-sbert")
# async def get_clustered_messages_sbert():
#     # Check if clusters are already saved globally or in database
#     global saved_clusters
#     if saved_clusters:
#         return {"clusters": saved_clusters}
    
#     # Fetch all documents from the collection
#     documents = list(collection.find())
    
#     user_messages = []
#     for doc in documents:
#         if 'messages' in doc:
#             user_messages.extend([msg for msg in doc['messages'] if msg.get('sender') == 'user'])
    
#     if not user_messages:
#         return {"message": "No user messages found"}
    
#     # Cluster user messages using SBERT embeddings and cosine similarity
#     clustered_messages = cluster_messages_sbert(user_messages)
    
#     # Save clusters for future use
#     saved_clusters = clustered_messages
    
#     # Prepare clusters for response serialization
#     serialized_clusters = []
#     for cluster in clustered_messages:
#         serialized_cluster = [{"_id": msg["_id"], "message": msg["message"]} for msg in cluster]
#         serialized_clusters.append(serialized_cluster)
    
#     return {"clusters": serialized_clusters}

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
    prompt = f"Rephrase the following question as instruction to admin to check and fulfill the user's want in one sentence: {question}"
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


# @app.websocket("/notifications")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()

#     # Fetch rephrased questions from the database or a global variable
#     clusters_cursor = group_collections.find()
#     clusters = list(clusters_cursor)

#     if not clusters:
#         await websocket.send_json({"message": "No clusters found"})
#         return

#     rephrased_questions = []
#     for cluster_document in clusters:
#         cluster = cluster_document.get("messages", [])
#         # Skip clusters that are primarily greetings or thank you messages
#         if all(is_greeting_or_thank_you(msg['message']) for msg in cluster):
#             continue
#         rep_question = get_representative_question(cluster)
#         if rep_question:
#             rephrased_question = rephrase_question(rep_question, model_gemini_pro)
#             rephrased_questions.append(rephrased_question)

#     if not rephrased_questions:
#         await websocket.send_json({"message": "No rephrased questions found"})
#         return

#     # Send each rephrased question as a notification with buttons
#     for rephrased_question in rephrased_questions:
#         notification = {
#             "question": rephrased_question,
#             "buttons": ["Done", "Ignore"]
#         }
#         await websocket.send_json(notification)

#     # Close the connection after sending notifications
#     await websocket.close()

# @app.websocket("/notifications")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()

#     clusters_cursor = group_collections.find()
#     clusters = list(clusters_cursor)

#     if not clusters:
#         await websocket.send_json({"message": "No clusters found"})
#         return

#     rephrased_questions = []
#     cluster_ids = []  # Store cluster IDs to reference later
#     for cluster_document in clusters:
#         cluster = cluster_document.get("messages", [])
#         if all(is_greeting_or_thank_you(msg['message']) for msg in cluster):
#             continue
#         rep_question = get_representative_question(cluster)
#         if rep_question:
#             rephrased_question = rephrase_question(rep_question, model_gemini_pro)
#             rephrased_questions.append(rephrased_question)
#             cluster_ids.append(str(cluster_document["_id"]))  # Save the cluster ID

#     if not rephrased_questions:
#         await websocket.send_json({"message": "No rephrased questions found"})
#         return

#     for rephrased_question, cluster_id in zip(rephrased_questions, cluster_ids):
#         notification = {
#             "question": rephrased_question,
#             "buttons": ["Done", "Ignore"],
#             "cluster_id": cluster_id
#         }
#         await websocket.send_json(notification)

#     while True:
#         data = await websocket.receive_json()
#         action = data.get("action")
#         cluster_id = data.get("cluster_id")

#         if action in ["Done", "Ignore"] and cluster_id:
#             group_collections.delete_one({"_id": ObjectId(cluster_id)})
#             await websocket.send_json({"message": f"Cluster {cluster_id} deleted"})

#     await websocket.close()
  
# @app.delete("/clusters/{cluster_id}")
# async def delete_cluster(cluster_id: str):
#     result = group_collections.delete_one({"_id": ObjectId(cluster_id)})
#     if result.deleted_count == 1:
#         return {"message": "Cluster deleted"}
#     else:
#         raise HTTPException(status_code=404, detail="Cluster not found")  
# # @app.websocket("/notifications")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()

#     clusters_cursor = group_collections.find()
#     clusters = list(clusters_cursor)

#     if not clusters:
#         await websocket.send_json({"message": "No clusters found"})
#         return

#     rephrased_questions = []
#     cluster_ids = []  # Store cluster IDs to reference later
#     for cluster_document in clusters:
#         cluster = cluster_document.get("messages", [])
#         if all(is_greeting_or_thank_you(msg['message']) for msg in cluster):
#             continue
#         rep_question = get_representative_question(cluster)
#         if rep_question:
#             rephrased_question = rephrase_question(rep_question, model_gemini_pro)
#             rephrased_questions.append(rephrased_question)
#             cluster_ids.append(str(cluster_document["_id"]))  # Save the cluster ID

#     if not rephrased_questions:
#         await websocket.send_json({"message": "No rephrased questions found"})
#         return

#     for rephrased_question, cluster_id in zip(rephrased_questions, cluster_ids):
#         notification = {
#             "question": rephrased_question,
#             "buttons": ["Done", "Ignore"],
#             "cluster_id": cluster_id
#         }
#         await websocket.send_json(notification)

#     while True:
#         data = await websocket.receive_json()
#         action = data.get("action")
#         cluster_id = data.get("cluster_id")

#         if action in ["Done", "Ignore"] and cluster_id:
#             group_collections.delete_one({"_id": ObjectId(cluster_id)})
#             await websocket.send_json({"message": f"Cluster {cluster_id} deleted"})
#         else:
#             await websocket.send_json({"message": "Invalid action or cluster ID"})   







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
        cluster_ids = []  # Store cluster IDs to reference later
        for cluster_document in clusters:
            cluster = cluster_document.get("messages", [])
            if all(is_greeting_or_thank_you(msg['message']) for msg in cluster):
                continue
            rep_question = get_representative_question(cluster)
            if rep_question:
                rephrased_question = rephrase_question(rep_question, model_gemini_pro)
                rephrased_questions.append(rephrased_question)
                cluster_ids.append(str(cluster_document["_id"]))  # Save the cluster ID

        if not rephrased_questions:
            await websocket.send_json({"message": "No rephrased questions found"})
            continue

        for rephrased_question, cluster_id in zip(rephrased_questions, cluster_ids):
            notification = {
                "question": rephrased_question,
                "buttons": ["Done", "Ignore"],
                "cluster_id": cluster_id
            }
            await websocket.send_json(notification)

        data = await websocket.receive_json()
        action = data.get("action")
        cluster_id = data.get("cluster_id")

        if action in ["Done", "Ignore"] and cluster_id:
            # Delete the cluster from database
            result = group_collections.delete_one({"_id": ObjectId(cluster_id)})
            if result.deleted_count == 1:
                await websocket.send_json({"message": f"Cluster {cluster_id} deleted"})
            else:
                await websocket.send_json({"message": f"Failed to delete cluster {cluster_id}"})
        else:
            await websocket.send_json({"message": "Invalid action or cluster ID"})

        # Check if the action was "Done" to create a new cluster for related messages
        if action == "Done":
            new_message = data.get("new_message")
            if new_message:
                # Cluster new message
                new_cluster = cluster_messages_sbert([{"message": new_message}])
                new_cluster_id = str(group_collections.insert_one({"messages": new_cluster[0]}).inserted_id)

                # Rephrase the new question
                rephrased_new_question = rephrase_question(new_message, model_gemini_pro)

                # Send notification for the new question
                new_notification = {
                    "question": rephrased_new_question,
                    "buttons": ["Done", "Ignore"],
                    "cluster_id": new_cluster_id
                }
                await websocket.send_json(new_notification)

    await websocket.close()
    
    
# # Endpoint to fetch and save representative questions
# @app.get("/representative-questions")
# async def get_and_save_representative_questions():
#     # Fetch all clusters from the group_collections collection
#     clusters_cursor = group_collections.find()
#     clusters = list(clusters_cursor)
    
#     if not clusters:
#         raise HTTPException(status_code=404, detail="No clusters found in the group_collections collection.")
    
#     representative_questions = []
#     for cluster_document in clusters:
#         cluster = cluster_document.get("messages", [])
#         # Skip clusters that are primarily greetings or thank you messages
#         if all(is_greeting_or_thank_you(msg['message']) for msg in cluster):
#             continue
#         rep_question = get_representative_question(cluster)
#         if rep_question:
#             # Save representative question in MongoDB
#             result = representative_questions_collection.insert_one({
#                 "cluster_id": str(cluster_document["_id"]),  # Convert ObjectId to string
#                 "representative_question": rep_question
#             })
#             # Add to response list
#             representative_questions.append({
#                 "cluster_id": str(cluster_document["_id"]),
#                 "representative_question": rep_question
#             })
    
#     # Manually serialize response to JSON
#     return json.dumps({"representative_questions": representative_questions})


# Endpoint to fetch and save representative questions
# @app.get("/representative-questions")
# async def get_and_save_representative_questions():
#     # Fetch all user messages from the collection
#     documents = list(collection.find())
    
#     user_messages = []
#     for doc in documents:
#         if 'messages' in doc:
#             user_messages.extend([msg for msg in doc['messages'] if msg.get('sender') == 'user'])
    
#     if not user_messages:
#         raise HTTPException(status_code=404, detail="No user messages found.")
    
#     # Cluster user messages using SBERT embeddings and cosine similarity
#     clustered_messages = cluster_messages_sbert(user_messages)
    
#     # Prepare representative questions
#     representative_questions = []
#     for cluster in clustered_messages:
#         # Skip clusters that are primarily greetings or thank you messages
#         if all(is_greeting_or_thank_you(msg['message']) for msg in cluster):
#             continue
#         rep_question = cluster[0]['message']  # Using the first message as representative question
#         representative_questions.append({
#             "cluster_id": cluster[0]["_id"],
#             "representative_question": rep_question
#         })
    
    # Return representative questions
    # return {"representative_questions": representative_questions}

# Endpoint to fetch user messages from MongoDB and cluster them using SBERT    

            
# def get_representative_question(cluster):
#     if not cluster:
#         return None
#     messages = [msg['message'] for msg in cluster]
#     embeddings = model.encode(messages)
    
#     # Compute pairwise cosine similarity
#     cosine_matrix = util.pytorch_cos_sim(embeddings, embeddings).numpy()
    
#     # Sum of similarities for each message
#     similarity_sum = np.sum(cosine_matrix, axis=1)
    
#     # Index of the message with the highest sum of similarities
#     representative_idx = np.argmax(similarity_sum)
    
#     return cluster[representative_idx]['message']

# # Function to rephrase a question using the generative AI model
# def rephrase_question(question, model_gemini_pro):
#     prompt = f"Rephrase the following question as instrusction to admin to check and full fill the users want in one sentence: {question}"
#     response = model_gemini_pro.generate_content(prompt)
#     return response.text.strip()

# @app.get("/rephrased-representative-questions")
# async def get_rephrased_representative_questions():
#     # Fetch all clusters from the group_collections collection
#     clusters_cursor = group_collections.find()
#     clusters = list(clusters_cursor)
    
#     if not clusters:
#         raise HTTPException(status_code=404, detail="No clusters found in the group_collections collection.")
    
#     representative_questions = []
#     rephrased_questions = []
#     for cluster_document in clusters:
#         cluster = cluster_document.get("messages", [])
#         rep_question = get_representative_question(cluster)
#         if rep_question:
#             representative_questions.append(rep_question)
#             rephrased_question = rephrase_question(rep_question, model_gemini_pro)
#             rephrased_questions.append(rephrased_question)
    
#     return {"rephrased_representative_questions": rephrased_questions}

# @app.post("/generate_summaries")
# async def generate_summaries(background_tasks: BackgroundTasks):
#     summaries = []

#     # Fetch all documents from the collection
#     documents = collection.find()

#     for document in documents:
#         # Extract user messages
#         messages = document.get("messages", [])

#         # Identify user's main intention
#         user_intent = ""
#         for msg in messages:
#             if msg["sender"] == "user" and "download" in msg["message"].lower():
#                 user_intent = msg["message"].strip()
#                 break

#         # Generate summary based on identified user intent or default prompt
#         if user_intent:
#             prompt = f"i need the main intension of the sender(user) in one sentence: {user_intent}"
#         else:
#             prompt = "Unable to identify user's main intention."

#         # Generate content using the generative AI model
#         response = model_gemini_pro.generate_content(prompt)
#         summary = to_markdown(response.text)

#         # Convert ObjectId to string for JSON serialization
#         document['_id'] = str(document['_id'])

#         # Append the document with its summary to the list
#         summaries.append({"document": jsonable_encoder(document), "summary": summary})

#         # Save the summary into the summaries collection
#         summaries_collection.insert_one({"document_id": document['_id'], "summary": summary})

#     if not summaries:
#         raise HTTPException(status_code=404, detail="No documents found in the collection")

#     return summaries

# # Function to perform clustering using SBERT embeddings and cosine similarity
# def cluster_messages_sbert(messages, threshold=0.75):
#     clusters = []
#     remaining_messages = messages.copy()
    
#     while remaining_messages:
#         seed_message = remaining_messages.pop(0)['message']
#         seed_embedding = encode_text_sbert(seed_message)
        
#         # Find the most suitable cluster or create a new one
#         cluster_found = False
#         for cluster in clusters:
#             centroid = np.mean([encode_text_sbert(msg['message']) for msg in cluster], axis=0)
#             similarity = cosine_similarity(seed_embedding, centroid)
#             if similarity >= threshold:
#                 cluster.append({"_id": str(len(cluster) + 1), "message": seed_message})
#                 cluster_found = True
#                 break
        
#         if not cluster_found:
#             clusters.append([{"_id": "1", "message": seed_message}])
    