
# from configparser import ConfigParser
# import motor.motor_asyncio
# import google.generativeai as genai
# from fastapi import FastAPI, BackgroundTasks, HTTPException, WebSocket
# from fastapi.encoders import jsonable_encoder
# from bson import ObjectId
# from bson.json_util import dumps
# import textwrap
# from pydantic import BaseModel
# from typing import List, Dict
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer
# import numpy as np
# from datetime import datetime
# import logging
# import asyncio
# import json

# # Initialize FastAPI app
# app = FastAPI()

# # Read configuration
# config = ConfigParser()
# config.read("credentials.ini")
# api_key = config["API_KEY"]["google_api_key"]

# # MongoDB connection with motor
# username = 'ayesha'
# password = 'pTPivwignr4obw2U'
# cluster = 'cluster0.mhaksto.mongodb.net'
# uri = f'mongodb+srv://{username}:{password}@{cluster}/?retryWrites=true&w=majority&appName=Cluster0&ssl=true'
# client = motor.motor_asyncio.AsyncIOMotorClient(uri)
# db = client['AGENTSMTHS']
# collection = db['support_chats']
# group_collections = db["group_collections"]
# representative_questions_collection = db["representative_questions"]

# # class RephraseRequest(BaseModel):
# #     question: str
    
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

# # @app.get("/fetch_data")
# # async def fetch_all_data():
# #     documents = await collection.find().to_list(length=None)  # Fetch all documents from the collection
# #     return {"data": jsonable_encoder(documents)}


# @app.get("/fetch_user_messages")
# async def fetch_user_messages():
#     documents = await collection.find({"status": "Unread"}).to_list(length=None)
    
#     if not documents:
#         raise HTTPException(status_code=404, detail="No unread messages found")

#     user_messages = []
#     for doc in documents:
#         if 'messages' in doc:
#             user_messages.extend([msg for msg in doc['messages'] if msg.get('sender') == 'user'])

#     # Convert ObjectId to string before encoding
#     documents = [str(doc['_id']) for doc in documents]

#     return {"user_messages": jsonable_encoder(user_messages), "documents": documents}

# async def check_message_against_clusters(message, representative_questions, threshold=0.75):
#     message_embedding = encode_text_sbert(message)
#     for message in representative_questions:
#         rep_question_embedding = encode_text_sbert(message)
#         similarity = cosine_similarity([message_embedding], [rep_question_embedding])[0][0]
#         if similarity >= threshold:
#             return True
#     return False

# # Filter out unwanted messages like greetings and thank-yous
# def filter_unwanted_messages(messages):
#     unwanted_phrases = [
#         "hello", "hi", "hey", "greetings", "good morning", "good afternoon",
#         "good evening", "thank you", "thanks", "bye", "goodbye"
#     ]
    
#     filtered_messages = [
#         message for message in messages
#         if isinstance(message, dict) and 'message' in message and not any(phrase in message['message'].lower() for phrase in unwanted_phrases)
#     ]
    
#     return filtered_messages

# async def periodic_cluster_update():
#     while True:
#         await update_or_create_clusters()
#         await asyncio.sleep(10)

# @app.on_event("startup")
# async def startup_event():
#     asyncio.create_task(periodic_cluster_update())
    


# # @app.post("/update_or_create_clusters")
# async def update_or_create_clusters():
#     new_messages = await fetch_user_messages()
#     documents = await collection.find({"status": "Unread"}).to_list(length=None)
    
#     user_messages = []
#     document_id_map = {}

#     for document in documents:
#         doc_id = document['_id']
#         messages = document.get('messages', [])
#         for msg in messages:
#             if msg.get('sender') == 'user':
#                 user_messages.append(msg)
#                 document_id_map[msg['message']] = doc_id
    
#     filtered_messages = filter_unwanted_messages(user_messages)
#     existing_clusters_docs = await group_collections.find({"status": "Unread"}).to_list(length=None)
#     existing_clusters_docs = [message['messages'] for message in existing_clusters_docs]

#     if not existing_clusters_docs:
#         new_cluster = {
#             "created_date": datetime.utcnow().isoformat(),
#             "status": "Unread",
#             "messages": filtered_messages[0]["message"] # Ensure messages are in a list
#         }
#         insert_result = await group_collections.insert_one(new_cluster)
#         filtered_messages = filtered_messages[1:]

#     updated_clusters = []
#     for message in filtered_messages:
#         cluster_id = await check_message_against_clusters(message["message"], existing_clusters_docs)
#         if cluster_id:
#             continue
#         else:
#             new_cluster = {
#                 "created_date": datetime.utcnow().isoformat(),
#                 "status": "Unread",
#                 "messages": message["message"] # Ensure messages are in a list
#             }
#             insert_result = await group_collections.insert_one(new_cluster)
#             updated_clusters.append(insert_result.inserted_id)
#         existing_clusters_docs = await group_collections.find({"status": "Unread"}).to_list(length=None)
#         existing_clusters_docs = [message['messages'] for message in existing_clusters_docs]

#     for message in filtered_messages:
#         doc_id = document_id_map.get(message["message"])
#         # if doc_id:
#         #     await collection.update_one({"_id": doc_id}, {"$set": {"status": "read"}})
#         if doc_id:
#             await collection.update_one({"_id":convert_objectid_to_str(doc_id)}, {"$set": {"status": "read"}})
    
#     return {"updated_clusters": updated_clusters}
# def convert_objectid_to_str(doc):
#     if isinstance(doc, list):
#         return [convert_objectid_to_str(item) for item in doc]
#     elif isinstance(doc, dict):
#         return {k: (str(v) if isinstance(v, ObjectId) else convert_objectid_to_str(v)) for k, v in doc.items()}
#     return doc
# # Load SBERT model for sentence embeddings
# sbert_model = SentenceTransformer('all-mpnet-base-v2')

# # Function to encode text using SBERT model
# def encode_text_sbert(text):
#     return sbert_model.encode([text])[0]

# # Function to rephrase a question using the generative AI model
# def rephrase_message(question, model_gemini_pro):
#     prompt = f"Rephrase the following question as instruction to admin from system to check and solve the user's problem in one sentence more accurately: {question}"
#     response = model_gemini_pro.generate_content(prompt)
#     return response.text.strip()

# logger = logging.getLogger(__name__)

# @app.post("/rephrase_messages")
# async def rephrase_messages():
#     try:
#         documents = await collection.find({"status": "Unread"}).to_list(length=None)

#         if not documents:
#             logger.info("No unread documents found.")
#             return {"message": "No unread documents found."}

#         rephrased_messages = []
#         for document in documents:
#             doc_id = document['_id']
#             messages = document.get('messages', '')

#             if isinstance(messages, str):
#                 original_message = messages
#                 rephrased_message = rephrase_message(original_message, model_gemini_pro)
#                 await collection.update_one({"_id": ObjectId(doc_id)}, {"$set": {"messages": rephrased_message}})
#                 rephrased_messages.append({
#                     "document_id": str(doc_id),
#                     "original_message": original_message,
#                     "rephrased_message": rephrased_message
#                 })

#             elif isinstance(messages, list):
#                 for original_message in messages:
#                     rephrased_message = rephrase_message(original_message, model_gemini_pro)
#                     await collection.update_one(
#                         {"_id": ObjectId(doc_id), "messages": original_message},
#                         {"$set": {"messages.$": rephrased_message}}
#                     )
#                     rephrased_messages.append({
#                         "document_id": str(doc_id),
#                         "original_message": original_message,
#                         "rephrased_message": rephrased_message
#                     })

#         return {"rephrased_messages": rephrased_messages}

#     except Exception as e:
#         logger.error(f"Error in rephrase_messages: {e}")
#         return {"error": str(e)}

# # WebSocket endpoint
# @app.websocket("/notifications")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()

#     while True:
#         unread_clusters = await group_collections.find({"status": "Unread"}).to_list(length=None)

#         if not unread_clusters:
#             await websocket.send_json({"message": "No unread clusters found"})
#             continue

#         for cluster_document in unread_clusters:
#             cluster_id = str(cluster_document["_id"])
#             messages = cluster_document.get("messages", [])

#             if isinstance(messages, str):
#                 messages = [{"message": messages}]

#             rephrased_messages = []
#             for message in messages:
#                 if isinstance(message, dict) and "message" in message:
#                     rephrased_message = rephrase_message(message["message"], model_gemini_pro)
#                     rephrased_messages.append(rephrased_message)
#                 else:
#                     # Handle unexpected message format gracefully
#                     logger.warning(f"Unexpected message format: {message}")


#             for message in rephrased_messages:
#                 notification = {
#                     "question": message,
#                     "cluster_id": cluster_id
#                 }
#                 await websocket.send_json(notification)

#         data = await websocket.receive_json()
#         action = data.get("action")
#         cluster_id = data.get("cluster_id")

#         if action in ["Done", "Ignore"] and cluster_id:
#             update_query = {"_id": ObjectId(cluster_id)}
#             update_fields = {"$set": {"status": "read"}}
#             update_result = await group_collections.update_one(update_query, update_fields)

#             if update_result.modified_count == 1:
#                 await websocket.send_json({"message": f"Cluster {cluster_id} status updated"})
#             else:
#                 await websocket.send_json({"message": f"Failed to update status for cluster {cluster_id}"})
#         else:
#             await websocket.send_json({"message": "Invalid action or cluster ID"})

#     await websocket.close()

from configparser import ConfigParser
from motor.motor_asyncio import AsyncIOMotorClient
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket
from urllib.parse import quote_plus
from bson import ObjectId
from bson.json_util import dumps
import textwrap
from pydantic import BaseModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import asyncio
import logging
from datetime import datetime
import json 
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
client = AsyncIOMotorClient(uri)
db = client['AGENTSMTHS']
collection = db['support_chats']
group_collections = db["group_collections"]
representative_questions_collection = db["representative_questions"]

# # class RephraseRequest(BaseModel):
# #     question: str
    
# def to_markdown(text):
#     text = text.replace("•", "  *")
#     indented_text = textwrap.indent(text, "\n ", predicate=lambda _: True)
#     return indented_text


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

# @app.get("/fetch_data")
# async def fetch_all_data():
#     documents = await collection.find().to_list(None)  # Fetch all documents from the collection
#     # Convert ObjectId to string for JSON serialization
#     for doc in documents:
#         doc['_id'] = str(doc['_id'])  # Convert ObjectId to string
#     return {"data": json.loads(dumps(documents))}

@app.get("/fetch_user_messages")
async def fetch_user_messages():
    documents = await collection.find({"status": "Unread"}).to_list(None)
    
    if not documents:
        raise HTTPException(status_code=404, detail="No unread messages found")

    user_messages = []
    for doc in documents:
        if 'messages' in doc:
            user_messages.extend([msg for msg in doc['messages'] if msg.get('sender') == 'user'])

    for doc in documents:
        doc['_id'] = str(doc['_id'])

    return {"user_messages": json.loads(dumps(user_messages)), "documents": json.loads(dumps(documents))}

def check_message_against_clusters(message, representative_questions, threshold=0.75):
    message_embedding = encode_text_sbert(message)
    for rep_question in representative_questions:
        rep_question_embedding = encode_text_sbert(rep_question)
        similarity = cosine_similarity([message_embedding], [rep_question_embedding])[0][0]
        if similarity >= threshold:
            return True
    return False

# Filter out unwanted messages like greetings and thank-yous
def filter_unwanted_messages(messages):
    unwanted_phrases = [
        "hello", "hi", "hey", "greetings", "good morning", "good afternoon",
        "good evening", "thank you", "thanks", "bye", "goodbye"
    ]
    
    filtered_messages = [
        message for message in messages
        if isinstance(message, dict) and 'message' in message and not any(phrase in message['message'].lower() for phrase in unwanted_phrases)
    ]
    
    return filtered_messages
async def periodic_cluster_update():
    while True:
        # Background task to run update_or_create_clusters
        await update_or_create_clusters()

        # Adjust sleep time as needed (e.g., every 10 seconds)
        await asyncio.sleep(3)

# Start the periodic task when the application starts
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(periodic_cluster_update())
    
async def update_or_create_clusters():
    new_messages = await fetch_user_messages()
    documents = await collection.find({"status": "Unread"}).to_list(None)
    
    # Extract user messages and their parent document IDs from all documents
    user_messages = []
    document_id_map = {}  # To map message to its parent document ID
    
    for document in documents:
        doc_id = document['_id']
        messages = document.get('messages', [])
        for msg in messages:
            if msg.get('sender') == 'user':
                user_messages.append(msg)
                document_id_map[msg['message']] = doc_id
    
    # Filter out unwanted messages
    filtered_messages = filter_unwanted_messages(user_messages)
    existing_clusters_docs = await group_collections.find({"status": "Unread"}).to_list(None)
    
    existing_clusters_messages = [message['messages'] for message in existing_clusters_docs]
    
    updated_clusters = []
    for message in filtered_messages:
        cluster_id = check_message_against_clusters(message["message"], existing_clusters_messages)
        
        if cluster_id:
            continue
        else:
            new_cluster = {
                "created_date": datetime.utcnow().isoformat(),
                "status": "Unread",
                "messages": message["message"] 
            }
            insert_result = await group_collections.insert_one(new_cluster)
            updated_clusters.append(str(insert_result.inserted_id))
        existing_clusters_docs = await group_collections.find({"status": "Unread"}).to_list(None)
        existing_clusters_messages = [message['messages'] for message in existing_clusters_docs]
    
    for message in filtered_messages:
        doc_id = document_id_map.get(message["message"])
        if doc_id:
            await collection.update_one({"_id": doc_id}, {"$set": {"status": "read"}})
    
    return {"updated_clusters": updated_clusters}

# Load SBERT model for sentence embeddings
sbert_model = SentenceTransformer('all-mpnet-base-v2')

# Function to encode text using SBERT model
def encode_text_sbert(text):
    return sbert_model.encode([text])[0]

# Function to rephrase a question using the generative AI model
def rephrase_message(question, model_gemini_pro):
    prompt = f"Rephrase the following question as instruction to admin from system to check and solve the user's problem in one sentence more accurately: {question}"
    response = model_gemini_pro.generate_content(prompt)
    return response.text.strip()

# WebSocket endpoint remains unchanged
@app.websocket("/notifications")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    while True:
        unread_clusters = await group_collections.find({"status": "Unread"}).to_list(None)

        if not unread_clusters:
            await websocket.send_json({"message": "No unread clusters found"})
            continue

        for cluster_document in unread_clusters:
            cluster_id = str(cluster_document["_id"])
            messages = cluster_document.get("messages", [])

            if isinstance(messages, str):
                messages = [{"message": messages}]

            rephrased_messages = []
            for message in messages:
                rephrased_message = rephrase_message(message["message"], model_gemini_pro)
                rephrased_messages.append(rephrased_message)

            for message in rephrased_messages:
                notification = {
                    "question": message,
                    "cluster_id": cluster_id
                }
                await websocket.send_json(notification)

        data = await websocket.receive_json()
        action = data.get("action")
        cluster_id = data.get("cluster_id")

        if action in ["Done", "Ignore"] and cluster_id:
            update_query = {"_id": ObjectId(cluster_id)}
            update_fields = {"$set": {"status": "read"}}
            update_result = await group_collections.update_one(update_query, update_fields)

            if update_result.modified_count == 1:
                await websocket.send_json({"message": f"Cluster {cluster_id} status updated"})
            else:
                await websocket.send_json({"message": f"Failed to update status for cluster {cluster_id}"})
        else:
            await websocket.send_json({"message": "Invalid action or cluster ID"})

    await websocket.close()

