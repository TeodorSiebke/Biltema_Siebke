from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import chromadb
import os
import google.generativeai as genai
import json
import re
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from prompts import llm1_prompts, llm2_prompts
import random

# Load environment variables from .env file
load_dotenv()

# --- Utility Functions ---
def load_categories(file_path):
    """Loads categories and their URLs from a text file."""
    category_map = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if ',' in line:
                parts = line.split(',', 1)
                if len(parts) == 2 and 'https://' in parts[1]:
                    name = parts[0].strip()
                    url = parts[1].strip()
                    category_map[name] = url
    return category_map

def sanitize_llm_list_output(data):
    """Ensures the output from the LLM is a flat list of strings."""
    if not isinstance(data, list):
        return []
    
    sanitized_list = []
    for item in data:
        if isinstance(item, str):
            sanitized_list.append(item)
    return sanitized_list

# --- FastAPI App Initialization ---
app = FastAPI()

# Load categories on startup
category_file_path = os.path.join(os.path.dirname(__file__), "..", "kategorier.txt")
category_data = load_categories(category_file_path)
category_names = list(category_data.keys())

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Generative AI
api_key = os.getenv("GOOGLE_AI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    raise ValueError("GOOGLE_AI_API_KEY not found in .env file.")

keyword_model = genai.GenerativeModel('gemini-2.0-flash')
rag_model = genai.GenerativeModel('gemini-2.0-flash')

# Initialize ChromaDB
db_path = os.path.join(os.path.dirname(__file__), "..", "products_db_2")
try:
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection("product_embeddings")
except Exception as e:
    print(f"Error initializing ChromaDB: {e}")
    collection = None

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Welcome to SiebkeAI Search"}

@app.get("/api/search")
async def search_products(q: str, lang: str = 'en', prompt_llm1: int = 0, prompt_llm2: int = 0):

    async def event_stream():
        try:
            yield f"data: {json.dumps({'type': 'status', 'message': 'Analyzing your request...'})}\n\n"
            await asyncio.sleep(1)

            # --- Stage 1: Query Expansion & Category Recommendation ---
            yield f"data: {json.dumps({'type': 'status', 'message': 'Generating keywords and finding categories...'})}\n\n"
            await asyncio.sleep(1)
            
            if collection is None:
                raise ValueError("Database collection is not available.")

            llm1_prompt_template = llm1_prompts[prompt_llm1]
            category_list_str = "\n".join([f"- {name}" for name in category_names])
            keyword_prompt = llm1_prompt_template.format(query=q, categories=category_list_str)

            keyword_response = keyword_model.generate_content(keyword_prompt)
            response_text = keyword_response.text.strip()
            
            keywords = []
            recommended_category_names = []
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
            if json_match:
                json_str = json_match.group(1)
                try:
                    llm1_output = json.loads(json_str)
                    if isinstance(llm1_output, dict):
                        keywords = sanitize_llm_list_output(llm1_output.get("keywords", []))
                        recommended_category_names = sanitize_llm_list_output(llm1_output.get("categories", []))
                    else:
                        print(f"Warning: LLM1 output was not a dictionary: {llm1_output}")
                except Exception as e:
                    print(f"Error parsing LLM1 response: {e}. Raw response: {json_str}")

            if not keywords:
                keywords = [q]

            # Ensure recommended_categories is always a dictionary, even if empty
            recommended_categories = {name: category_data.get(name) for name in recommended_category_names if category_data.get(name)}

            print(f"Generated keywords: {keywords[:5]}")
            print(f"Recommended categories: {recommended_categories}")

            # --- Stage 2: Similarity Search ---
            yield f"data: {json.dumps({'type': 'status', 'message': 'Searching for relevant products...'})}\n\n"
            await asyncio.sleep(1)
            all_product_data = {}
            for keyword in keywords[:5]:
                embedding = genai.embed_content(
                    model="models/gemini-embedding-001",
                    content=keyword,
                    output_dimensionality=768,
                    task_type="retrieval_query"
                )['embedding']
                
                results = collection.query(query_embeddings=[embedding], n_results=5, include=['metadatas'])

                if results and results['ids'] and results['ids'][0]:
                    for i in range(len(results['ids'][0])):
                        product_id = results['ids'][0][i]
                        if product_id not in all_product_data:
                            metadata = results['metadatas'][0][i]
                            all_product_data[product_id] = {
                                'id': product_id,
                                'name': metadata.get('name'),
                                'price': metadata.get('price'),
                                'image': metadata.get('image'),
                                'short_description': metadata.get('short_description', '')
                            }
            product_data = list(all_product_data.values())
            print(f"Found {len(product_data)} unique products from keywords.")

            # --- Stage 3: Response Generation (RAG) ---
            yield f"data: {json.dumps({'type': 'status', 'message': 'Consulting with our AI assistant...'})}\n\n"
            await asyncio.sleep(1)
            product_paragraphs = [f"ID: {p['id']}\nName: {p['name']}\nPrice: {p['price']}\nDescription: {p.get('short_description', 'N/A')}" for p in product_data]
            product_data_for_llm = "\n\n".join(product_paragraphs)

            llm2_prompt_template = llm2_prompts[prompt_llm2]
            prompt = llm2_prompt_template.format(query=q, products=product_data_for_llm, lang=lang)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(os.path.join("..", "logs", f"{timestamp}_llm2_prompt.txt"), "w", encoding="utf-8") as f:
                f.write(prompt)

            print(f"Sending prompt to LLM2. Prompt length: {len(prompt)}")
            response = rag_model.generate_content(prompt, generation_config=genai.GenerationConfig(max_output_tokens=5000))
            print("Received response from LLM2.")
            raw_response_text = response.parts[0].text
            print(f"Raw LLM2 response text (first 500 chars): {raw_response_text[:500]}")

            with open(os.path.join("..", "logs", f"{timestamp}_llm2_response.txt"), "w", encoding="utf-8") as f:
                f.write(raw_response_text)

            recommended_ids = []
            llm_explanation = raw_response_text
            json_code_block_match = re.search(r'```json\s*([\s\S]*?)\s*```', raw_response_text)
            if json_code_block_match:
                json_str = json_code_block_match.group(1).strip()
                try:
                    parsed_ids = json.loads(json_str)
                    recommended_ids = sanitize_llm_list_output(parsed_ids)
                    llm_explanation = raw_response_text[:json_code_block_match.start()].strip()
                except Exception as e:
                    print(f"Error parsing LLM2 response: {e}. Raw response: {json_str}")

            product_data_map = {p['id']: p for p in product_data}
            ordered_products = [product_data_map[pid] for pid in recommended_ids if pid in product_data_map]

            if not ordered_products and product_data:
                ordered_products = product_data

            yield f"data: {json.dumps({'type': 'result', 'llm_response': llm_explanation, 'products': ordered_products, 'recommended_categories': recommended_categories})}\\n\n"

        except Exception as e:
            print(f"An error occurred during search: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
