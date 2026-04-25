from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
import nomic
import fitz  # PyMuPDF
from nomic import embed
from pinecone import Pinecone, Vector
import os
from dotenv import load_dotenv
import uuid
import requests
from bs4 import BeautifulSoup
import base64
import io
from pydantic import BaseModel
from PIL import Image
from urllib.parse import urlparse, parse_qs, urljoin
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from groq import Groq

load_dotenv()

# -------- CLIENTS --------
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

from google import genai # 

# Client initialize karein
client_gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
NOMIC_API_KEY = os.getenv("NOMIC_API_KEY")

if NOMIC_API_KEY:
    nomic.login(NOMIC_API_KEY)
else:
    print("Warning: NOMIC_API_KEY not found!")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("context-hub")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- NAMESPACES --------
PDF_NAMESPACE = "pdf"  
current_pdf_namespace = "pdf_default"
IMAGE_NAMESPACE = "image"
YOUTUBE_NAMESPACE = "youtube"

# -------- HELPER FUNCTIONS --------

def chunk_text(text: str, size: int = 500):
    return [text[i:i+size] for i in range(0, len(text), size)]


def create_embeddings(chunks: list):
    if not chunks:
        return []
    try:
        response = embed.text(
            texts=chunks,
            model="nomic-embed-text-v1",
            task_type="search_document"
        )
        result = dict(response)
        if 'embeddings' in result and len(result['embeddings']) > 0:
            return result['embeddings']
        else:
            print("DEBUG: Nomic returned no embeddings!")
            return []
    except Exception as e:
        print(f"Embedding error: {e}")
        return []


def store_embeddings(chunks: list, embeddings: list, namespace: str = ""):
    if not chunks or not embeddings:
        print("DEBUG: Nothing to store.")
        return
    vectors = [
        Vector(
            id=str(uuid.uuid4()),
            values=embeddings[i],
            metadata={"text": chunks[i]}
        )
        for i in range(len(embeddings))
    ]
    if vectors:
        index.upsert(vectors=vectors, namespace=namespace)   # ← namespace added


def search(query: str, namespace: str = ""):
    try:
        query_response = embed.text(
            texts=[query],
            model="nomic-embed-text-v1",
            task_type="search_query"
        )
        result = dict(query_response)
        query_emb = result['embeddings'][0]

        results = index.query(
            vector=query_emb,
            top_k=3,
            include_metadata=True,
            namespace=namespace    
        )
        relevant_chunks = []
        for match in results['matches']:
            if match.get('metadata') and 'text' in match['metadata']:
                relevant_chunks.append(match['metadata']['text'])
        return relevant_chunks

    except Exception as e:
        print(f"Search error: {e}")
        return []


def generate_answer(query: str, context: str):
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Answer ONLY using the provided context. "
                        "If the answer is not in the context, say you don't know. "
                        "When you find an answer, elaborate and provide more information."
                    )
                },
                {
                    "role": "user",
                    "content": f"Context: {context}\n\nQuestion: {query}"
                }
            ]
        )
        return response.choices[0].message.content

    except Exception as e:
        print(f"Groq Error: {str(e)}")
        return f"Groq Error: {str(e)}"

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global current_pdf_namespace
    try:
        # Generate a fresh unique namespace for every new upload
        current_pdf_namespace = f"pdf_{uuid.uuid4().hex}"
        print(f"DEBUG: Using new namespace: {current_pdf_namespace}")

        content = await file.read()
        pdf = fitz.open(stream=content, filetype="pdf")

        text = ""
        for page in pdf:
            text += page.get_text()

        if not text.strip():
            return {"error": "PDF is empty or could not be read."}

        chunks = chunk_text(text)
        embeddings = create_embeddings(chunks)
        store_embeddings(chunks, embeddings, namespace=current_pdf_namespace)
        return {"message": "PDF processed successfully."}

    except Exception as e:
        print(f"Upload Error: {e}")
        return {"error": f"Internal Server Error: {str(e)}"}
    
    
@app.post("/query")
async def query_pdf(query: str = Query(...)):
    global current_pdf_namespace
    try:
        results = search(query, namespace=current_pdf_namespace)
        if not results:
            return {"answer": "I couldn't find any relevant information in the document."}
        context = " ".join(results)
        answer = generate_answer(query, context)
        return {"answer": answer}
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}
    
# -------- IMAGE HELPERS --------

def create_image_embedding(image):
    output = embed.image(
        images=[image],
        model="nomic-embed-vision-v1.5"
    )
    result = dict(output)
    return result["embeddings"][0]


def store_image_embedding(image_vector, filename: str):
    index.upsert(
        vectors=[Vector(
            id=str(uuid.uuid4()),
            values=image_vector,
            metadata={"type": "image", "filename": filename}
        )],
        namespace=IMAGE_NAMESPACE
    )

def generate_image_answer(query, image):
    try:
        # Convert image → bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_bytes = img_byte_arr.getvalue()

        # Convert to base64
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        response = client_gemini.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {"text": query},
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": img_base64
                            }
                        }
                    ]
                }
            ]
        )

        return response.text if response.text else "No answer generated."

    except Exception as e:
        return f"Error: {str(e)}"
    


# -------- URL HELPERS --------

def normalize_url(url: str):
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


def extract_text_from_url(url: str):
    try:
        url = normalize_url(url)   #  FIX ADDED

        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove unwanted tags
        for tag in soup(["script", "style"]):
            tag.decompose()

        text = soup.get_text(separator=" ")

        return text[:5000]

    except Exception as e:
        print(f"URL Error: {e}")
        return None


def extract_about_contact(base_url: str):
    """Optional enhancement: fetch About & Contact pages"""
    try:
        base_url = normalize_url(base_url)

        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(base_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        links = [a.get("href") for a in soup.find_all("a", href=True)]

        about_url = None
        contact_url = None

        for link in links:
            full_link = urljoin(base_url, link)

            if "about" in link.lower() and not about_url:
                about_url = full_link

            if "contact" in link.lower() and not contact_url:
                contact_url = full_link

        content = ""

        if about_url:
            content += extract_text_from_url(about_url) or ""

        if contact_url:
            content += extract_text_from_url(contact_url) or ""

        return content[:5000]

    except:
        return ""


def generate_url_answer(text: str, url: str, query: str = None):
    try:
        user_msg = (
            f"URL: {url}\n\n"
            f"Content:\n{text[:5000]}\n\n"
            f"Task: Explain what this page is about in simple words."
        )

        if query:
            user_msg += f"\n\nUser Question: {query}"

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions about webpage content."
                },
                {
                    "role": "user",
                    "content": user_msg
                }
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"


current_url = None
current_url_text = None

@app.post("/process-url")
async def process_url(url: str = Query(...)):
    global current_url, current_url_text

    url = normalize_url(url)

    main_text = extract_text_from_url(url)
    extra_text = extract_about_contact(url)

    combined_text = (main_text or "") + "\n\n" + (extra_text or "")

    if not combined_text.strip():
        return {"error": "Failed to extract content"}

    current_url = url
    current_url_text = combined_text   #   store everything once

    return {"message": "URL processed successfully"}


@app.post("/query-url")
async def query_url(query: str = Query(...)):
    global current_url, current_url_text

    if not current_url_text:
        return {"error": "No URL processed"}

    try:
        answer = generate_url_answer(current_url_text, current_url, query)
        return {"answer": answer}

    except Exception as e:
        return {"error": str(e)}


# -------- YOUTUBE HELPERS --------

def get_video_id(url: str):
    parsed_url = urlparse(url)
    if "youtube.com" in url:
        return parse_qs(parsed_url.query).get("v", [None])[0]
    elif "youtu.be" in url:
        return parsed_url.path.strip("/")
    return None


def get_transcript(url: str):
    try:
        video_id = get_video_id(url)
        if not video_id:
            return "ERROR: Invalid YouTube URL"

        ytt_api = YouTubeTranscriptApi()

        try:
            transcript_data = ytt_api.fetch(video_id, languages=['en'])
        except NoTranscriptFound:
            transcript_list = ytt_api.list(video_id)
            available = list(transcript_list)
            if not available:
                return "ERROR: No transcripts available for this video"
            transcript_data = available[0].fetch()

        text = " ".join([str(t.text) for t in transcript_data])
        return text

    except TranscriptsDisabled:
        return "ERROR: Transcripts are disabled for this video"
    except Exception as e:
        return f"ERROR: {str(e)}"


# -------- ROUTES --------

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        try:
            index.delete(delete_all=True, namespace="")
            print("DEBUG: Pinecone default namespace cleared.")
        except Exception as e:
            print(f"DEBUG: Delete skipped: {e}")

        content = await file.read()
        pdf = fitz.open(stream=content, filetype="pdf")

        text = ""
        for page in pdf:
            text += page.get_text()

        if not text.strip():
            return {"error": "PDF is empty or could not be read."}

        chunks = chunk_text(text)
        embeddings = create_embeddings(chunks)
        store_embeddings(chunks, embeddings)
        return {"message": "PDF processed successfully."}

    except Exception as e:
        print(f"Upload Error: {e}")
        return {"error": f"Internal Server Error: {str(e)}"}


@app.post("/query")
async def query_pdf(query: str = Query(...)):
    try:
        results = search(query)
        if not results:
            return {"answer": "I couldn't find any relevant information in the document."}
        context = " ".join(results)
        answer = generate_answer(query, context)
        return {"answer": answer}
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}


current_image = None

@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    global current_image
    try:
        image = Image.open(file.file)
        current_image = image
        image_vector = create_image_embedding(image)
        store_image_embedding(image_vector, file.filename)
        return {"message": "Image processed successfully"}
    except Exception as e:
        return {"error": str(e)}


@app.post("/query-image")
async def query_image(query: str = Query(...)):
    global current_image
    try:
        if current_image is None:
            return {"error": "No image uploaded"}
        answer = generate_image_answer(query, current_image)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}


current_url = None
current_url_text = None

@app.post("/process-url")
async def process_url(url: str = Query(...)):
    global current_url, current_url_text
    text = extract_text_from_url(url)
    if not text:
        return {"error": "Failed to extract content"}
    current_url = url
    current_url_text = text
    return {"message": "URL processed successfully"}


@app.post("/query-url")
async def query_url(query: str = Query(...)):
    global current_url, current_url_text
    if not current_url_text:
        return {"error": "No URL processed"}
    try:
        answer = generate_url_answer(current_url, query)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}
# ####################################################################################

class YouTubeRequest(BaseModel):
    url: str

class YouTubeQueryRequest(BaseModel):
    query: str
    video_id: str

@app.post("/process-youtube")
async def process_youtube(request: YouTubeRequest):
    try:
        url = request.url  # ✅ now correctly reads from JSON body

        try:
            index.delete(delete_all=True, namespace=YOUTUBE_NAMESPACE)
        except Exception:
            pass

        text = get_transcript(url)
        if text.startswith("ERROR"):
            return {"error": text}

        chunks = chunk_text(text, size=1000)
        embeddings = create_embeddings(chunks)

        if not embeddings:
            return {"error": "Failed to create embeddings"}

        vectors = [
            Vector(
                id=str(uuid.uuid4()),
                values=embeddings[i],
                metadata={"text": chunks[i]}
            )
            for i in range(len(embeddings))
        ]

        index.upsert(vectors=vectors, namespace=YOUTUBE_NAMESPACE)
        return {"message": f"YouTube video processed successfully ({len(chunks)} chunks)"}

    except Exception as e:
        return {"error": str(e)}


@app.post("/query-youtube")
async def query_youtube(request: YouTubeQueryRequest):
    try:
        query = request.query      # ✅ from JSON body
        video_id = request.video_id  # ✅ available if needed for filtering

        query_response = embed.text(
            texts=[query],
            model="nomic-embed-text-v1",
            task_type="search_query"
        )
        query_emb = dict(query_response)["embeddings"][0]

        results = index.query(
            vector=query_emb,
            top_k=5,
            include_metadata=True,
            namespace=YOUTUBE_NAMESPACE
        )

        context = "\n\n".join([
            match["metadata"]["text"]
            for match in results["matches"]
        ])

        if not context.strip():
            return {"answer": "I couldn't find relevant information in this video."}

        answer = generate_answer(query, context)
        return {"answer": answer}

    except Exception as e:
        return {"error": str(e)}