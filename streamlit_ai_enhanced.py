"""
AI-Enhanced Video Analysis with OpenAI & Vector Database
Features: Frame chunking, semantic search, GPT-powered queries
"""

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import cv2
import numpy as np
from collections import deque
import time
from datetime import datetime
import json

# AI & Vector DB imports
from openai import OpenAI
import chromadb
from chromadb.config import Settings

# Try to import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="AI Video Analysis",
    page_icon="🎥",
    layout="wide"
)

# Global state for async video processing (avoids session_state issues)
import threading

class VideoState:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame_chunks = deque(maxlen=100)
        self.chunk_id = 0
        self.object_count = 0
        self.detected_objects = []
        self.pending_chunks = []

# Create global video state
@st.cache_resource
def get_video_state():
    return VideoState()

video_state = get_video_state()

# Initialize session state
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ''
if 'openai_client' not in st.session_state:
    st.session_state.openai_client = None
if 'chroma_client' not in st.session_state:
    st.session_state.chroma_client = None
if 'video_collection' not in st.session_state:
    st.session_state.video_collection = None
if 'event_log' not in st.session_state:
    st.session_state.event_log = deque(maxlen=50)

# Initialize Vector Database
@st.cache_resource
def init_vector_db():
    """Initialize ChromaDB for storing video events"""
    try:
        client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            allow_reset=True
        ))

        # Create or get collection for video events
        collection = client.get_or_create_collection(
            name="video_events",
            metadata={"hnsw:space": "cosine"}
        )

        return client, collection
    except Exception as e:
        st.error(f"Failed to initialize vector DB: {e}")
        return None, None

# Load YOLO model
@st.cache_resource
def load_yolo():
    if YOLO_AVAILABLE:
        return YOLO('yolov8n.pt')
    return None

model = load_yolo()

# Initialize OpenAI client
def init_openai(api_key):
    """Initialize OpenAI client"""
    try:
        client = OpenAI(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"Failed to initialize OpenAI: {e}")
        return None

# Get OpenAI embeddings
def get_embedding(text, client):
    """Get embeddings from OpenAI"""
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return None

# Color detection helper
def get_dominant_color(image_region):
    """Get dominant color from image region"""
    if image_region.size == 0:
        return "unknown"

    # Convert to HSV
    hsv = cv2.cvtColor(image_region, cv2.COLOR_BGR2HSV)
    h = np.mean(hsv[:, :, 0])
    s = np.mean(hsv[:, :, 1])
    v = np.mean(hsv[:, :, 2])

    # Low saturation = gray/white/black
    if s < 40:
        if v < 50:
            return "black"
        elif v > 200:
            return "white"
        else:
            return "gray"

    # Determine color based on hue
    if h < 10 or h > 160:
        return "red"
    elif h < 25:
        return "orange"
    elif h < 35:
        return "yellow"
    elif h < 85:
        return "green"
    elif h < 125:
        return "blue"
    elif h < 155:
        return "purple"
    else:
        return "pink"

# Video frame callback with chunking
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    """Process each video frame with chunking and vector storage"""
    try:
        # Convert frame to numpy array
        img = frame.to_ndarray(format="bgr24")

        if model:
            # Run YOLO detection
            results = model(img, conf=0.4, verbose=False)

            detected_objects = []
            events_text = []

            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = box.conf[0].item()
                        cls = int(box.cls[0].item())
                        label = model.names[cls]

                        # Get color of object (wrapped in try-catch)
                        try:
                            roi = img[y1:y2, x1:x2]
                            color = get_dominant_color(roi)
                        except:
                            color = "unknown"

                        detected_objects.append({
                            'label': label,
                            'color': color,
                            'confidence': conf,
                            'bbox': (x1, y1, x2, y2)
                        })

                        events_text.append(f"{color} {label}")

                        # Draw bounding box
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Draw label
                        text = f"{color} {label} {conf:.2f}"
                        cv2.putText(img, text, (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Update global video state (thread-safe)
            with video_state.lock:
                video_state.object_count = len(detected_objects)
                video_state.detected_objects = detected_objects

                # CHUNKING: Create chunks every 30 frames (~1 second at 30fps)
                if video_state.chunk_id % 30 == 0 and events_text:
                    chunk_description = f"At {datetime.now().strftime('%H:%M:%S')}: Detected {', '.join(events_text)}"

                    # Store chunk in memory
                    video_state.frame_chunks.append({
                        'id': video_state.chunk_id,
                        'timestamp': time.time(),
                        'description': chunk_description,
                        'objects': detected_objects.copy()
                    })

                    # Queue chunk for embedding (don't block video with API calls)
                    video_state.pending_chunks.append({
                        'id': video_state.chunk_id,
                        'description': chunk_description,
                        'timestamp': time.time(),
                        'object_count': len(detected_objects)
                    })

                video_state.chunk_id += 1

                # Add stats overlay
                chunk_count = len(video_state.frame_chunks)

            cv2.putText(img, f"Objects: {len(detected_objects)} | Chunks: {chunk_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    except Exception as e:
        # If anything fails, return original frame
        return frame

# Process pending chunks (embed and store)
def process_pending_chunks():
    """Process chunks waiting to be embedded - called before queries"""
    with video_state.lock:
        if not video_state.pending_chunks or not st.session_state.video_collection:
            return 0

        chunks_to_process = video_state.pending_chunks[:5]  # Process max 5 at a time

    processed = 0
    for chunk in chunks_to_process:
        try:
            embedding = get_embedding(chunk['description'], st.session_state.openai_client)
            if embedding:
                st.session_state.video_collection.add(
                    documents=[chunk['description']],
                    embeddings=[embedding],
                    ids=[f"chunk_{chunk['id']}"],
                    metadatas=[{
                        'timestamp': chunk['timestamp'],
                        'object_count': chunk['object_count']
                    }]
                )
                with video_state.lock:
                    video_state.pending_chunks.remove(chunk)
                processed += 1
        except Exception as e:
            st.session_state.event_log.append(f"✗ Embed error: {str(e)[:30]}")
            break

    return processed

# Query with GPT and vector search
def query_with_ai(question):
    """Answer questions using GPT with context from vector database"""
    if not st.session_state.openai_client:
        return "Please enter your OpenAI API key first."

    try:
        # First, process any pending chunks
        with video_state.lock:
            has_pending = len(video_state.pending_chunks) > 0

        if has_pending:
            processed = process_pending_chunks()
            if processed > 0:
                st.session_state.event_log.append(f"✓ Embedded {processed} chunks")

        # Get relevant context from vector database
        context_docs = []
        if st.session_state.video_collection:
            # Get embedding for question
            question_embedding = get_embedding(question, st.session_state.openai_client)
            if question_embedding:
                # Search vector database
                results = st.session_state.video_collection.query(
                    query_embeddings=[question_embedding],
                    n_results=5
                )
                if results and results['documents']:
                    context_docs = results['documents'][0]

        # Build context
        context = "\n".join(context_docs) if context_docs else "No video events stored yet."

        # Build current state
        with video_state.lock:
            current_objects = video_state.detected_objects.copy()

        if current_objects:
            obj_descriptions = [f"{o['color']} {o['label']}" for o in current_objects]
            current_state = f"Currently visible: {', '.join(obj_descriptions)}"
        else:
            current_state = "No objects currently visible"

        # Create prompt
        prompt = f"""You are a video analysis assistant. Answer the question based on the video footage context.

Video Event History (from vector database):
{context}

Current Frame:
{current_state}

Question: {question}

Provide a concise, helpful answer based on the video data."""

        # Call GPT
        response = st.session_state.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful video analysis assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error querying AI: {str(e)}"

# Main UI
st.title("🎥 AI-Enhanced Video Analysis")
st.caption("With OpenAI GPT, Frame Chunking & Vector Database")

# Sidebar - API Key and Configuration
with st.sidebar:
    st.header("🔑 Configuration")

    # OpenAI API Key input
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.openai_api_key,
        help="Enter your OpenAI API key to enable AI-powered queries"
    )

    if api_key and api_key != st.session_state.openai_api_key:
        st.session_state.openai_api_key = api_key
        st.session_state.openai_client = init_openai(api_key)
        if st.session_state.openai_client:
            st.success("✓ OpenAI connected!")
            # Initialize vector DB
            client, collection = init_vector_db()
            st.session_state.chroma_client = client
            st.session_state.video_collection = collection
            if collection:
                st.success("✓ Vector DB initialized!")

    st.divider()

    # System Status
    st.header("📊 System Status")
    with video_state.lock:
        st.metric("Chunks Stored", len(video_state.frame_chunks))
        st.metric("Current Objects", video_state.object_count)
        st.metric("Pending Embeddings", len(video_state.pending_chunks))

    if st.session_state.video_collection:
        try:
            count = st.session_state.video_collection.count()
            st.metric("Vector DB Entries", count)
        except:
            st.metric("Vector DB Entries", "Error")

    st.divider()

    # Event Log
    st.header("📝 Event Log")
    if st.session_state.event_log:
        for event in list(st.session_state.event_log)[-10:]:
            st.caption(event)
    else:
        st.caption("No events yet")

if not YOLO_AVAILABLE:
    st.error("YOLO not available. Install with: pip install ultralytics")
    st.stop()

# Main Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Live Video Feed")

    if not st.session_state.openai_api_key:
        st.warning("⚠️ Enter your OpenAI API key in the sidebar to enable AI features!")

    # WebRTC streamer
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    webrtc_ctx = webrtc_streamer(
        key="ai-video-analysis",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    st.info("📹 Click 'START' to begin. Video frames are chunked every ~1 second and stored in the vector database!")

with col2:
    st.header("🤖 AI Query Interface")

    # Query input
    query = st.text_area(
        "Ask anything about the video:",
        placeholder="e.g., What objects appeared in the last 30 seconds?\nWhen did you see a red cup?\nDescribe what happened recently.",
        height=100
    )

    if st.button("🔍 Ask AI", type="primary"):
        if query:
            if not st.session_state.openai_api_key:
                st.error("Please enter your OpenAI API key in the sidebar first!")
            else:
                with st.spinner("AI is analyzing..."):
                    answer = query_with_ai(query)
                    st.success("**AI Answer:**")
                    st.write(answer)
        else:
            st.warning("Please enter a question")

    st.divider()

    # Current detections
    st.subheader("📦 Current Frame Objects")
    with video_state.lock:
        current_detections = video_state.detected_objects.copy()

    if current_detections:
        for i, obj in enumerate(current_detections):
            st.write(f"{i+1}. **{obj['color']} {obj['label']}** ({obj['confidence']:.2f})")
    else:
        st.caption("No objects detected")

    st.divider()

    # Recent chunks
    st.subheader("🎬 Recent Video Chunks")
    with video_state.lock:
        recent_chunks = list(video_state.frame_chunks)[-5:]

    if recent_chunks:
        for chunk in recent_chunks:
            st.caption(f"[{chunk['id']}] {chunk['description']}")
    else:
        st.caption("No chunks yet - start the video!")

# Footer info
with st.expander("ℹ️ How This Works"):
    st.markdown("""
    ### 🎯 Features Implemented:

    **1. Frame Chunking:**
    - Video frames are grouped into 1-second chunks (30 frames)
    - Each chunk contains detection data and timestamp
    - Chunks are stored in memory (last 100) and vector database

    **2. Vector Database (ChromaDB):**
    - Stores semantic embeddings of video events
    - Enables similarity search across video history
    - Persistent storage of all detection events

    **3. OpenAI Integration:**
    - GPT-4o-mini for intelligent query answering
    - text-embedding-3-small for semantic search
    - Context-aware responses based on video history

    **4. AI-Powered Queries:**
    - Ask temporal questions ("what happened in the last minute?")
    - Semantic search ("when did you see a red object?")
    - Object tracking queries ("how many cups appeared?")

    ### 🔧 Technical Stack:
    - **YOLOv8**: Real-time object detection
    - **OpenAI GPT**: Natural language understanding
    - **ChromaDB**: Vector similarity search
    - **Streamlit WebRTC**: Smooth video streaming
    """)
