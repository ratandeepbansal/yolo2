"""
AI-Enhanced Video Analysis with Gradio Live Video
Features: Real-time YOLO detection, GPT queries, Vector DB storage
Optimized for Hugging Face Spaces deployment
"""

import gradio as gr
import cv2
import numpy as np
from collections import deque
import time
from datetime import datetime
import json
import os
from threading import Lock

# Ensure Ultralytics writes settings/cache inside the project workspace
ULTRALYTICS_BASE = os.path.join(os.path.dirname(__file__), ".ultralytics")
os.environ.setdefault("ULTRALYTICS_SETTINGS_DIR", ULTRALYTICS_BASE)
os.environ.setdefault("ULTRALYTICS_CACHE_DIR", os.path.join(ULTRALYTICS_BASE, "cache"))
os.makedirs(os.environ["ULTRALYTICS_SETTINGS_DIR"], exist_ok=True)
os.makedirs(os.environ["ULTRALYTICS_CACHE_DIR"], exist_ok=True)

# AI & Vector DB imports
from openai import OpenAI
import chromadb
from chromadb.config import Settings

# YOLO import
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Global state management
class VideoAnalysisState:
    def __init__(self):
        self.lock = Lock()
        self.frame_chunks = deque(maxlen=100)
        self.chunk_id = 0
        self.detected_objects = []
        self.pending_chunks = []
        self.event_log = deque(maxlen=50)
        self.openai_client = None
        self.chroma_client = None
        self.video_collection = None
        self.model = None
        self.frames_processed = 0
        self.frames_processed = 0

    def init_openai(self, api_key):
        """Initialize OpenAI client"""
        if not api_key:
            return False
        try:
            self.openai_client = OpenAI(api_key=api_key)
            # Test the connection
            self.openai_client.models.list()
            return True
        except Exception as e:
            self.event_log.append(f"❌ OpenAI error: {str(e)[:50]}")
            return False

    def init_vector_db(self):
        """Initialize ChromaDB"""
        try:
            self.chroma_client = chromadb.Client(Settings(
                anonymized_telemetry=False,
                allow_reset=True
            ))
            self.video_collection = self.chroma_client.get_or_create_collection(
                name="video_events",
                metadata={"hnsw:space": "cosine"}
            )
            return True
        except Exception as e:
            self.event_log.append(f"❌ Vector DB error: {str(e)[:50]}")
            return False

    def init_yolo(self):
        """Initialize YOLO model"""
        if YOLO_AVAILABLE and self.model is None:
            try:
                self.model = YOLO('yolov8n.pt')
                self.event_log.append("✓ YOLO model loaded")
                return True
            except Exception as e:
                self.event_log.append(f"❌ YOLO error: {str(e)[:50]}")
                return False
        return self.model is not None

# Global state
state = VideoAnalysisState()

def get_dominant_color(image_region):
    """Get dominant color from image region"""
    if image_region.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(image_region, cv2.COLOR_BGR2HSV)
    h = np.mean(hsv[:, :, 0])
    s = np.mean(hsv[:, :, 1])
    v = np.mean(hsv[:, :, 2])

    if s < 40:
        if v < 50:
            return "black"
        elif v > 200:
            return "white"
        else:
            return "gray"

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

def process_frame(frame):
    """Process video frame with YOLO detection"""
    if frame is None:
        return gr.update(value=None, visible=False)

    if state.model is None:
        return gr.update(value=frame, visible=True)

    # Convert incoming RGB frame to BGR for OpenCV/YOLO processing
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    try:
        # Run YOLO detection
        results = state.model(frame_bgr, conf=0.4, verbose=False)

        detected_objects = []
        events_text = []

        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    label = state.model.names[cls]

                    # Get color
                    try:
                        roi = frame_bgr[y1:y2, x1:x2]
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
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Draw label
                    text = f"{color} {label} {conf:.2f}"
                    cv2.putText(frame_bgr, text, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Update state (thread-safe)
        with state.lock:
            state.detected_objects = detected_objects
            state.frames_processed += 1

            # Create chunks every 30 frames
            if state.chunk_id % 30 == 0 and events_text:
                chunk_description = f"At {datetime.now().strftime('%H:%M:%S')}: Detected {', '.join(events_text)}"

                state.frame_chunks.append({
                    'id': state.chunk_id,
                    'timestamp': time.time(),
                    'description': chunk_description,
                    'objects': detected_objects.copy()
                })

                state.pending_chunks.append({
                    'id': state.chunk_id,
                    'description': chunk_description,
                    'timestamp': time.time(),
                    'object_count': len(detected_objects)
                })

            state.chunk_id += 1
            chunk_count = len(state.frame_chunks)

        # Add stats overlay
        cv2.putText(frame_bgr, f"Objects: {len(detected_objects)} | Chunks: {chunk_count}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Convert back to RGB for display in Gradio
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return gr.update(value=frame_rgb, visible=True)

    except Exception as e:
        state.event_log.append(f"❌ Frame error: {str(e)[:50]}")
        return gr.update(value=frame, visible=True)

def get_embedding(text):
    """Get embeddings from OpenAI"""
    if not state.openai_client:
        return None
    try:
        response = state.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        state.event_log.append(f"❌ Embedding error: {str(e)[:50]}")
        return None

def process_pending_chunks():
    """Process chunks waiting to be embedded"""
    with state.lock:
        if not state.pending_chunks or not state.video_collection:
            return 0
        chunks_to_process = state.pending_chunks[:5]

    processed = 0
    for chunk in chunks_to_process:
        try:
            embedding = get_embedding(chunk['description'])
            if embedding:
                state.video_collection.add(
                    documents=[chunk['description']],
                    embeddings=[embedding],
                    ids=[f"chunk_{chunk['id']}"],
                    metadatas=[{
                        'timestamp': chunk['timestamp'],
                        'object_count': chunk['object_count']
                    }]
                )
                with state.lock:
                    state.pending_chunks.remove(chunk)
                processed += 1
        except Exception as e:
            state.event_log.append(f"❌ Embed error: {str(e)[:30]}")
            break

    return processed

def query_with_ai(question):
    """Answer questions using GPT with vector database context"""
    if not state.openai_client:
        return "⚠️ Please enter your OpenAI API key first."

    if not question or not question.strip():
        return "⚠️ Please enter a question."

    try:
        # Process pending chunks
        with state.lock:
            has_pending = len(state.pending_chunks) > 0

        if has_pending:
            processed = process_pending_chunks()
            if processed > 0:
                state.event_log.append(f"✓ Embedded {processed} chunks")

        # Get context from vector DB
        context_docs = []
        if state.video_collection:
            question_embedding = get_embedding(question)
            if question_embedding:
                results = state.video_collection.query(
                    query_embeddings=[question_embedding],
                    n_results=5
                )
                if results and results['documents']:
                    context_docs = results['documents'][0]

        context = "\n".join(context_docs) if context_docs else "No video events stored yet."

        # Get current state
        with state.lock:
            current_objects = state.detected_objects.copy()
            frames_seen = state.frames_processed

        if current_objects:
            obj_descriptions = [f"{o['color']} {o['label']}" for o in current_objects]
            current_state = f"Currently visible: {', '.join(obj_descriptions)}"
        else:
            if frames_seen > 0:
                current_state = "Video stream active but no objects detected in the latest frame."
            else:
                current_state = "No video frames processed yet."

        if not context_docs and frames_seen > 0:
            context = "Video stream active, waiting for notable detections to log."

        # Create prompt
        prompt = f"""You are a video analysis assistant. Answer the question based on the video footage context.

Video Event History (from vector database):
{context}

Current Frame:
{current_state}

Question: {question}

Provide a concise, helpful answer based on the video data."""

        # Call GPT
        response = state.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful video analysis assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )

        answer = response.choices[0].message.content
        state.event_log.append(f"✓ Query answered")
        return f"**AI Answer:**\n\n{answer}"

    except Exception as e:
        error_msg = f"Error querying AI: {str(e)}"
        state.event_log.append(f"❌ Query error: {str(e)[:30]}")
        return error_msg

def setup_api_key(api_key):
    """Setup OpenAI API key and initialize services"""
    if not api_key or not api_key.strip():
        return "⚠️ Please enter a valid API key", get_stats()

    success = state.init_openai(api_key)
    if success:
        state.init_vector_db()
        state.init_yolo()
        return "✅ OpenAI connected! Vector DB initialized!", get_stats()
    else:
        return "❌ Failed to connect to OpenAI. Check your API key.", get_stats()

def get_stats():
    """Get current system statistics"""
    with state.lock:
        chunks = len(state.frame_chunks)
        objects = len(state.detected_objects)
        pending = len(state.pending_chunks)

    vector_count = 0
    if state.video_collection:
        try:
            vector_count = state.video_collection.count()
        except:
            vector_count = 0

    stats = f"""**System Status:**
- Chunks Stored: {chunks}
- Current Objects: {objects}
- Pending Embeddings: {pending}
- Vector DB Entries: {vector_count}
"""
    return stats

def get_current_detections():
    """Get list of currently detected objects"""
    with state.lock:
        current = state.detected_objects.copy()

    if not current:
        return "No objects detected"

    output = "**Current Detections:**\n\n"
    for i, obj in enumerate(current):
        output += f"{i+1}. {obj['color']} {obj['label']} ({obj['confidence']:.2f})\n"

    return output

def get_recent_chunks():
    """Get recent video chunks"""
    with state.lock:
        recent = list(state.frame_chunks)[-5:]

    if not recent:
        return "No chunks yet - start the video!"

    output = "**Recent Video Chunks:**\n\n"
    for chunk in recent:
        output += f"[{chunk['id']}] {chunk['description']}\n\n"

    return output

def get_event_log():
    """Get recent event log"""
    with state.lock:
        events = list(state.event_log)[-10:]

    if not events:
        return "No events yet"

    return "\n".join(events)

# Initialize YOLO on startup
state.init_yolo()

# Build Gradio interface
with gr.Blocks(title="AI Video Analysis", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎥 AI-Enhanced Video Analysis")
    gr.Markdown("*Real-time object detection with GPT queries and vector database storage*")

    with gr.Row():
        # Left column - Video and controls
        with gr.Column(scale=2):
            gr.Markdown("## 📹 Live Video Feed")

            # API Key setup
            with gr.Row():
                api_key_input = gr.Textbox(
                    label="OpenAI API Key",
                    type="password",
                    placeholder="sk-...",
                    scale=3
                )
                setup_btn = gr.Button("Connect", scale=1, variant="primary")

            api_status = gr.Markdown("⚠️ Enter your OpenAI API key to enable AI features")

            # Live Video Stream
            if YOLO_AVAILABLE:
                processed_feed = gr.Image(
                    label="YOLO Detection Feed",
                    interactive=False,
                    type="numpy",
                    visible=False
                )
                webcam_stream = gr.Image(
                    label="Webcam Stream",
                    sources=["webcam"],
                    streaming=True,
                    type="numpy"
                )
                webcam_stream.stream(
                    fn=process_frame,
                    inputs=webcam_stream,
                    outputs=processed_feed
                )
                gr.Markdown("📹 Start the webcam to reveal the YOLO view above. Detections update in real-time and frames are chunked every ~1 second!")
            else:
                gr.Markdown("❌ YOLO not available. Install with: `pip install ultralytics`")

            # Troubleshooting
            with gr.Accordion("⚠️ Connection Troubleshooting", open=False):
                gr.Markdown("""
                **If video doesn't connect:**

                1. **Allow camera permissions** in your browser
                2. **Use HTTPS** - Hugging Face Spaces provides this automatically
                3. **Try Chrome/Edge** - Best webcam streaming support
                4. **Wait 30-60 seconds** on first load for YOLO model download
                5. **Check browser console** for errors (F12)

                Live streaming uses browser-based webcam APIs; ensure camera access is allowed.
                """)

        # Right column - AI Query and Stats
        with gr.Column(scale=1):
            gr.Markdown("## 🤖 AI Query Interface")

            query_input = gr.Textbox(
                label="Ask about the video",
                placeholder="e.g., What objects appeared in the last 30 seconds?",
                lines=3
            )
            query_btn = gr.Button("🔍 Ask AI", variant="primary")
            query_output = gr.Markdown("*AI response will appear here*")

            gr.Markdown("---")

            # Stats
            stats_display = gr.Markdown(value=get_stats, every=10)
            refresh_btn = gr.Button("🔄 Refresh Stats", size="sm")

            gr.Markdown("---")

            # Current detections
            detections_display = gr.Markdown(
                value=get_current_detections,
                every=10
            )

            gr.Markdown("---")

            # Recent chunks
            chunks_display = gr.Markdown(
                value=get_recent_chunks,
                every=10
            )

            gr.Markdown("---")

            # Event log
            gr.Markdown("### 📝 Event Log")
            log_display = gr.Markdown(
                value=get_event_log,
                every=10
            )

    # How it works
    with gr.Accordion("ℹ️ How This Works", open=False):
        gr.Markdown("""
        ### 🎯 Features:

        **1. Real-time Object Detection:**
        - YOLOv8 detects objects in your webcam feed
        - Color detection identifies object colors
        - Bounding boxes drawn in real-time

        **2. Frame Chunking:**
        - Video frames grouped into 1-second chunks (30 frames)
        - Chunks stored in memory (last 100) and vector database

        **3. Vector Database (ChromaDB):**
        - Semantic embeddings of video events
        - Similarity search across video history

        **4. OpenAI Integration:**
        - GPT-4o-mini for intelligent query answering
        - text-embedding-3-small for semantic search
        - Context-aware responses based on video history

        ### 🔧 Tech Stack:
        - **YOLOv8**: Real-time object detection
        - **Gradio Live Video**: Smooth webcam streaming
        - **OpenAI GPT**: Natural language understanding
        - **ChromaDB**: Vector similarity search
        - **Hugging Face Spaces**: Free deployment with TURN servers

        ### 💰 Costs:
        - **Hugging Face Spaces**: Free (or $9/month PRO for better resources)
        - **OpenAI API**: Pay-as-you-go (minimal for this use case)
        - **TURN Servers**: Free 10GB/month via Cloudflare FastRTC
        """)

    # Event handlers
    setup_btn.click(
        fn=setup_api_key,
        inputs=[api_key_input],
        outputs=[api_status, stats_display]
    )

    query_btn.click(
        fn=query_with_ai,
        inputs=[query_input],
        outputs=[query_output]
    )

    refresh_btn.click(
        fn=lambda: [get_stats(), get_current_detections(), get_recent_chunks(), get_event_log()],
        outputs=[stats_display, detections_display, chunks_display, log_display]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()
