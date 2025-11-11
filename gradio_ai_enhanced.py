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

        # Object tracking data structures
        self.track_history = {}  # track_id -> {trajectory, first_seen, last_seen, class, color, etc.}
        self.unique_tracks = set()  # Set of all unique track IDs ever seen
        self.active_tracks = set()  # Currently visible track IDs
        self.track_lifetimes = {}  # track_id -> {first_seen, last_seen, duration}
        self.movement_data = {}  # track_id -> deque of positions for velocity calculation
        self.track_classes = {}  # track_id -> object class name
        self.track_colors = {}  # track_id -> dominant color

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
        # Run YOLO tracking (ByteTrack algorithm)
        results = state.model.track(
            frame_bgr,
            conf=0.4,
            verbose=False,
            persist=True,  # Persist tracks between frames
            tracker="bytetrack.yaml"  # Use ByteTrack algorithm
        )

        detected_objects = []
        events_text = []
        current_frame_tracks = set()

        for r in results:
            boxes = r.boxes
            if boxes is not None and boxes.id is not None:
                # Tracking is active
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    label = state.model.names[cls]
                    track_id = int(box.id[0].item())  # Extract track ID

                    # Get color
                    try:
                        roi = frame_bgr[y1:y2, x1:x2]
                        color = get_dominant_color(roi)
                    except:
                        color = "unknown"

                    # Calculate center position
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    detected_objects.append({
                        'label': label,
                        'color': color,
                        'confidence': conf,
                        'bbox': (x1, y1, x2, y2),
                        'track_id': track_id,
                        'center': (center_x, center_y)
                    })

                    current_frame_tracks.add(track_id)
                    events_text.append(f"{color} {label} (ID:{track_id})")

                    # Draw bounding box
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Draw label with track ID
                    text = f"ID:{track_id} {color} {label} {conf:.2f}"
                    cv2.putText(frame_bgr, text, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            elif boxes is not None:
                # Fallback: tracking not available, use regular detection
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
                        'bbox': (x1, y1, x2, y2),
                        'track_id': None
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
            current_time = time.time()

            # Update tracking data structures
            for obj in detected_objects:
                if obj.get('track_id') is not None:
                    track_id = obj['track_id']

                    # Track this ID as seen
                    state.unique_tracks.add(track_id)

                    # Initialize tracking data if new track
                    if track_id not in state.track_history:
                        state.track_history[track_id] = {
                            'first_seen': current_time,
                            'last_seen': current_time,
                            'trajectory': deque(maxlen=30),  # Last 30 positions
                            'class': obj['label'],
                            'color': obj['color']
                        }
                        state.track_classes[track_id] = obj['label']
                        state.track_colors[track_id] = obj['color']
                        state.movement_data[track_id] = deque(maxlen=30)

                    # Update tracking data
                    state.track_history[track_id]['last_seen'] = current_time
                    state.track_history[track_id]['trajectory'].append(obj['center'])
                    state.movement_data[track_id].append(obj['center'])

                    # Update lifetime
                    duration = current_time - state.track_history[track_id]['first_seen']
                    state.track_lifetimes[track_id] = {
                        'first_seen': state.track_history[track_id]['first_seen'],
                        'last_seen': current_time,
                        'duration': duration
                    }

            # Update active tracks
            state.active_tracks = current_frame_tracks.copy()

            # Create chunks every 30 frames
            if state.chunk_id % 30 == 0 and events_text:
                # Build enhanced chunk description with tracking info
                unique_track_count = len(current_frame_tracks)
                total_unique_count = len(state.unique_tracks)

                # Build detailed tracking description
                track_details = []
                for obj in detected_objects:
                    if obj.get('track_id') is not None:
                        track_id = obj['track_id']
                        duration = state.track_lifetimes.get(track_id, {}).get('duration', 0)
                        track_details.append(
                            f"Track#{track_id} ({obj['color']} {obj['label']}, seen for {duration:.1f}s)"
                        )

                chunk_description = f"At {datetime.now().strftime('%H:%M:%S')}: "
                chunk_description += f"{unique_track_count} active objects, {total_unique_count} unique tracks total. "
                chunk_description += f"Details: {', '.join(track_details) if track_details else 'No tracked objects'}"

                state.frame_chunks.append({
                    'id': state.chunk_id,
                    'timestamp': time.time(),
                    'description': chunk_description,
                    'objects': detected_objects.copy(),
                    'active_tracks': list(current_frame_tracks),
                    'unique_tracks_count': total_unique_count
                })

                state.pending_chunks.append({
                    'id': state.chunk_id,
                    'description': chunk_description,
                    'timestamp': time.time(),
                    'object_count': len(detected_objects),
                    'active_tracks': list(current_frame_tracks),
                    'unique_tracks_total': total_unique_count
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
                # Build metadata with tracking information
                metadata = {
                    'timestamp': chunk['timestamp'],
                    'object_count': chunk['object_count']
                }

                # Add tracking metadata if available
                if 'active_tracks' in chunk:
                    metadata['active_tracks'] = ','.join(map(str, chunk['active_tracks']))
                    metadata['active_track_count'] = len(chunk['active_tracks'])

                if 'unique_tracks_total' in chunk:
                    metadata['unique_tracks_total'] = chunk['unique_tracks_total']

                state.video_collection.add(
                    documents=[chunk['description']],
                    embeddings=[embedding],
                    ids=[f"chunk_{chunk['id']}"],
                    metadatas=[metadata]
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

        # Get current state with tracking information
        with state.lock:
            current_objects = state.detected_objects.copy()
            frames_seen = state.frames_processed
            total_unique_tracks = len(state.unique_tracks)
            active_tracks_count = len(state.active_tracks)
            track_classes_summary = {}
            for track_id, class_name in state.track_classes.items():
                track_classes_summary[class_name] = track_classes_summary.get(class_name, 0) + 1

        # Build current state description with tracking info
        if current_objects:
            obj_descriptions = []
            for o in current_objects:
                if o.get('track_id') is not None:
                    obj_descriptions.append(f"Track#{o['track_id']} ({o['color']} {o['label']})")
                else:
                    obj_descriptions.append(f"{o['color']} {o['label']}")

            current_state = f"Currently visible: {', '.join(obj_descriptions)}\n"
            current_state += f"Active tracks: {active_tracks_count}, Total unique tracks seen: {total_unique_tracks}"
        else:
            if frames_seen > 0:
                current_state = f"Video stream active but no objects detected in the latest frame.\nTotal unique tracks seen: {total_unique_tracks}"
            else:
                current_state = "No video frames processed yet."

        # Add tracking statistics
        tracking_summary = f"\nTracking Statistics:\n"
        tracking_summary += f"- Total unique objects tracked: {total_unique_tracks}\n"
        tracking_summary += f"- Currently active: {active_tracks_count}\n"
        if track_classes_summary:
            tracking_summary += "- Unique objects by class: " + ", ".join([f"{count} {cls}" for cls, count in track_classes_summary.items()])
        else:
            tracking_summary += "- No objects tracked yet"

        current_state += tracking_summary

        if not context_docs and frames_seen > 0:
            context = "Video stream active, waiting for notable detections to log."

        # Create prompt
        prompt = f"""You are a video analysis assistant with object tracking capabilities. Answer the question based on the video footage context.

IMPORTANT: The system uses persistent object tracking, which means each detected object has a unique Track ID that persists across frames. This allows you to:
- Count UNIQUE objects (not just current detections)
- Identify when the SAME object appears multiple times
- Track how long objects remain visible
- Distinguish between different instances of the same object class

Video Event History (from vector database):
{context}

Current Frame:
{current_state}

Question: {question}

Provide a concise, helpful answer based on the video data. When answering questions about "how many" objects, use the unique track counts, not just current detections. When asked if an object appeared, check both current state and historical tracking data."""

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
        unique_tracks = len(state.unique_tracks)
        active_tracks = len(state.active_tracks)

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
- **Unique Tracks (Total): {unique_tracks}**
- **Active Tracks (Now): {active_tracks}**
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
        track_info = f" [Track #{obj['track_id']}]" if obj.get('track_id') is not None else ""
        output += f"{i+1}. {obj['color']} {obj['label']} ({obj['confidence']:.2f}){track_info}\n"

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

def get_tracking_stats():
    """Get detailed tracking statistics"""
    with state.lock:
        unique_count = len(state.unique_tracks)
        active_count = len(state.active_tracks)
        track_classes = state.track_classes.copy()
        track_lifetimes = state.track_lifetimes.copy()

    if unique_count == 0:
        return "**Tracking Statistics:**\n\nNo objects tracked yet - start the video!"

    # Build class summary
    class_counts = {}
    for track_id, class_name in track_classes.items():
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    output = "**Tracking Statistics:**\n\n"
    output += f"**Total Unique Objects:** {unique_count}\n"
    output += f"**Currently Active:** {active_count}\n\n"

    if class_counts:
        output += "**Unique Objects by Class:**\n"
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            output += f"- {class_name}: {count}\n"

    # Show longest tracked objects
    if track_lifetimes:
        sorted_tracks = sorted(track_lifetimes.items(), key=lambda x: x[1]['duration'], reverse=True)[:5]
        output += f"\n**Longest Tracked (Top 5):**\n"
        for track_id, lifetime in sorted_tracks:
            class_name = track_classes.get(track_id, "unknown")
            duration = lifetime['duration']
            output += f"- Track #{track_id} ({class_name}): {duration:.1f}s\n"

    return output

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

            # Tracking statistics (NEW)
            gr.Markdown("### 🎯 Object Tracking")
            tracking_stats_display = gr.Markdown(
                value=get_tracking_stats,
                every=10
            )

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

        **1. Real-time Object Detection with Tracking:**
        - YOLOv8 detects objects in your webcam feed
        - ByteTrack algorithm for persistent object tracking
        - Unique Track IDs maintain object identity across frames
        - Color detection identifies object colors
        - Bounding boxes drawn in real-time with Track IDs

        **2. Persistent Object Tracking:**
        - Each detected object gets a unique Track ID
        - Track objects even when they leave and return to frame
        - Count unique objects (not just current detections)
        - Measure object dwell time and trajectories
        - Distinguish between different instances of same object class

        **3. Frame Chunking:**
        - Video frames grouped into 1-second chunks (30 frames)
        - Chunks stored in memory (last 100) and vector database
        - Tracking metadata included in each chunk

        **4. Vector Database (ChromaDB):**
        - Semantic embeddings of video events with tracking data
        - Similarity search across video history
        - Track-aware metadata for advanced queries

        **5. OpenAI Integration:**
        - GPT-4o-mini for intelligent query answering
        - text-embedding-3-small for semantic search
        - Context-aware responses based on video history and tracking data
        - Answer questions like "How many unique people appeared?"

        ### 🔧 Tech Stack:
        - **YOLOv8 + ByteTrack**: Real-time object detection with persistent tracking
        - **Gradio Live Video**: Smooth webcam streaming
        - **OpenAI GPT-4o-mini**: Natural language understanding
        - **ChromaDB**: Vector similarity search with tracking metadata
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
        fn=lambda: [get_stats(), get_tracking_stats(), get_current_detections(), get_recent_chunks(), get_event_log()],
        outputs=[stats_display, tracking_stats_display, detections_display, chunks_display, log_display]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()
