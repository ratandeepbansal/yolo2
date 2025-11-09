"""
Smooth Video Analysis using streamlit-webrtc
Real-time object tracking without page refresh glitching
"""

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import cv2
import numpy as np
from collections import deque
import time

# Try to import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="Video Analysis - Smooth",
    page_icon="🎥",
    layout="wide"
)

# Initialize session state
if 'object_count' not in st.session_state:
    st.session_state.object_count = 0
if 'detected_objects' not in st.session_state:
    st.session_state.detected_objects = []
if 'event_log' not in st.session_state:
    st.session_state.event_log = deque(maxlen=20)

# Load YOLO model
@st.cache_resource
def load_yolo():
    if YOLO_AVAILABLE:
        return YOLO('yolov8n.pt')
    return None

model = load_yolo()

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

# Video frame callback
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    """Process each video frame"""
    # Convert frame to numpy array
    img = frame.to_ndarray(format="bgr24")

    if model:
        # Run YOLO detection
        results = model(img, conf=0.4, verbose=False)

        detected_objects = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    label = model.names[cls]

                    # Get color of object
                    roi = img[y1:y2, x1:x2]
                    color = get_dominant_color(roi)

                    detected_objects.append({
                        'label': label,
                        'color': color,
                        'confidence': conf
                    })

                    # Draw bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Draw label
                    text = f"{color} {label} {conf:.2f}"
                    cv2.putText(img, text, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Update session state
        st.session_state.object_count = len(detected_objects)
        st.session_state.detected_objects = detected_objects

        # Add stats overlay
        cv2.putText(img, f"Objects: {len(detected_objects)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# Main UI
st.title("🎥 Smooth Video Analysis - Object Tracking")

if not YOLO_AVAILABLE:
    st.warning("YOLO not available. Install with: pip install ultralytics")
    st.stop()

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Live Video Feed")

    # WebRTC streamer - this provides smooth video without page refresh
    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    st.info("Click 'START' to begin video analysis. The video will stream smoothly without page refreshes!")

with col2:
    st.header("📊 Stats & Query")

    # Stats display
    stats_container = st.empty()

    # Auto-update stats
    if webrtc_ctx.state.playing:
        with stats_container.container():
            st.metric("Objects Detected", st.session_state.object_count)

            if st.session_state.detected_objects:
                st.subheader("Current Objects:")
                for i, obj in enumerate(st.session_state.detected_objects):
                    st.write(f"{i+1}. {obj['color']} {obj['label']} ({obj['confidence']:.2f})")

    # Query interface
    st.subheader("🔍 Ask Questions")
    query = st.text_input("Ask about what you see:", placeholder="e.g., How many people?")

    if st.button("Ask"):
        if query:
            # Simple query processing
            objects = st.session_state.detected_objects

            if "how many" in query.lower():
                for obj_type in ['person', 'cup', 'bottle', 'phone', 'laptop']:
                    if obj_type in query.lower():
                        count = sum(1 for obj in objects if obj_type in obj['label'].lower())
                        st.success(f"I see {count} {obj_type}(s)")
                        break
                else:
                    st.success(f"I see {len(objects)} objects total")

            elif "color" in query.lower():
                colors = list(set(obj['color'] for obj in objects))
                if colors:
                    st.success(f"Colors: {', '.join(colors)}")
                else:
                    st.info("No objects detected yet")

            else:
                st.info(f"Currently tracking {len(objects)} objects")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This app uses:
    - **streamlit-webrtc** for smooth video streaming
    - **YOLOv8** for object detection
    - **Real-time processing** without page refresh

    **How to use:**
    1. Click START to begin video
    2. Objects will be detected and labeled
    3. Use the query interface to ask questions

    **Smooth video!** No more glitching or page refreshes.
    """)

    st.header("🎯 Features")
    st.write("""
    - ✅ Real-time object detection
    - ✅ Color identification
    - ✅ Smooth video feed
    - ✅ Interactive queries
    - ✅ No page refresh glitching
    """)
