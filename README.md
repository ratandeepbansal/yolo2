# AI-Enhanced Video Analysis with Object Tracking

Real-time video analysis combining YOLOv8 object detection with **persistent object tracking**, AI-powered semantic search, and natural language queries.

## Features

- **Real-time Object Detection**: YOLOv8n for fast, accurate object detection
- **🎯 Persistent Object Tracking**: ByteTrack algorithm maintains unique IDs for each object across frames
- **Unique Object Counting**: Track and count individual objects, not just current detections
- **Object Lifecycle Tracking**: Monitor how long objects remain visible and their movement patterns
- **Color Recognition**: Automatic dominant color extraction for detected objects
- **Frame Chunking**: Groups video into 1-second chunks for efficient processing
- **Vector Database**: ChromaDB for semantic similarity search across video history with tracking metadata
- **AI-Powered Queries**: Ask natural language questions about your video using GPT-4o-mini
- **Tracking-Aware AI**: Answer questions like "How many unique people appeared?" and "Did the same car pass by twice?"
- **WebRTC Streaming**: Smooth, low-latency video streaming

## Architecture

```
Webcam → WebRTC → YOLO + ByteTrack Tracking → Track Management → Frame Chunking → Vector DB (ChromaDB)
                         (Persistent IDs)      (Trajectories)                      (with tracking metadata)
                                                                                           ↓
User Query → OpenAI Embeddings → Similarity Search → GPT-4o-mini (tracking-aware) → Answer
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Gradio Version (Recommended - with Object Tracking)

1. Run the Gradio app:
```bash
python gradio_ai_enhanced.py
```

2. Enter your OpenAI API key and click "Connect"

3. Allow camera access when prompted

4. Ask tracking-aware questions about the video:
   - "How many unique people have appeared?"
   - "Did the same car pass by twice?"
   - "What's the longest someone stayed in frame?"
   - "How many different red objects appeared?"

### Streamlit Version (Legacy)

1. Run the Streamlit app:
```bash
streamlit run streamlit_ai_enhanced.py
```

2. Enter your OpenAI API key in the sidebar

3. Click "START" to begin video analysis

4. Ask questions about the video:
   - "What objects appeared in the last 30 seconds?"
   - "Did you see any red objects?"
   - "How many people were detected?"

## Requirements

- Python 3.8+
- Webcam
- OpenAI API key

## Tech Stack

- **Gradio**: Modern web interface with live video streaming
- **YOLOv8 + ByteTrack**: Real-time object detection with persistent tracking
- **OpenAI GPT-4o-mini**: Natural language understanding (tracking-aware)
- **OpenAI text-embedding-3-small**: Semantic embeddings
- **ChromaDB**: Vector similarity search with tracking metadata
- **OpenCV**: Image processing and color detection
- **Streamlit** (legacy): Alternative web interface

## License

MIT
