# AI-Enhanced Video Analysis

Real-time video analysis combining YOLOv8 object detection with AI-powered semantic search and natural language queries.

## Features

- **Real-time Object Detection**: YOLOv8n for fast, accurate object detection
- **Color Recognition**: Automatic dominant color extraction for detected objects
- **Frame Chunking**: Groups video into 1-second chunks for efficient processing
- **Vector Database**: ChromaDB for semantic similarity search across video history
- **AI-Powered Queries**: Ask natural language questions about your video using GPT-4o-mini
- **WebRTC Streaming**: Smooth, low-latency video streaming with streamlit-webrtc

## Architecture

```
Webcam → WebRTC → YOLO Detection → Frame Chunking → Vector DB (ChromaDB)
                                                            ↓
User Query → OpenAI Embeddings → Similarity Search → GPT-4o-mini → Answer
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

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

- **Streamlit**: Web interface
- **YOLOv8**: Object detection
- **OpenAI GPT-4o-mini**: Natural language understanding
- **ChromaDB**: Vector similarity search
- **streamlit-webrtc**: Video streaming
- **OpenCV**: Image processing

## License

MIT
