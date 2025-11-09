---
title: AI Video Analysis
emoji: 🎥
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.0.0
app_file: gradio_ai_enhanced.py
pinned: false
license: mit
---

# 🎥 AI-Enhanced Video Analysis

Real-time object detection from your webcam with AI-powered query capabilities using GPT-4o-mini and vector search.

## 🚀 Features

- **Live Object Detection**: YOLOv8 analyzes your webcam feed in real-time
- **Color Recognition**: Identifies object colors (red, blue, green, etc.)
- **AI Queries**: Ask questions about what appeared in the video
- **Vector Search**: Semantic search through video history using ChromaDB
- **Frame Chunking**: Automatic grouping of video events for efficient storage

## 🎯 How to Use

1. **Enter your OpenAI API key** in the text box and click "Connect"
   - Get a key from: https://platform.openai.com/api-keys
   - Alternatively, the Space admin can set it as a repository secret

2. **Click the webcam button** to start video streaming
   - Allow camera permissions when prompted
   - Wait a few seconds for YOLO model to load (first time only)

3. **Watch objects being detected** in real-time with bounding boxes and labels

4. **Ask questions** about the video:
   - "What objects have appeared in the last minute?"
   - "When did you see a red object?"
   - "How many different objects were detected?"

## 🔧 Technical Stack

- **YOLOv8**: Real-time object detection
- **Gradio WebRTC**: Smooth video streaming with Cloudflare TURN servers
- **OpenAI GPT-4o-mini**: Natural language query understanding
- **OpenAI Embeddings**: Semantic search capabilities
- **ChromaDB**: Vector database for storing video events

## 💰 Costs

- **Hugging Face Spaces**: Free (this Space)
- **Cloudflare TURN Servers**: Free 10GB/month via Gradio FastRTC
- **OpenAI API**: Pay-as-you-go
  - Embeddings: ~$0.0001 per chunk
  - GPT-4o-mini: ~$0.0001 per query
  - Typical usage: <$1/month for moderate use

## 🛠️ Local Development

```bash
# Clone the repo
git clone https://github.com/ratandeepbansal/yolo2.git
cd yolo2

# Install dependencies
pip install -r requirements_gradio.txt

# Set up API key
cp .env.example .env
# Edit .env and add your OpenAI API key

# Run the app
python gradio_ai_enhanced.py
```

## 📝 Notes

- First load takes ~30-60 seconds to download YOLOv8n model (~6MB)
- WebRTC works best in Chrome/Edge browsers
- Camera permissions required for webcam access
- HTTPS required (automatically provided by HF Spaces)

## 🤝 Contributing

This is an open-source project. Feel free to:
- Report issues
- Suggest features
- Submit pull requests

## 📄 License

MIT License - see LICENSE file for details

---

Built with ❤️ using Gradio and YOLOv8
