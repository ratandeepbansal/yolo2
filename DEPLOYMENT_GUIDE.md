# 🚀 Hugging Face Spaces Deployment Guide

Complete step-by-step guide to deploy the Gradio AI Video Analysis app to Hugging Face Spaces.

## Prerequisites

- [x] Hugging Face account (create at https://huggingface.co/join)
- [x] GitHub account with this repository
- [x] OpenAI API key (get from https://platform.openai.com/api-keys)

## 📋 Deployment Steps

### Step 1: Create a Hugging Face Space

1. **Go to Hugging Face Spaces**
   - Visit https://huggingface.co/spaces
   - Click "Create new Space"

2. **Configure Your Space**
   - **Space name**: `ai-video-analysis` (or your preferred name)
   - **License**: MIT
   - **SDK**: Select "Gradio"
   - **Space hardware**: CPU (basic) - Free tier
     - *Optional: Upgrade to GPU for faster YOLO inference*
   - **Visibility**: Public or Private (your choice)

3. **Click "Create Space"**

### Step 2: Connect to GitHub Repository

You have two options:

#### Option A: Direct Upload (Simpler)

1. **Upload files to your Space**:
   - Click "Files" tab in your Space
   - Upload these files:
     - `gradio_ai_enhanced.py`
     - `requirements_gradio.txt` (rename to `requirements.txt`)
     - `README_SPACES.md` (rename to `README.md`)

2. **Wait for build**:
   - HF Spaces will automatically detect the SDK and build your app
   - First build takes 2-5 minutes (downloads YOLO model)

#### Option B: GitHub Sync (Recommended)

1. **Prepare your repository**:
   ```bash
   # In your local yolo2 directory

   # Copy requirements for HF Spaces
   cp requirements_gradio.txt requirements.txt

   # Copy README for HF Spaces
   cp README_SPACES.md README.md

   # Commit changes
   git add .
   git commit -m "Prepare for HF Spaces deployment"
   git push
   ```

2. **Link GitHub repo to Space**:
   - In your Space, click "Settings" tab
   - Scroll to "Repository"
   - Click "Link to GitHub"
   - Select your repository: `ratandeepbansal/yolo2`
   - Choose branch: `main`
   - **App file**: `gradio_ai_enhanced.py`
   - Click "Link repository"

3. **Automatic syncing**:
   - Every push to GitHub will auto-deploy to your Space
   - Great for continuous updates!

### Step 3: Configure Environment Variables (Optional)

If you want the app to have a default OpenAI key (users can still override):

1. **Go to Space Settings**
   - Click "Settings" tab
   - Scroll to "Repository secrets"

2. **Add OpenAI API Key**:
   - Click "New secret"
   - **Name**: `OPENAI_API_KEY`
   - **Value**: Your OpenAI API key (starts with `sk-...`)
   - Click "Add secret"

3. **Update code to use secret** (optional):
   ```python
   # In gradio_ai_enhanced.py, add at top:
   import os
   DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "")

   # Then use DEFAULT_API_KEY as default value in the UI
   ```

### Step 4: Wait for Deployment

1. **Building Space**:
   - HF Spaces will install dependencies
   - Download YOLOv8n model (~6MB)
   - Start the Gradio app
   - **First build**: 2-5 minutes
   - **Subsequent builds**: 30-60 seconds

2. **Check Build Logs**:
   - Click "Logs" tab to see progress
   - Look for: `Running on public URL: https://...`

3. **Space Status**:
   - "Building" 🟡 → Installing dependencies
   - "Running" 🟢 → App is live!
   - "Error" 🔴 → Check logs for issues

### Step 5: Test Your Deployment

1. **Visit your Space URL**:
   - `https://huggingface.co/spaces/YOUR_USERNAME/ai-video-analysis`

2. **Test WebRTC connection**:
   - Enter OpenAI API key and click "Connect"
   - Click webcam button
   - Allow camera permissions
   - Wait 5-10 seconds for connection
   - You should see your video with object detection!

3. **Test AI queries**:
   - Let the app run for 30+ seconds
   - Ask: "What objects have appeared?"
   - Should get relevant GPT response

4. **Check stats**:
   - Click "Refresh Stats"
   - Verify chunks are being stored
   - Check event log for errors

## 🎨 Customization Options

### Change Space Theme

In `gradio_ai_enhanced.py`:
```python
# Current
demo = gr.Blocks(title="AI Video Analysis", theme=gr.themes.Soft())

# Options:
theme=gr.themes.Default()  # Default Gradio theme
theme=gr.themes.Glass()    # Glass-morphic style
theme=gr.themes.Monochrome()  # Black & white
theme=gr.themes.Base()     # Minimal theme
```

### Enable GPU Acceleration (Paid)

1. **Upgrade Space Hardware**:
   - Settings → Hardware → Select GPU
   - Options: T4 (small), A10G (medium), A100 (large)
   - Costs: ~$0.60-3/hour

2. **Add ZeroGPU (Free GPU bursts)**:
   ```python
   # Add to requirements.txt
   spaces==0.19.0

   # In gradio_ai_enhanced.py
   import spaces

   @spaces.GPU
   def process_frame(frame):
       # Your YOLO processing code
   ```

### Custom Domain (PRO)

- Upgrade to HF PRO ($9/month)
- Get custom subdomain: `your-app.hf.space`

## 🐛 Troubleshooting

### Issue: Build fails with dependency errors

**Solution**:
```bash
# Make sure requirements.txt has exact versions
gradio==5.0.0
opencv-python-headless==4.8.1.78
ultralytics==8.0.196
```

### Issue: YOLO model download fails

**Solution**:
- First build may timeout if model download is slow
- Click "Factory reboot" in Settings
- Try again - model is cached after first success

### Issue: WebRTC doesn't connect

**Checklist**:
- ✅ Using HTTPS (HF Spaces provides this)
- ✅ Camera permissions allowed
- ✅ Using Chrome/Edge browser
- ✅ Not behind restrictive firewall
- ✅ Wait 30 seconds on first connection

### Issue: "OpenAI API error"

**Solution**:
- Check API key is valid: https://platform.openai.com/api-keys
- Verify API key has credits
- Check you didn't hit rate limits

### Issue: Space shows "Sleeping"

**Solution**:
- Free tier Spaces sleep after 48h of inactivity
- Click to wake it up (takes ~10 seconds)
- Upgrade to PRO for always-on Spaces

## 📊 Monitoring & Analytics

### View Usage Stats

1. **Space Analytics** (PRO only):
   - Settings → Analytics
   - See visitor counts, uptime, etc.

2. **Check Logs**:
   - Logs tab shows real-time output
   - Useful for debugging errors

### Monitor Costs

1. **OpenAI API Usage**:
   - https://platform.openai.com/usage
   - Track embedding + GPT costs
   - Set usage limits if needed

2. **Cloudflare TURN Usage**:
   - Free 10GB/month
   - Gradio tracks this automatically
   - Overage: pay-as-you-go

## 🔄 Updating Your Space

### Via GitHub (Recommended)

```bash
# Make changes locally
git add .
git commit -m "Update feature X"
git push

# HF Spaces auto-deploys in ~30-60 seconds
```

### Via Web Interface

1. Files tab → Click file to edit
2. Make changes
3. Commit changes
4. Space rebuilds automatically

## 🎉 Going Live Checklist

- [ ] Space deployed and running
- [ ] WebRTC connection works
- [ ] YOLO detection accurate
- [ ] AI queries responding correctly
- [ ] No errors in logs
- [ ] README clear and helpful
- [ ] Optional: Custom branding/theme
- [ ] Optional: Set up custom domain
- [ ] Share your Space! 🚀

## 📱 Sharing Your Space

Your Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME`

**Share on:**
- Twitter/X with #HuggingFace #Gradio
- LinkedIn
- Your portfolio
- GitHub README

**Embed in website:**
```html
<iframe
  src="https://YOUR_USERNAME-ai-video-analysis.hf.space"
  width="100%"
  height="800px"
></iframe>
```

## 💡 Next Steps

1. **Optimize performance**:
   - Use smaller YOLO model (yolov8n) for speed
   - Adjust detection confidence threshold
   - Tune frame processing rate

2. **Add features**:
   - Object tracking across frames
   - Export video clips
   - Multi-camera support
   - Custom YOLO models

3. **Scale up**:
   - Upgrade to GPU for faster inference
   - Enable ZeroGPU for burst performance
   - Add rate limiting for public use

## 📚 Additional Resources

- **Gradio Docs**: https://gradio.app/docs
- **HF Spaces Docs**: https://huggingface.co/docs/hub/spaces
- **Gradio WebRTC Guide**: https://gradio.app/guides/real-time-object-detection
- **YOLO Ultralytics**: https://docs.ultralytics.com
- **Get Help**: https://discuss.huggingface.co/

---

**Deployment Time**: ~25 minutes
**Difficulty**: Easy
**Cost**: $0-9/month (vs $50-220/month for Streamlit + TURN)

Good luck with your deployment! 🚀
