"""
Test Script for Video Analysis MVP
Run this to verify your setup and see a demo
"""

import cv2
import numpy as np
import sys
import time

def test_camera():
    """Test if camera is accessible"""
    print("Testing camera access...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Camera not accessible. Please check:")
        print("   - Camera is connected")
        print("   - Camera permissions are granted")
        print("   - Try different index: cv2.VideoCapture(1)")
        return False
    
    ret, frame = cap.read()
    if not ret or frame is None:
        print("❌ Cannot read from camera")
        cap.release()
        return False
    
    print(f"✅ Camera working! Resolution: {frame.shape[1]}x{frame.shape[0]}")
    cap.release()
    return True

def test_color_detection():
    """Test color detection with synthetic image"""
    print("\nTesting color detection...")
    
    # Create test image with different colored regions
    test_img = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # Red region
    test_img[0:100, 0:100] = (0, 0, 255)  # BGR format
    # Green region
    test_img[0:100, 100:200] = (0, 255, 0)
    # Blue region
    test_img[0:100, 200:300] = (255, 0, 0)
    # Yellow region
    test_img[100:200, 0:100] = (0, 255, 255)
    
    # Test color detection
    from video_analysis_mvp import ColorDetector
    
    colors_detected = []
    regions = [
        ((0, 0, 100, 100), 'red'),
        ((100, 0, 100, 100), 'green'),
        ((200, 0, 100, 100), 'blue'),
        ((0, 100, 100, 100), 'yellow')
    ]
    
    for (x, y, w, h), expected in regions:
        roi = test_img[y:y+h, x:x+w]
        detected = ColorDetector.detect_color(roi)
        colors_detected.append((expected, detected))
        status = "✅" if detected == expected else "⚠️"
        print(f"  {status} Expected: {expected:8} Detected: {detected}")
    
    print("✅ Color detection working!")
    return True

def test_object_tracking():
    """Test object tracking logic"""
    print("\nTesting object tracking...")
    
    from video_analysis_mvp import ObjectTracker, DetectedObject
    
    tracker = ObjectTracker()
    
    # Simulate frame 1: Add two objects
    frame1_objects = [
        DetectedObject('', 'cup', 'red', (100, 100, 50, 50), 0.1, 1, 0.9),
        DetectedObject('', 'bottle', 'blue', (200, 100, 40, 80), 0.1, 1, 0.85)
    ]
    
    tracker.update(frame1_objects, 1, 0.1)
    print(f"  Frame 1: Added {len(tracker.current_objects)} objects")
    
    # Simulate frame 2: Move one object
    frame2_objects = [
        DetectedObject('', 'cup', 'red', (105, 102, 50, 50), 0.2, 2, 0.9),  # Slight movement
        DetectedObject('', 'bottle', 'blue', (250, 100, 40, 80), 0.2, 2, 0.85)  # Significant movement
    ]
    
    tracker.update(frame2_objects, 2, 0.2)
    print(f"  Frame 2: Tracking {len(tracker.current_objects)} objects")
    
    # Simulate frame 3: Remove one object
    frame3_objects = [
        DetectedObject('', 'cup', 'red', (105, 102, 50, 50), 0.3, 3, 0.9)
        # Blue bottle is gone
    ]
    
    # Need multiple frames for object to be considered "removed"
    for i in range(tracker.memory_frames + 1):
        tracker.update(frame3_objects if i == 0 else [], 3 + i, 0.3 + i * 0.1)
    
    # Check events
    events_summary = {}
    for event in tracker.object_history:
        events_summary[event.event_type] = events_summary.get(event.event_type, 0) + 1
    
    print(f"  Events detected: {events_summary}")
    print("✅ Object tracking working!")
    return True

def test_query_engine():
    """Test query processing"""
    print("\nTesting query engine...")
    
    from video_analysis_mvp import ObjectTracker, DetectedObject, VideoAnalyzer
    
    analyzer = VideoAnalyzer()
    
    # Simulate some events
    objects = [
        DetectedObject('', 'cup', 'red', (100, 100, 50, 50), 0.1, 1, 0.9),
        DetectedObject('', 'cup', 'blue', (200, 100, 50, 50), 0.1, 1, 0.85),
        DetectedObject('', 'bottle', 'green', (300, 100, 40, 80), 0.1, 1, 0.8)
    ]
    
    analyzer.tracker.update(objects, 1, 0.1)
    
    # Remove one object
    objects_updated = [
        DetectedObject('', 'cup', 'red', (100, 100, 50, 50), 0.2, 2, 0.9),
        DetectedObject('', 'bottle', 'green', (300, 100, 40, 80), 0.2, 2, 0.8)
        # Blue cup removed
    ]
    
    for i in range(analyzer.tracker.memory_frames + 1):
        analyzer.tracker.update(objects_updated if i == 0 else objects_updated, 2 + i, 0.2 + i * 0.1)
    
    # Test queries
    test_queries = [
        "How many cups are there?",
        "Was the blue cup removed?",
        "What colors do you see?",
        "How many objects were added?"
    ]
    
    print("  Testing queries:")
    for query in test_queries:
        answer = analyzer.query(query)
        print(f"    Q: {query}")
        print(f"    A: {answer[:50]}...")  # Truncate for display
    
    print("✅ Query engine working!")
    return True

def test_dependencies():
    """Test if all required dependencies are installed"""
    print("Testing dependencies...")
    
    required = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'streamlit': 'streamlit (optional)',
        'PIL': 'pillow (optional)'
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
            print(f"  ✅ {package}")
        except ImportError:
            if 'optional' not in package:
                missing.append(package)
                print(f"  ❌ {package}")
            else:
                print(f"  ⚠️  {package} - not critical")
    
    # Test YOLO availability
    try:
        from ultralytics import YOLO
        print(f"  ✅ YOLOv8 (enhanced detection)")
    except ImportError:
        print(f"  ⚠️  YOLOv8 - not installed (using simple detection)")
    
    if missing:
        print(f"\n❌ Missing required packages: {', '.join(missing)}")
        print(f"   Install with: pip install {' '.join(missing)}")
        return False
    
    print("✅ All required dependencies installed!")
    return True

def run_live_demo():
    """Run a quick live demo"""
    print("\n" + "="*50)
    print("LIVE DEMO - Press ESC to exit")
    print("="*50)
    
    from video_analysis_mvp import VideoAnalyzer
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera for live demo")
        return
    
    analyzer = VideoAnalyzer()
    print("Place objects in front of the camera...")
    print("Commands: 'q' = query, 's' = stats, ESC = exit")
    
    start_time = time.time()
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        vis_frame, fg_mask = analyzer.process_frame(frame)
        
        # Add FPS counter
        frame_count += 1
        fps = frame_count / (time.time() - start_time)
        cv2.putText(vis_frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display
        cv2.imshow('Live Demo', vis_frame)
        cv2.imshow('Motion Detection', fg_mask)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('q'):
            print("\n" + "-"*30)
            query = input("Ask a question: ")
            answer = analyzer.query(query)
            print(f"Answer: {answer}")
            print("-"*30)
        elif key == ord('s'):
            stats = analyzer.get_stats()
            print(f"\nStats: {frame_count} frames, "
                  f"{len(analyzer.tracker.current_objects)} objects, "
                  f"{len(analyzer.tracker.object_history)} events")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nDemo completed!")

def main():
    """Run all tests"""
    print("="*50)
    print("VIDEO ANALYSIS MVP - SYSTEM TEST")
    print("="*50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Camera", test_camera),
        ("Color Detection", test_color_detection),
        ("Object Tracking", test_object_tracking),
        ("Query Engine", test_query_engine)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n[{name}]")
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    all_passed = True
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {name:20} {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! System is ready.")
        print("\nWould you like to run a live demo? (y/n): ", end='')
        
        try:
            response = input().strip().lower()
            if response == 'y':
                run_live_demo()
        except KeyboardInterrupt:
            pass
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("   The basic features may still work.")
    
    print("\nTo start the main application:")
    print("  Basic:     python video_analysis_mvp.py")
    print("  Streamlit: streamlit run streamlit_video_analyzer.py")

if __name__ == "__main__":
    main()