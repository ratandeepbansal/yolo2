"""
Enhanced Video Analysis MVP with Streamlit UI
Real-time object tracking with YOLO and interactive query interface
"""

import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import json
import time
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional
import threading
import queue
from collections import deque
import tempfile
import os

# Try to import YOLO, fall back to simple detection if not available
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    st.warning("YOLO not available. Using simple detection. Install with: pip install ultralytics")

# Page config
st.set_page_config(
    page_title="Video Analysis MVP",
    page_icon="🎥",
    layout="wide"
)

@dataclass
class DetectedObject:
    """Represents a detected object in a frame"""
    id: str
    label: str
    color: str
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    timestamp: float
    frame_number: int
    confidence: float
    center: Tuple[float, float] = field(default_factory=tuple)
    
    def __post_init__(self):
        if not self.center:
            x, y, w, h = self.bbox
            self.center = (x + w/2, y + h/2)

@dataclass
class SceneEvent:
    """Represents an event in the scene"""
    event_type: str  # 'added', 'removed', 'moved'
    object_id: str
    object_label: str
    object_color: str
    timestamp: float
    frame_number: int
    details: Dict = field(default_factory=dict)

class EnhancedObjectTracker:
    """Enhanced tracking with better object persistence"""
    
    def __init__(self, iou_threshold=0.3, max_disappeared=10):
        self.current_objects = {}
        self.object_history = []
        self.disappeared_count = {}
        self.next_id = 1
        self.iou_threshold = iou_threshold
        self.max_disappeared = max_disappeared
        self.object_trajectories = {}  # Store movement paths
        
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def update(self, detections: List[DetectedObject], frame_number: int, timestamp: float):
        """Update tracking with new detections"""
        matched_objects = set()
        events = []
        
        # Match detections to existing objects
        for detection in detections:
            best_match = None
            best_score = self.iou_threshold
            
            for obj_id, obj in self.current_objects.items():
                # Match based on IOU and appearance
                iou = self.calculate_iou(obj.bbox, detection.bbox)
                
                # Bonus for matching color and label
                appearance_bonus = 0.2 if (obj.label == detection.label and 
                                          obj.color == detection.color) else 0
                
                score = iou + appearance_bonus
                
                if score > best_score:
                    best_score = score
                    best_match = obj_id
            
            if best_match:
                # Update existing object
                matched_objects.add(best_match)
                old_obj = self.current_objects[best_match]
                
                # Check for significant movement
                distance = np.sqrt((old_obj.center[0] - detection.center[0])**2 + 
                                 (old_obj.center[1] - detection.center[1])**2)
                
                if distance > 30:
                    events.append(SceneEvent(
                        event_type='moved',
                        object_id=best_match,
                        object_label=detection.label,
                        object_color=detection.color,
                        timestamp=timestamp,
                        frame_number=frame_number,
                        details={'distance': distance}
                    ))
                
                # Update object
                detection.id = best_match
                self.current_objects[best_match] = detection
                self.disappeared_count[best_match] = 0
                
                # Update trajectory
                if best_match not in self.object_trajectories:
                    self.object_trajectories[best_match] = deque(maxlen=100)
                self.object_trajectories[best_match].append(detection.center)
            else:
                # New object detected
                obj_id = f"obj_{self.next_id}"
                self.next_id += 1
                detection.id = obj_id
                
                self.current_objects[obj_id] = detection
                self.disappeared_count[obj_id] = 0
                self.object_trajectories[obj_id] = deque([detection.center], maxlen=100)
                
                events.append(SceneEvent(
                    event_type='added',
                    object_id=obj_id,
                    object_label=detection.label,
                    object_color=detection.color,
                    timestamp=timestamp,
                    frame_number=frame_number
                ))
        
        # Handle disappeared objects
        for obj_id in list(self.current_objects.keys()):
            if obj_id not in matched_objects:
                self.disappeared_count[obj_id] += 1
                
                if self.disappeared_count[obj_id] > self.max_disappeared:
                    obj = self.current_objects[obj_id]
                    events.append(SceneEvent(
                        event_type='removed',
                        object_id=obj_id,
                        object_label=obj.label,
                        object_color=obj.color,
                        timestamp=timestamp,
                        frame_number=frame_number
                    ))
                    
                    del self.current_objects[obj_id]
                    del self.disappeared_count[obj_id]
                    if obj_id in self.object_trajectories:
                        del self.object_trajectories[obj_id]
        
        # Add events to history
        self.object_history.extend(events)
        return events

class YOLODetector:
    """YOLO-based object detection"""
    
    def __init__(self, model_name='yolov8n.pt'):
        self.model = YOLO(model_name)
        self.class_names = self.model.names
        
    def detect(self, frame):
        """Detect objects using YOLO"""
        results = self.model(frame, conf=0.4)
        detections = []
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    
                    # Get object label
                    label = self.class_names[cls]
                    
                    # Extract ROI for color detection
                    x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
                    roi = frame[y:y+h, x:x+w]
                    color = self.detect_color(roi)
                    
                    detections.append({
                        'label': label,
                        'color': color,
                        'bbox': (x, y, w, h),
                        'confidence': conf
                    })
        
        return detections
    
    @staticmethod
    def detect_color(roi):
        """Detect dominant color in ROI"""
        if roi.size == 0:
            return 'unknown'
        
        # Convert to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Define color ranges
        colors = {
            'red': [(0, 120, 70), (10, 255, 255), (170, 120, 70), (180, 255, 255)],
            'blue': [(94, 80, 50), (126, 255, 255)],
            'green': [(36, 50, 50), (89, 255, 255)],
            'yellow': [(15, 50, 50), (36, 255, 255)],
            'white': [(0, 0, 200), (180, 30, 255)],
            'black': [(0, 0, 0), (180, 255, 30)]
        }
        
        best_color = 'unknown'
        best_score = 0
        
        for color_name, ranges in colors.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            
            if color_name == 'red':
                # Red wraps around in HSV
                mask1 = cv2.inRange(hsv, ranges[0], ranges[1])
                mask2 = cv2.inRange(hsv, ranges[2], ranges[3])
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = cv2.inRange(hsv, ranges[0], ranges[1])
            
            score = cv2.countNonZero(mask) / (hsv.shape[0] * hsv.shape[1])
            
            if score > best_score:
                best_score = score
                best_color = color_name
        
        return best_color if best_score > 0.2 else 'unknown'

class QueryEngine:
    """Natural language query processing"""
    
    def __init__(self, tracker):
        self.tracker = tracker
        
    def answer(self, question: str) -> Dict:
        """Process query and return structured answer"""
        q = question.lower()
        
        # Counting queries
        if 'how many' in q:
            return self._count_query(q)
        
        # Existence queries
        elif any(word in q for word in ['is there', 'are there', 'do you see']):
            return self._existence_query(q)
        
        # Event queries
        elif any(word in q for word in ['picked up', 'removed', 'taken']):
            return self._removal_query(q)
        
        elif 'added' in q or 'placed' in q:
            return self._addition_query(q)
        
        # Movement queries
        elif 'moved' in q:
            return self._movement_query(q)
        
        # History queries
        elif 'what happened' in q or 'history' in q:
            return self._history_query(q)
        
        # Color queries
        elif 'what color' in q:
            return self._color_query(q)
        
        else:
            return {
                'answer': "I don't understand. Try: 'How many cups?', 'Was the red cup removed?'",
                'confidence': 0.0,
                'evidence': []
            }
    
    def _count_query(self, q):
        """Handle counting queries"""
        # Determine what to count
        target = None
        for item in ['cup', 'bottle', 'person', 'object', 'thing']:
            if item in q:
                target = item
                break
        
        if not target:
            target = 'object'
        
        # Count current objects or events
        if 'added' in q:
            events = [e for e in self.tracker.object_history 
                     if e.event_type == 'added' and 
                     (target == 'object' or target in e.object_label.lower())]
            count = len(events)
            answer = f"{count} {target}(s) were added to the scene"
        else:
            objects = [obj for obj in self.tracker.current_objects.values()
                      if target == 'object' or target in obj.label.lower()]
            count = len(objects)
            answer = f"There are currently {count} {target}(s) in the scene"
        
        return {
            'answer': answer,
            'confidence': 0.9,
            'evidence': [f"Found {count} matching objects"]
        }
    
    def _removal_query(self, q):
        """Handle removal/pickup queries"""
        # Extract color and object
        color = None
        for c in ['red', 'blue', 'green', 'yellow', 'black', 'white']:
            if c in q:
                color = c
                break
        
        object_type = None
        for obj in ['cup', 'bottle', 'object']:
            if obj in q:
                object_type = obj
                break
        
        # Find removal events
        events = [e for e in self.tracker.object_history 
                 if e.event_type == 'removed']
        
        if color:
            events = [e for e in events if e.object_color == color]
        if object_type:
            events = [e for e in events if object_type in e.object_label.lower()]
        
        if events:
            last_event = events[-1]
            answer = f"Yes, a {last_event.object_color} {last_event.object_label} was removed at {last_event.timestamp:.1f}s"
            confidence = 0.95
        else:
            answer = f"No {''.join([color + ' ' if color else '', object_type if object_type else 'object'])} has been removed"
            confidence = 0.9
        
        return {
            'answer': answer,
            'confidence': confidence,
            'evidence': [f"Found {len(events)} removal events"]
        }
    
    def _history_query(self, q):
        """Show recent events"""
        if not self.tracker.object_history:
            return {
                'answer': "No events recorded yet",
                'confidence': 1.0,
                'evidence': []
            }
        
        recent = self.tracker.object_history[-5:]
        events_str = []
        for e in recent:
            events_str.append(f"• {e.event_type.title()}: {e.object_color} {e.object_label} at {e.timestamp:.1f}s")
        
        return {
            'answer': "Recent events:\n" + "\n".join(events_str),
            'confidence': 1.0,
            'evidence': [f"Total {len(self.tracker.object_history)} events"]
        }
    
    def _addition_query(self, q):
        """Handle addition queries"""
        color = None
        for c in ['red', 'blue', 'green', 'yellow', 'black', 'white']:
            if c in q:
                color = c
                break
        
        events = [e for e in self.tracker.object_history 
                 if e.event_type == 'added']
        
        if color:
            events = [e for e in events if e.object_color == color]
        
        if events:
            answer = f"Yes, {len(events)} object(s) were added"
            if color:
                answer = f"Yes, {len(events)} {color} object(s) were added"
        else:
            answer = "No objects have been added" if not color else f"No {color} objects have been added"
        
        return {
            'answer': answer,
            'confidence': 0.9,
            'evidence': [f"Found {len(events)} addition events"]
        }
    
    def _movement_query(self, q):
        """Handle movement queries"""
        events = [e for e in self.tracker.object_history if e.event_type == 'moved']
        
        if events:
            answer = f"{len(events)} object(s) have moved significantly"
            details = [f"{e.object_color} {e.object_label}" for e in events[-3:]]
            answer += f"\nRecent: {', '.join(details)}"
        else:
            answer = "No significant movement detected"
        
        return {
            'answer': answer,
            'confidence': 0.85,
            'evidence': [f"Found {len(events)} movement events"]
        }
    
    def _existence_query(self, q):
        """Check if objects exist"""
        color = None
        for c in ['red', 'blue', 'green', 'yellow', 'black', 'white']:
            if c in q:
                color = c
                break
        
        objects = list(self.tracker.current_objects.values())
        if color:
            objects = [o for o in objects if o.color == color]
        
        if objects:
            answer = f"Yes, I see {len(objects)} {color if color else ''} object(s)"
        else:
            answer = f"No {color if color else ''} objects currently visible"
        
        return {
            'answer': answer,
            'confidence': 0.9,
            'evidence': [f"Currently tracking {len(self.tracker.current_objects)} objects"]
        }
    
    def _color_query(self, q):
        """Identify colors in scene"""
        if not self.tracker.current_objects:
            return {
                'answer': "No objects currently in view",
                'confidence': 1.0,
                'evidence': []
            }
        
        colors = set(obj.color for obj in self.tracker.current_objects.values())
        answer = f"I see these colors: {', '.join(colors)}"
        
        return {
            'answer': answer,
            'confidence': 0.95,
            'evidence': [f"From {len(self.tracker.current_objects)} objects"]
        }

class StreamlitVideoAnalyzer:
    """Main Streamlit application"""
    
    def __init__(self):
        if 'analyzer' not in st.session_state:
            st.session_state.analyzer = self.initialize_analyzer()
            st.session_state.frame_placeholder = None
            st.session_state.is_running = False
            st.session_state.query_history = []
            st.session_state.event_log = deque(maxlen=20)
            st.session_state.cap = None
    
    def initialize_analyzer(self):
        """Initialize the analysis components"""
        tracker = EnhancedObjectTracker()
        
        if YOLO_AVAILABLE:
            detector = YOLODetector()
        else:
            # Fallback to simple detection
            from video_analysis_mvp import SimpleObjectDetector
            detector = SimpleObjectDetector()
        
        query_engine = QueryEngine(tracker)
        
        return {
            'tracker': tracker,
            'detector': detector,
            'query_engine': query_engine,
            'frame_count': 0,
            'start_time': time.time()
        }
    
    
    def run(self):
        """Main Streamlit UI"""
        st.title("🎥 Video Analysis MVP - Object Tracking & Query System")
        
        # Sidebar
        with st.sidebar:
            st.header("📊 Statistics")
            stats_placeholder = st.empty()
            
            st.header("📝 Event Log")
            event_placeholder = st.empty()
            
            st.header("💾 Export Data")
            if st.button("Save Session"):
                self.save_session()
        
        # Main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("Video Feed")
            
            # Control buttons
            col_start, col_stop = st.columns(2)
            with col_start:
                if st.button("▶️ Start Camera", key="start"):
                    st.session_state.is_running = True
                    # Initialize camera capture
                    if 'cap' not in st.session_state or st.session_state.cap is None:
                        st.session_state.cap = cv2.VideoCapture(0)
                        st.session_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        st.session_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            with col_stop:
                if st.button("⏹️ Stop Camera", key="stop"):
                    st.session_state.is_running = False
                    if 'cap' in st.session_state and st.session_state.cap is not None:
                        st.session_state.cap.release()
                        st.session_state.cap = None

            # Video display
            video_placeholder = st.empty()

            # Process and display frame if camera is running
            if st.session_state.is_running and 'cap' in st.session_state and st.session_state.cap is not None:
                ret, frame = st.session_state.cap.read()
                if ret:
                    # Process frame
                    analyzer = st.session_state.analyzer
                    timestamp = time.time() - analyzer['start_time']

                    # Detect objects
                    if YOLO_AVAILABLE:
                        detections = analyzer['detector'].detect(frame)
                    else:
                        detections, _ = analyzer['detector'].detect_objects(frame)

                    # Convert to DetectedObject
                    detected_objects = []
                    for det in detections:
                        obj = DetectedObject(
                            id='',
                            label=det['label'],
                            color=det['color'],
                            bbox=det['bbox'],
                            timestamp=timestamp,
                            frame_number=analyzer['frame_count'],
                            confidence=det['confidence']
                        )
                        detected_objects.append(obj)

                    # Update tracker
                    events = analyzer['tracker'].update(
                        detected_objects,
                        analyzer['frame_count'],
                        timestamp
                    )

                    # Log events
                    for event in events:
                        st.session_state.event_log.append(
                            f"[{timestamp:.1f}s] {event.event_type.upper()}: "
                            f"{event.object_color} {event.object_label}"
                        )

                    # Draw visualizations
                    vis_frame = frame.copy()
                    for obj_id, obj in analyzer['tracker'].current_objects.items():
                        x, y, w, h = obj.bbox

                        # Choose color based on object state
                        if obj_id in analyzer['tracker'].disappeared_count:
                            if analyzer['tracker'].disappeared_count[obj_id] > 0:
                                box_color = (0, 165, 255)  # Orange for disappearing
                            else:
                                box_color = (0, 255, 0)  # Green for tracked
                        else:
                            box_color = (0, 255, 0)

                        cv2.rectangle(vis_frame, (x, y), (x+w, y+h), box_color, 2)

                        # Draw trajectory
                        if obj_id in analyzer['tracker'].object_trajectories:
                            points = list(analyzer['tracker'].object_trajectories[obj_id])
                            for i in range(1, len(points)):
                                cv2.line(vis_frame,
                                        (int(points[i-1][0]), int(points[i-1][1])),
                                        (int(points[i][0]), int(points[i][1])),
                                        box_color, 1)

                        # Label
                        label = f"{obj.color} {obj.label} [{obj_id}]"
                        cv2.putText(vis_frame, label, (x, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                    # Display frame
                    frame_rgb = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                    analyzer['frame_count'] += 1

                    # Trigger rerun to get next frame
                    time.sleep(0.1)  # Slow down to ~10 FPS to reduce flickering
                    st.rerun()
            elif 'current_frame' in st.session_state:
                # Display last frame if camera stopped
                frame_rgb = cv2.cvtColor(st.session_state.current_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
        with col2:
            st.header("🔍 Query Interface")
            
            # Query input
            query = st.text_input("Ask a question:", 
                                 placeholder="e.g., How many cups were added?")
            
            if st.button("Submit Query", key="query_submit"):
                if query:
                    # Process query
                    result = st.session_state.analyzer['query_engine'].answer(query)
                    
                    # Add to history
                    st.session_state.query_history.append({
                        'question': query,
                        'answer': result['answer'],
                        'confidence': result['confidence'],
                        'timestamp': time.time()
                    })
                    
                    # Display result
                    st.success(result['answer'])
                    st.caption(f"Confidence: {result['confidence']:.1%}")
                    
                    if result.get('evidence'):
                        with st.expander("Evidence"):
                            for e in result['evidence']:
                                st.write(f"• {e}")
            
            # Example queries
            st.subheader("Example Queries")
            examples = [
                "How many cups were added?",
                "Was the red cup removed?",
                "How many objects are there?",
                "What colors do you see?",
                "What happened recently?",
                "Has anything moved?"
            ]
            
            for example in examples:
                if st.button(example, key=f"ex_{example}"):
                    result = st.session_state.analyzer['query_engine'].answer(example)
                    st.success(result['answer'])
            
            # Query history
            if st.session_state.query_history:
                st.subheader("Query History")
                for i, item in enumerate(reversed(st.session_state.query_history[-5:])):
                    with st.expander(f"Q: {item['question'][:50]}..."):
                        st.write(f"**Answer:** {item['answer']}")
                        st.caption(f"Confidence: {item['confidence']:.1%}")
        
        # Update sidebar stats
        if st.session_state.is_running:
            analyzer = st.session_state.analyzer
            
            # Update statistics
            stats = {
                'Frames': analyzer['frame_count'],
                'Objects': len(analyzer['tracker'].current_objects),
                'Events': len(analyzer['tracker'].object_history),
                'Runtime': f"{time.time() - analyzer['start_time']:.1f}s"
            }
            
            stats_text = "\n\n".join([f"**{k}:** {v}" for k, v in stats.items()])
            stats_placeholder.markdown(stats_text)
            
            # Update event log
            if st.session_state.event_log:
                events_text = "\n\n".join(list(st.session_state.event_log)[-10:])
                event_placeholder.text(events_text)
            
            # Auto-refresh
            time.sleep(0.1)
            st.rerun()
    
    def save_session(self):
        """Save session data to file"""
        analyzer = st.session_state.analyzer
        
        session_data = {
            'stats': {
                'frame_count': analyzer['frame_count'],
                'runtime': time.time() - analyzer['start_time']
            },
            'events': [asdict(e) for e in analyzer['tracker'].object_history],
            'queries': st.session_state.query_history
        }
        
        filename = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        st.success(f"Session saved to {filename}")

if __name__ == "__main__":
    app = StreamlitVideoAnalyzer()
    app.run()