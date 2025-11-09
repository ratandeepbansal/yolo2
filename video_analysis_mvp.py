"""
Video Analysis MVP - Object Tracking and Query System
Tracks objects in real-time and answers questions about scene changes
"""

import cv2
import numpy as np
from collections import defaultdict, deque
from datetime import datetime, timedelta
import json
import threading
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import colorsys

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

@dataclass
class SceneEvent:
    """Represents an event in the scene"""
    event_type: str  # 'added', 'removed', 'moved'
    object_id: str
    object_label: str
    object_color: str
    timestamp: float
    frame_number: int
    details: Dict

class ObjectTracker:
    """Tracks objects across frames and maintains scene history"""
    
    def __init__(self, iou_threshold=0.3, memory_frames=30):
        self.current_objects = {}  # Currently visible objects
        self.object_history = []   # All historical events
        self.next_id = 1
        self.iou_threshold = iou_threshold
        self.memory_frames = memory_frames
        self.frame_buffer = deque(maxlen=memory_frames)
        
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
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
        # Match current detections with tracked objects
        matched_objects = set()
        new_detections = []
        
        for detection in detections:
            best_match = None
            best_iou = self.iou_threshold
            
            for obj_id, obj in self.current_objects.items():
                if obj.label == detection.label and obj.color == detection.color:
                    iou = self.calculate_iou(obj.bbox, detection.bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_match = obj_id
            
            if best_match:
                # Update existing object
                matched_objects.add(best_match)
                old_bbox = self.current_objects[best_match].bbox
                self.current_objects[best_match] = detection
                self.current_objects[best_match].id = best_match
                
                # Check if object moved significantly
                center_old = (old_bbox[0] + old_bbox[2]/2, old_bbox[1] + old_bbox[3]/2)
                center_new = (detection.bbox[0] + detection.bbox[2]/2, 
                            detection.bbox[1] + detection.bbox[3]/2)
                distance = np.sqrt((center_old[0] - center_new[0])**2 + 
                                 (center_old[1] - center_new[1])**2)
                
                if distance > 50:  # Significant movement threshold
                    self.object_history.append(SceneEvent(
                        event_type='moved',
                        object_id=best_match,
                        object_label=detection.label,
                        object_color=detection.color,
                        timestamp=timestamp,
                        frame_number=frame_number,
                        details={'distance': distance, 'from': old_bbox, 'to': detection.bbox}
                    ))
            else:
                new_detections.append(detection)
        
        # Handle objects that disappeared
        for obj_id in list(self.current_objects.keys()):
            if obj_id not in matched_objects:
                obj = self.current_objects[obj_id]
                self.object_history.append(SceneEvent(
                    event_type='removed',
                    object_id=obj_id,
                    object_label=obj.label,
                    object_color=obj.color,
                    timestamp=timestamp,
                    frame_number=frame_number,
                    details={'last_position': obj.bbox}
                ))
                del self.current_objects[obj_id]
        
        # Add new objects
        for detection in new_detections:
            obj_id = f"obj_{self.next_id}"
            self.next_id += 1
            detection.id = obj_id
            self.current_objects[obj_id] = detection
            
            self.object_history.append(SceneEvent(
                event_type='added',
                object_id=obj_id,
                object_label=detection.label,
                object_color=detection.color,
                timestamp=timestamp,
                frame_number=frame_number,
                details={'position': detection.bbox}
            ))

class ColorDetector:
    """Detects dominant colors in image regions"""
    
    # Define color ranges in HSV
    COLOR_RANGES = {
        'red': [(0, 50, 50), (10, 255, 255), (170, 50, 50), (180, 255, 255)],
        'green': [(35, 50, 50), (85, 255, 255)],
        'blue': [(85, 50, 50), (135, 255, 255)],
        'yellow': [(20, 50, 50), (35, 255, 255)],
        'orange': [(10, 50, 50), (20, 255, 255)],
        'purple': [(135, 50, 50), (170, 255, 255)],
        'white': [(0, 0, 200), (180, 30, 255)],
        'black': [(0, 0, 0), (180, 255, 30)],
        'gray': [(0, 0, 31), (180, 30, 199)]
    }
    
    @staticmethod
    def detect_color(image_region):
        """Detect dominant color in an image region"""
        if image_region.size == 0:
            return 'unknown'
        
        # Convert to HSV
        hsv = cv2.cvtColor(image_region, cv2.COLOR_BGR2HSV)
        
        # Check each color range
        color_scores = {}
        for color_name, ranges in ColorDetector.COLOR_RANGES.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            
            # Handle red special case (wraps around in HSV)
            if color_name == 'red':
                mask1 = cv2.inRange(hsv, ranges[0], ranges[1])
                mask2 = cv2.inRange(hsv, ranges[2], ranges[3])
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = cv2.inRange(hsv, ranges[0], ranges[1])
            
            # Calculate percentage of pixels matching this color
            score = cv2.countNonZero(mask) / (hsv.shape[0] * hsv.shape[1])
            color_scores[color_name] = score
        
        # Return color with highest score
        best_color = max(color_scores.items(), key=lambda x: x[1])
        return best_color[0] if best_color[1] > 0.1 else 'unknown'

class SimpleObjectDetector:
    """Simple object detection using contours and shape analysis"""
    
    def __init__(self):
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
    def detect_objects(self, frame):
        """Detect objects in frame using contours"""
        detections = []
        
        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(frame)
        
        # Remove shadows and noise
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:  # Filter small contours
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Classify object based on shape and size
            aspect_ratio = w / h if h > 0 else 0
            extent = area / (w * h) if w * h > 0 else 0
            
            # Simple heuristics for object classification
            if 0.8 < aspect_ratio < 1.3 and extent > 0.6:
                label = 'cup' if area < 5000 else 'bowl'
            elif aspect_ratio > 2 or aspect_ratio < 0.5:
                label = 'bottle' if area < 3000 else 'box'
            else:
                label = 'object'
            
            # Detect color
            roi = frame[y:y+h, x:x+w]
            color = ColorDetector.detect_color(roi)
            
            detections.append({
                'label': label,
                'color': color,
                'bbox': (x, y, w, h),
                'confidence': extent
            })
        
        return detections, fg_mask

class VideoAnalyzer:
    """Main video analysis system"""
    
    def __init__(self):
        self.tracker = ObjectTracker()
        self.detector = SimpleObjectDetector()
        self.frame_count = 0
        self.start_time = time.time()
        self.is_running = False
        self.current_frame = None
        self.processed_frame = None
        
    def process_frame(self, frame):
        """Process a single frame"""
        self.current_frame = frame.copy()
        timestamp = time.time() - self.start_time
        
        # Detect objects
        detections, fg_mask = self.detector.detect_objects(frame)
        
        # Convert to DetectedObject instances
        detected_objects = []
        for det in detections:
            obj = DetectedObject(
                id='',  # Will be assigned by tracker
                label=det['label'],
                color=det['color'],
                bbox=det['bbox'],
                timestamp=timestamp,
                frame_number=self.frame_count,
                confidence=det['confidence']
            )
            detected_objects.append(obj)
        
        # Update tracker
        self.tracker.update(detected_objects, self.frame_count, timestamp)
        
        # Draw visualizations
        vis_frame = frame.copy()
        for obj_id, obj in self.tracker.current_objects.items():
            x, y, w, h = obj.bbox
            color = (0, 255, 0)  # Green for tracked objects
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, 2)
            label = f"{obj.color} {obj.label} [{obj_id}]"
            cv2.putText(vis_frame, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        self.processed_frame = vis_frame
        self.frame_count += 1
        
        return vis_frame, fg_mask
    
    def query(self, question: str) -> str:
        """Answer questions about the scene"""
        question_lower = question.lower()
        
        # Count queries
        if 'how many' in question_lower:
            if 'cup' in question_lower:
                if 'added' in question_lower:
                    events = [e for e in self.tracker.object_history 
                             if e.event_type == 'added' and 'cup' in e.object_label]
                    return f"{len(events)} cup(s) were added to the scene"
                else:
                    current_cups = [obj for obj in self.tracker.current_objects.values() 
                                   if 'cup' in obj.label]
                    return f"There are currently {len(current_cups)} cup(s) in the scene"
            
            elif 'object' in question_lower:
                return f"There are currently {len(self.tracker.current_objects)} object(s) in the scene"
        
        # Specific object queries
        elif 'red cup' in question_lower:
            if 'picked up' in question_lower or 'removed' in question_lower:
                events = [e for e in self.tracker.object_history 
                         if e.event_type == 'removed' and 
                         e.object_color == 'red' and 'cup' in e.object_label]
                if events:
                    return f"Yes, a red cup was removed from the scene at {events[-1].timestamp:.1f}s"
                return "No red cup has been removed from the scene"
            
            elif 'added' in question_lower:
                events = [e for e in self.tracker.object_history 
                         if e.event_type == 'added' and 
                         e.object_color == 'red' and 'cup' in e.object_label]
                if events:
                    return f"Yes, a red cup was added at {events[-1].timestamp:.1f}s"
                return "No red cup has been added to the scene"
        
        # Color queries
        elif 'what color' in question_lower:
            objects = list(self.tracker.current_objects.values())
            if objects:
                colors = [obj.color for obj in objects]
                return f"I see these colors: {', '.join(set(colors))}"
            return "No objects currently in the scene"
        
        # Movement queries
        elif 'moved' in question_lower:
            events = [e for e in self.tracker.object_history if e.event_type == 'moved']
            if events:
                return f"{len(events)} object(s) have moved significantly"
            return "No significant object movement detected"
        
        # History queries
        elif 'what happened' in question_lower:
            if not self.tracker.object_history:
                return "No events recorded yet"
            
            recent_events = self.tracker.object_history[-5:]  # Last 5 events
            event_strs = []
            for e in recent_events:
                event_strs.append(f"- {e.event_type}: {e.object_color} {e.object_label} at {e.timestamp:.1f}s")
            return "Recent events:\n" + "\n".join(event_strs)
        
        return "I don't understand that question. Try: 'How many cups?', 'Was the red cup picked up?', etc."
    
    def get_stats(self) -> Dict:
        """Get current statistics"""
        return {
            'frame_count': self.frame_count,
            'current_objects': len(self.tracker.current_objects),
            'total_events': len(self.tracker.object_history),
            'runtime': time.time() - self.start_time
        }

def main():
    """Main application loop"""
    print("=" * 60)
    print("VIDEO ANALYSIS MVP - Object Tracking & Query System")
    print("=" * 60)
    print("\nINSTRUCTIONS:")
    print("- Place objects (cups, bottles, etc.) in front of camera")
    print("- Press 'q' to ask a question")
    print("- Press 's' to see statistics")
    print("- Press 'h' to see event history")
    print("- Press ESC to exit")
    print("\nExample questions:")
    print("  - How many cups were added?")
    print("  - Was the red cup picked up?")
    print("  - What happened?")
    print("  - How many objects are there?")
    print("-" * 60)
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    analyzer = VideoAnalyzer()
    
    print("\nStarting video analysis... (may take a few seconds to calibrate)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        vis_frame, fg_mask = analyzer.process_frame(frame)
        
        # Display results
        cv2.imshow('Video Analysis', vis_frame)
        cv2.imshow('Motion Mask', fg_mask)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == ord('q'):
            # Ask a question
            cv2.waitKey(1)  # Brief pause
            print("\n" + "=" * 40)
            question = input("Ask a question: ")
            answer = analyzer.query(question)
            print(f"Answer: {answer}")
            print("=" * 40)
        elif key == ord('s'):
            # Show statistics
            stats = analyzer.get_stats()
            print("\n" + "=" * 40)
            print("STATISTICS:")
            print(f"  Frames processed: {stats['frame_count']}")
            print(f"  Current objects: {stats['current_objects']}")
            print(f"  Total events: {stats['total_events']}")
            print(f"  Runtime: {stats['runtime']:.1f} seconds")
            print("=" * 40)
        elif key == ord('h'):
            # Show history
            print("\n" + "=" * 40)
            print("EVENT HISTORY (last 10):")
            history = analyzer.tracker.object_history[-10:]
            if history:
                for event in history:
                    print(f"  [{event.timestamp:.1f}s] {event.event_type}: "
                          f"{event.object_color} {event.object_label}")
            else:
                print("  No events yet")
            print("=" * 40)
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Save session data
    session_data = {
        'stats': analyzer.get_stats(),
        'events': [asdict(e) for e in analyzer.tracker.object_history]
    }
    
    with open('session_data.json', 'w') as f:
        json.dump(session_data, f, indent=2)
    
    print("\nSession data saved to session_data.json")
    print("Goodbye!")

if __name__ == "__main__":
    main()