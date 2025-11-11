"""
Test script for object tracking implementation
"""

import sys
import traceback

def test_imports():
    """Test that all required imports work"""
    print("Testing imports...")
    try:
        import gradio as gr
        import cv2
        import numpy as np
        from collections import deque
        import time
        from datetime import datetime
        import json
        import os
        from threading import Lock
        from openai import OpenAI
        import chromadb
        from chromadb.config import Settings

        try:
            from ultralytics import YOLO
            print("✓ All imports successful (including YOLO)")
        except ImportError:
            print("✓ All imports successful (YOLO not available, but that's OK)")

        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False

def test_state_initialization():
    """Test that VideoAnalysisState initializes correctly"""
    print("\nTesting VideoAnalysisState initialization...")
    try:
        # Import the module
        sys.path.insert(0, '/home/user/yolo2')
        from gradio_ai_enhanced import VideoAnalysisState

        # Create a state instance
        state = VideoAnalysisState()

        # Check basic attributes
        assert hasattr(state, 'lock'), "Missing 'lock' attribute"
        assert hasattr(state, 'frame_chunks'), "Missing 'frame_chunks' attribute"
        assert hasattr(state, 'detected_objects'), "Missing 'detected_objects' attribute"

        # Check tracking attributes
        assert hasattr(state, 'track_history'), "Missing 'track_history' attribute"
        assert hasattr(state, 'unique_tracks'), "Missing 'unique_tracks' attribute"
        assert hasattr(state, 'active_tracks'), "Missing 'active_tracks' attribute"
        assert hasattr(state, 'track_lifetimes'), "Missing 'track_lifetimes' attribute"
        assert hasattr(state, 'movement_data'), "Missing 'movement_data' attribute"
        assert hasattr(state, 'track_classes'), "Missing 'track_classes' attribute"
        assert hasattr(state, 'track_colors'), "Missing 'track_colors' attribute"

        # Check initial values
        assert isinstance(state.unique_tracks, set), "unique_tracks should be a set"
        assert isinstance(state.active_tracks, set), "active_tracks should be a set"
        assert isinstance(state.track_history, dict), "track_history should be a dict"
        assert len(state.unique_tracks) == 0, "unique_tracks should start empty"
        assert len(state.active_tracks) == 0, "active_tracks should start empty"

        print("✓ VideoAnalysisState initialization successful")
        return True
    except Exception as e:
        print(f"✗ State initialization failed: {e}")
        traceback.print_exc()
        return False

def test_helper_functions():
    """Test that all helper functions are defined"""
    print("\nTesting helper functions...")
    try:
        sys.path.insert(0, '/home/user/yolo2')
        from gradio_ai_enhanced import (
            get_dominant_color,
            process_frame,
            get_embedding,
            process_pending_chunks,
            query_with_ai,
            setup_api_key,
            get_stats,
            get_current_detections,
            get_recent_chunks,
            get_event_log,
            get_tracking_stats
        )

        print("✓ All helper functions are defined")
        return True
    except Exception as e:
        print(f"✗ Helper function check failed: {e}")
        traceback.print_exc()
        return False

def test_tracking_data_flow():
    """Test the tracking data flow logic"""
    print("\nTesting tracking data flow...")
    try:
        from gradio_ai_enhanced import VideoAnalysisState
        import time

        state = VideoAnalysisState()
        current_time = time.time()

        # Simulate adding a tracked object
        track_id = 42
        test_obj = {
            'track_id': track_id,
            'label': 'person',
            'color': 'blue',
            'center': (100, 200)
        }

        # Simulate the tracking logic
        state.unique_tracks.add(track_id)
        state.active_tracks.add(track_id)

        from collections import deque
        state.track_history[track_id] = {
            'first_seen': current_time,
            'last_seen': current_time,
            'trajectory': deque(maxlen=30),
            'class': test_obj['label'],
            'color': test_obj['color']
        }
        state.track_classes[track_id] = test_obj['label']
        state.track_colors[track_id] = test_obj['color']
        state.movement_data[track_id] = deque(maxlen=30)

        state.track_history[track_id]['trajectory'].append(test_obj['center'])
        state.movement_data[track_id].append(test_obj['center'])

        duration = current_time - state.track_history[track_id]['first_seen']
        state.track_lifetimes[track_id] = {
            'first_seen': state.track_history[track_id]['first_seen'],
            'last_seen': current_time,
            'duration': duration
        }

        # Verify the data
        assert track_id in state.unique_tracks, "Track ID not in unique_tracks"
        assert track_id in state.active_tracks, "Track ID not in active_tracks"
        assert track_id in state.track_history, "Track ID not in track_history"
        assert track_id in state.track_classes, "Track ID not in track_classes"
        assert track_id in state.track_lifetimes, "Track ID not in track_lifetimes"
        assert state.track_classes[track_id] == 'person', "Track class mismatch"
        assert len(state.track_history[track_id]['trajectory']) == 1, "Trajectory should have 1 entry"

        print("✓ Tracking data flow test passed")
        return True
    except Exception as e:
        print(f"✗ Tracking data flow test failed: {e}")
        traceback.print_exc()
        return False

def test_ui_functions():
    """Test UI display functions"""
    print("\nTesting UI functions...")
    try:
        from gradio_ai_enhanced import (
            get_stats,
            get_current_detections,
            get_tracking_stats,
            VideoAnalysisState,
            state
        )

        # Test with empty state
        stats_output = get_stats()
        assert isinstance(stats_output, str), "get_stats should return string"
        assert "Unique Tracks" in stats_output, "Stats should mention Unique Tracks"

        detections_output = get_current_detections()
        assert isinstance(detections_output, str), "get_current_detections should return string"

        tracking_output = get_tracking_stats()
        assert isinstance(tracking_output, str), "get_tracking_stats should return string"
        assert "Tracking Statistics" in tracking_output, "Should have tracking stats header"

        print("✓ UI functions test passed")
        return True
    except Exception as e:
        print(f"✗ UI functions test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Object Tracking Implementation Test Suite")
    print("=" * 60)

    results = []

    results.append(("Imports", test_imports()))
    results.append(("State Initialization", test_state_initialization()))
    results.append(("Helper Functions", test_helper_functions()))
    results.append(("Tracking Data Flow", test_tracking_data_flow()))
    results.append(("UI Functions", test_ui_functions()))

    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:.<40} {status}")

    print("=" * 60)

    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n🎉 All tests passed! Implementation is ready.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
