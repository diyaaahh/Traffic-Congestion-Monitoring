import os
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

# Initialize environment (keep your original settings)
HOME = os.getcwd()
model = YOLO("detect2/train/weights/best.pt")
CLASS_NAMES_DICT = model.model.names
print(CLASS_NAMES_DICT)

# Initialize components from your original code
byte_tracker = sv.ByteTrack(
    track_activation_threshold=0.25,
    lost_track_buffer=30,
    minimum_matching_threshold=0.8,
    frame_rate=30,
    minimum_consecutive_frames=3
)

# Keep your original annotators
box_annotator = sv.BoxAnnotator(thickness=4)
label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1.5, text_color=sv.Color.BLACK)
trace_annotator = sv.TraceAnnotator(thickness=4, trace_length=50)

def process_frame(frame: np.ndarray) -> np.ndarray:
    """Process each frame with your original detection logic"""
    # Your original detection and tracking code
    results = model(frame,conf=0.25,  verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = byte_tracker.update_with_detections(detections)
    
    # Your original labeling logic
    labels = [
        f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
        for confidence, class_id, tracker_id
        in zip(detections.confidence, detections.class_id, detections.tracker_id)
    ]
    
    # Your original annotation pipeline
    annotated_frame = frame.copy()
    annotated_frame = trace_annotator.annotate(annotated_frame, detections)
    annotated_frame = box_annotator.annotate(annotated_frame, detections)
    annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)
    
    return annotated_frame

def main():
    # Initialize camera (using your original line coordinates)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Initialize line counter with your original positions (adjusted for camera resolution)
    line_zone = sv.LineZone(
        start=sv.Point(50, 240),  # Adjusted for 640x480 resolution
        end=sv.Point(590, 240)
    )
    line_zone_annotator = sv.LineZoneAnnotator()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame with your original detection logic
            processed_frame = process_frame(frame)
            
            # Update line counter (using your original trigger logic)
            results = model(frame, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            line_zone.trigger(detections)
            
            # Add line annotations (your original visualization)
            line_zone_annotator.annotate(processed_frame, line_zone)
            
            # Display counts
            cv2.putText(processed_frame, 
                       f"In: {line_zone.in_count} Out: {line_zone.out_count}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show output
            cv2.imshow('Real-Time Vehicle Detection', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Final Counts - In: {line_zone.in_count}, Out: {line_zone.out_count}")

if __name__ == "__main__":
    main()