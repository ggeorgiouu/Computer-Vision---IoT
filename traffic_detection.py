import torch
import cv2
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda.amp")

# Load YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "custom", path="yolov5/yolov5s.pt")

# Define the classes to detect (vehicles)
vehicle_classes = ["car", "motorcycle", "bus", "truck"]


# Function to filter detections
def filter_detections(detections):
    filtered = []
    for det in detections:
        if det["name"] in vehicle_classes:
            filtered.append(det)
    return filtered


# Load video or camera feed
cap = cv2.VideoCapture("video_1_simple.mp4")  # Replace with 0 for webcam

# Get the frame rate of the video
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps * 3)  # Number of frames to skip for a 3-second interval

# Get the width and height of the frame
ret, frame = cap.read()
if not ret:
    raise ValueError("Failed to read the video")

frame_height, frame_width = frame.shape[:2]

# Calculate the center of the frame
center_x, center_y = frame_width // 2, frame_height // 2

# Define the polygon relative to the center of the frame
polygon_size = 400  # Example size
polygon = [
    (center_x - polygon_size, center_y - polygon_size),
    (center_x + polygon_size, center_y - polygon_size),
    (center_x + polygon_size, center_y + polygon_size),
    (center_x - polygon_size, center_y + polygon_size),
]

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Skip frames to achieve a 3-second interval
    if frame_count % frame_interval == 0:
        # Perform detection
        results = model(frame)

        # Filter detections
        filtered_results = filter_detections(
            results.pandas().xyxy[0].to_dict(orient="records")
        )

        # Count the number of each class within the polygon
        class_counts = {cls: 0 for cls in vehicle_classes}
        for det in filtered_results:
            xmin, ymin, xmax, ymax = (
                int(det["xmin"]),
                int(det["ymin"]),
                int(det["xmax"]),
                int(det["ymax"]),
            )
            label = det["name"]
            center_x = (xmin + xmax) // 2
            center_y = (ymin + ymax) // 2

            # Check if the center of the bounding box is inside the polygon
            if (
                cv2.pointPolygonTest(
                    np.array(polygon, dtype=np.int32), (center_x, center_y), False
                )
                >= 0
            ):
                class_counts[label] += 1

                # Draw bounding boxes for detections inside the polygon
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{label} {det['confidence']:.2f}",
                    (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

        # Print the counts
        print(f"Frame {frame_count}: {class_counts}")

        # Draw the polygon on the frame
        cv2.polylines(
            frame,
            [np.array(polygon, dtype=np.int32)],
            isClosed=True,
            color=(0, 0, 255),
            thickness=2,
        )

        # Display the frame
        cv2.imshow("Vehicle Detection", frame)

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
