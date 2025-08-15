import cv2
import numpy as np

# Define color ranges in HSV
colors = {
    "Red": ([0, 120, 70], [10, 255, 255]),
    "Red2": ([170, 120, 70], [180, 255, 255]),
    "Green": ([36, 50, 70], [89, 255, 255]),
    "Blue": ([90, 50, 70], [128, 255, 255]),
    "Yellow": ([20, 100, 100], [30, 255, 255]),
    "Orange": ([10, 100, 20], [20, 255, 255]),
    "Purple": ([129, 50, 70], [158, 255, 255]),
    "Pink": ([145, 50, 70], [165, 255, 255]),
    "White": ([0, 0, 200], [180, 30, 255]),
    "Black": ([0, 0, 0], [180, 255, 30])
}

def detect_color(hsv_roi):
    hist = {}
    for name, (lower, upper) in colors.items():
        lower_np = np.array(lower, dtype=np.uint8)
        upper_np = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv_roi, lower_np, upper_np)
        count = cv2.countNonZero(mask)
        hist[name.replace("2", "")] = hist.get(name.replace("2", ""), 0) + count

    if hist:
        return max(hist, key=hist.get)
    return "Unknown"

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # Define the center box (200x200)
    box_size = 200
    x1 = w // 2 - box_size // 2
    y1 = h // 2 - box_size // 2
    x2 = x1 + box_size
    y2 = y1 + box_size

    # Extract ROI (still needed for color detection)
    roi = frame[y1:y2, x1:x2]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Detect color
    color_detected = detect_color(hsv_roi)

    # Display color name
    cv2.putText(frame, f'Color: {color_detected}', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Show frame
    cv2.imshow("Color Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
