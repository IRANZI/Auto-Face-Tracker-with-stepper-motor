import cv2
import time
import math

# Load the pre-trained face detector (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start video capture
cap = cv2.VideoCapture(0)

prev_cx, prev_cy = None, None
prev_time = time.time()
direction = "Center"
speed = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Compute face center
        cx = x + w // 2
        cy = y + h // 2

        # Draw center point
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        # Compute direction and speed
        if prev_cx is not None and prev_cy is not None:
            dx = cx - prev_cx
            dy = cy - prev_cy

            # Compute movement direction
            if abs(dx) > abs(dy):
                if dx > 10:
                    direction = "Right"
                elif dx < -10:
                    direction = "Left"
            else:
                if dy > 10:
                    direction = "Down"
                elif dy < -10:
                    direction = "Up"
                else:
                    direction = "Center"

            # Compute speed in pixels/second
            curr_time = time.time()
            dt = curr_time - prev_time
            distance = math.sqrt(dx**2 + dy**2)
            if dt > 0:
                speed = distance / dt

            prev_time = curr_time

        # Update previous center
        prev_cx, prev_cy = cx, cy

        break  # track only one face for simplicity

    # Display direction and speed
    cv2.putText(frame, f"Direction: {direction}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Speed: {speed:.2f} px/s", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Face Movement Tracker", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
