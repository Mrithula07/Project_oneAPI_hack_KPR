import cv2
import numpy as np
import mediapipe as mp

# Mediapipe hands initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Canvas setup
paintWindow = np.ones((471, 636, 3), dtype=np.uint8) * 255  # White canvas
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]  # Blue, Green, Red, Yellow
colorIndex = 0

# Define clear ROI (x1, y1, x2, y2) and save ROI
clear_roi = (10, 10, 160, 60)  # Clear ROI on the top-left corner
save_roi = (466, 10, 626, 60)   # Save ROI on the top-right corner

# Function to detect finger positions
def get_finger_positions(hand_landmarks):
    finger_positions = {}
    for id, lm in enumerate(hand_landmarks.landmark):
        finger_positions[id] = (lm.x, lm.y)
    return finger_positions

# Function to check if fingers are close together
def are_fingers_close(finger_positions, index_finger_tip_id=8, middle_finger_tip_id=12, threshold=0.05):
    index_tip = finger_positions[index_finger_tip_id]
    middle_tip = finger_positions[middle_finger_tip_id]
    distance = np.linalg.norm(np.array(index_tip) - np.array(middle_tip))
    return distance < threshold

# Function to draw a smooth line
def draw_smooth_line(img, pt1, pt2, color, thickness=5):
    cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)

# Main loop
cap = cv2.VideoCapture(0)

last_point = None  # Variable to keep track of the last drawn point
canvas_saved = False  # State variable to track if canvas has been saved

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Flip the frame

    # Mediapipe hands processing
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Draw the clear ROI rectangle on the frame
    cv2.rectangle(frame, (clear_roi[0], clear_roi[1]), (clear_roi[2], clear_roi[3]), (200, 200, 200), -1)  # Gray box
    text = "Clear All"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = clear_roi[0] + (clear_roi[2] - clear_roi[0] - text_size[0]) // 2
    text_y = clear_roi[1] + (clear_roi[3] - clear_roi[1] + text_size[1]) // 2
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # White text inside the box

    # Draw the save ROI rectangle on the frame
    cv2.rectangle(frame, (save_roi[0], save_roi[1]), (save_roi[2], save_roi[3]), (0, 255, 0), -1)  # Green box
    save_text = "Save"
    save_text_size = cv2.getTextSize(save_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    save_text_x = save_roi[0] + (save_roi[2] - save_roi[0] - save_text_size[0]) // 2
    save_text_y = save_roi[1] + (save_roi[3] - save_roi[1] + save_text_size[1]) // 2
    cv2.putText(frame, save_text, (save_text_x, save_text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # White text inside the box

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get finger positions
            finger_positions = get_finger_positions(hand_landmarks)

            # Get coordinates of the index and middle finger tip landmarks
            index_finger_top = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_top = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            index_x = int(index_finger_top.x * frame.shape[1])
            index_y = int(index_finger_top.y * frame.shape[0])
            middle_x = int(middle_finger_top.x * frame.shape[1])
            middle_y = int(middle_finger_top.y * frame.shape[0])

            # Check if both tips are within the clear ROI and close together
            if (clear_roi[0] < middle_x < clear_roi[2] and clear_roi[1] < middle_y < clear_roi[3]) and are_fingers_close(finger_positions):
                paintWindow[:] = 255  # Clear the canvas by setting it to white
                canvas_saved = False  # Reset canvas saved state when clearing

            # Check if both tips are within the save ROI and close together
            if (save_roi[0] < middle_x < save_roi[2] and save_roi[1] < middle_y < save_roi[3]) and are_fingers_close(finger_positions):
                if not canvas_saved:  # Only save if not already saved
                    # Save the canvas as an image
                    filename = 'canvas_output.png'
                    cv2.imwrite(filename, paintWindow)  # Save the canvas image
                    print(f"Canvas saved as {filename}")
                    canvas_saved = True  # Update state to indicate canvas has been saved
                continue  # Skip drawing if in save ROI

            # Check if index and middle fingers are close
            if are_fingers_close(finger_positions):
                # Get the center point for drawing
                center_x = int((finger_positions[8][0] + finger_positions[12][0]) * frame.shape[1] / 2)
                center_y = int((finger_positions[8][1] + finger_positions[12][1]) * frame.shape[0] / 2)
                current_point = (center_x, center_y)

                # Draw a line only if the last point exists
                if last_point is not None:
                    # Draw smooth lines between points
                    draw_smooth_line(paintWindow, last_point, current_point, colors[colorIndex], 5)  # Draw line to new point

                # Update last_point to current position
                last_point = current_point
            else:
                # Reset last point when fingers are not close
                last_point = None

            # Draw small circle (cursor) at the index finger tip only inside the canvas
            canvas_height, canvas_width = paintWindow.shape[:2]
            if 0 < index_x < canvas_width and 0 < index_y < canvas_height:
                # Draw the cursor but not as part of drawing (just a visual aid)
                temp_canvas = paintWindow.copy()  # Create a copy to show the cursor without altering the canvas
                cv2.circle(temp_canvas, (int(index_finger_top.x * canvas_width), int(index_finger_top.y * canvas_height)), 10, (0, 0, 0), 2)  # Black circle as cursor
                cv2.imshow("Paint", temp_canvas)  # Show canvas with cursor
            else:
                # Show the paint window without a cursor if the index finger is not on the canvas
                cv2.imshow("Paint", paintWindow)

            # Now, draw hand landmarks after drawing the ROI boxes
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show the tracking window with hand landmarks and ROIs
    cv2.imshow("Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
