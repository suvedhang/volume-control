# main.py

import cv2
import mediapipe as mp
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# --- 1. Pycaw Volume Control Setup ---
# Get the default audio playback device (your speakers/headphones)
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Get the valid volume range from your system (e.g., -65.0 to 0.0)
vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]


# --- 2. MediaPipe Hand Tracking Setup ---
mp_hands = mp.solutions.hands
# Set confidence thresholds to make detection more stable
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils


# --- 3. Webcam Setup ---
cap = cv2.VideoCapture(0) # Use 0 for the default webcam
if not cap.isOpened():
    print("Error: Could not open video stream from webcam.")
    exit()


# --- 4. Main Application Loop ---
while True:
    # Read a frame from the webcam
    success, img = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a more intuitive, mirror-like display
    img = cv2.flip(img, 1)

    # Convert the BGR image to RGB, as MediaPipe requires RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the image to find hands
    results = hands.process(img_rgb)

    # Check if any hands are detected
    if results.multi_hand_landmarks:
        # We will use the first hand detected
        hand_landmarks = results.multi_hand_landmarks[0]

        # Get the pixel coordinates of the thumb tip (landmark 4) and index finger tip (landmark 8)
        h, w, c = img.shape
        thumb_tip_coords = (int(hand_landmarks.landmark[4].x * w), int(hand_landmarks.landmark[4].y * h))
        index_tip_coords = (int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h))
        
        # Calculate the distance between the thumb and index finger
        # This distance will be our gesture for volume control
        length = math.hypot(index_tip_coords[0] - thumb_tip_coords[0], index_tip_coords[1] - thumb_tip_coords[1])

        # --- Map Hand Distance to Volume Range ---
        # Hand distance is empirically found to be roughly 20 to 200 pixels
        # We use numpy's interp function for a clean linear mapping
        vol_level = np.interp(length, [20, 200], [min_vol, max_vol])
        
        # --- Set the System Volume ---
        volume.SetMasterVolumeLevel(vol_level, None)

        # --- Visual Feedback ---
        # Draw a line and circles on the hand for visual feedback
        cv2.line(img, thumb_tip_coords, index_tip_coords, (0, 255, 0), 3)
        cv2.circle(img, thumb_tip_coords, 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, index_tip_coords, 10, (255, 0, 255), cv2.FILLED)
        
        # If the fingers are close together, change the line color to green
        if length < 25:
            cv2.line(img, thumb_tip_coords, index_tip_coords, (0, 0, 255), 3)

        # Draw all hand landmarks for a cool visual effect
        mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Draw a visual volume bar on the screen
        vol_percentage = np.interp(length, [20, 200], [0, 100])
        vol_bar_height = np.interp(length, [20, 200], [400, 150]) # Map length to pixel height
        cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
        cv2.rectangle(img, (50, int(vol_bar_height)), (85, 400), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{int(vol_percentage)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)


    # Display the resulting frame
    cv2.imshow("Hand Gesture Volume Control", img)

    # Allow the program to exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 5. Cleanup ---
# Release the webcam and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
hands.close()