import numpy as np
import cv2 as cv
img = cv.imread('circle.jpeg')
output = img.copy()
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = cv.medianBlur(gray, 5)
circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 20,
                          param1=50, param2=30, minRadius=0, maxRadius=0)
detected_circles = np.uint16(np.around(circles))
for (x, y ,r) in detected_circles[0, :]:
    cv.circle(output, (x, y), r, (0, 0, 0), 3)
    cv.circle(output, (x, y), 2, (0, 255, 255), 3)


cv.imshow('output',output)
cv.waitKey(0)
cv.destroyAllWindows()


# new added code by code-withprasad


import cv2
import mediapipe as mp
import numpy as np
import random
import tkinter as tk
from PIL import Image, ImageTk

# Function to detect circles in an image
def detect_circles(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise and help circle detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use HoughCircles to detect circles
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=50
    )
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        return circles[0]
    else:
        return []

# Function to draw circles on the image
def draw_circles(image, circles):
    for i in circles:
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)

# Main game loop
cap = cv2.VideoCapture(0)  # Open the camera

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize the position of the bowl
bowl_radius = 30

# Initialize the score
score = 0

# Create the Tkinter window
root = tk.Tk()
root.title("Circle Collection Game")

# Create a label for displaying video feed
video_label = tk.Label(root)
video_label.pack(padx=10, pady=10)

# Create a label for displaying the score
score_label = tk.Label(root, text="Score: 0", font=("Helvetica", 16))
score_label.pack(padx=10, pady=10)

# List to store falling circles
falling_circles = []

# Function to update the Tkinter window
def update_gui():
    global falling_circles  # Declare falling_circles as global
    global score  # Declare score as global

    ret, frame = cap.read()  # Read a frame from the camera

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe Hands
    results = hands.process(rgb_frame)

    # Detect circles in the frame
    detected_circles = detect_circles(frame)

    # Draw circles on the frame
    draw_circles(frame, detected_circles)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the coordinates of the hand's palm (center)
            x, y = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * frame.shape[1]), int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * frame.shape[0])

            # Draw a blue circle (bowl) at the palm's position
            cv2.circle(frame, (x, y), bowl_radius, (255, 0, 0), -1)

            # Check if the circles are inside the bowl
            for circle in falling_circles:
                circle[1] += 2  # Adjust the vertical speed (increase or decrease as needed)
                if np.linalg.norm(np.array((x, y)) - np.array(circle[:2])) < bowl_radius - circle[2]:
                    print("Circle Collected!")
                    score += 1

    # Generate a new circle with a certain probability
    if random.random() < 0.02:
        new_circle = [random.randint(0, frame.shape[1]), 0, random.randint(10, 50)]
        falling_circles.append(new_circle)

    # Draw and update falling circles
    for circle in falling_circles:
        cv2.circle(frame, (circle[0], circle[1]), circle[2], (0, 0, 255), 2)

    # Remove circles that have reached the bottom of the screen
    falling_circles = [circle for circle in falling_circles if circle[1] < frame.shape[0]]

    # Display the frame
    img = Image.fromarray(frame)
    img = ImageTk.PhotoImage(image=img)
    video_label.imgtk = img
    video_label.configure(image=img)

    # Update the score label
    score_label.config(text=f"Score: {score}")

    # Schedule the next update
    root.after(10, update_gui)

# Schedule the first update
update_gui()

# Start the Tkinter main loop
root.mainloop()

# Release the camera
cap.release()

