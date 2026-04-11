import cv2
import numpy as np
from skimage.filters import frangi

def live_vein_scanner():
    # 1. Initialize the Webcam (0 is usually the built-in webcam)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    # 2. Create the Window and Live Slider
    cv2.namedWindow("Live Vein Scanner")
    
    # Dummy function for the trackbar
    def nothing(x):
        pass
        
    cv2.createTrackbar("Threshold", "Live Vein Scanner", 12, 100, nothing)

    print("Starting Live Scanner...")
    print("Hold your hand inside the green box. Press 'q' to quit.")

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame so it acts like a mirror (much easier to use)
        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]

        # 3. Define the Targeting Box (ROI) in the center of the screen
        # Making it 250x250 pixels keeps the math fast and real-time
        box_size = 250
        x = int((width - box_size) / 2)
        y = int((height - box_size) / 2)

        # Draw a white rectangle on the main frame to guide the user
        cv2.rectangle(frame, (x, y), (x + box_size, y + box_size), (255, 255, 255), 2)
        cv2.putText(frame, "FILL BOX WITH SKIN", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Extract just the pixels inside the box for heavy processing
        roi = frame[y:y+box_size, x:x+box_size]

        # --- 4. THE OPTICAL PIPELINE (Running only on the ROI) ---
        b, green_channel, r = cv2.split(roi)
        
        # Hair removal (Dull Razor)
        hair_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        hairless_green = cv2.morphologyEx(green_channel, cv2.MORPH_CLOSE, hair_kernel)
        
        # Contrast & Blur
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced_green = clahe.apply(hairless_green)
        blurred_green = cv2.GaussianBlur(enhanced_green, (5, 5), 0)

        # --- 5. THE FRANGI CALCULUS ---
        # We use fewer sigmas here to speed up processing for live video
        vein_probabilities = frangi(blurred_green, sigmas=np.arange(3, 9, 2), beta=0.7, gamma=25, black_ridges=True)
        vein_prob_normalized = cv2.normalize(vein_probabilities, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # --- 6. LIVE TUNING & NOISE FILTERING ---
        current_threshold = cv2.getTrackbarPos("Threshold", "Live Vein Scanner")
        _, binary_veins = cv2.threshold(vein_prob_normalized, current_threshold, 255, cv2.THRESH_BINARY)

        # Contour filtering (Kill the speckles)
        contours, _ = cv2.findContours(binary_veins, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean_mask = np.zeros_like(binary_veins)
        for contour in contours:
            if cv2.contourArea(contour) > 30: 
                cv2.drawContours(clean_mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Bridging the lines
        bridge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        final_veins = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, bridge_kernel)

        # --- 7. THE OVERLAY ---
        # Color the discovered veins bright green
        roi[final_veins == 255] = [0, 255, 0]

        # Put the processed ROI back into the main webcam feed!
        frame[y:y+box_size, x:x+box_size] = roi

        # Show the live feed
        cv2.imshow("Live Vein Scanner", frame)

        # Press 'q' to break the loop and close the app
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up and turn off the webcam
    cap.release()
    cv2.destroyAllWindows()

# Run the live scanner
live_vein_scanner()