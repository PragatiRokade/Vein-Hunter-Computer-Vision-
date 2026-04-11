import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import frangi
import os
from google import genai 

# Initialize the client with your key
client = genai.Client(api_key="YOUR_ACTUAL_API_KEY_HERE")

def process_veins_api_fixed(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image.")
        return

    # --- 1. STANDARDIZE RESOLUTION ---
    height, width = img.shape[:2]
    new_width = 600
    new_height = int((new_width / width) * height)
    img = cv2.resize(img, (new_width, new_height))

    # --- 2. THE MISSING PIECE: ROI SELECTION ---
    print("Draw a box strictly inside your hand. Press SPACE to confirm.")
    bbox = cv2.selectROI("1. Select Target Area", img, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("1. Select Target Area")
    x, y, w, h = bbox
    
    if w == 0 or h == 0:
        print("Selection failed. Exiting.")
        return

    roi_mask = np.zeros((new_height, new_width), dtype=np.uint8)
    cv2.rectangle(roi_mask, (x, y), (x+w, y+h), 255, -1)

    # --- 3. PREPROCESSING ---
    b, green_channel, r = cv2.split(img)
    hair_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    hairless_green = cv2.morphologyEx(green_channel, cv2.MORPH_CLOSE, hair_kernel)
    
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced_green = clahe.apply(hairless_green)
    blurred_green = cv2.GaussianBlur(enhanced_green, (5, 5), 0)

    # --- 4. MATH ---
    print("Calculating Frangi filter...")
    vein_probabilities = frangi(blurred_green, sigmas=np.arange(3, 11, 2), beta=0.7, gamma=25, black_ridges=True)
    
    # CRITICAL FIX: Mask the background before normalizing!
    vein_probabilities[roi_mask == 0] = 0
    vein_prob_normalized = cv2.normalize(vein_probabilities, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # --- 5. THE API CALL ---
    print("Saving temporary isolated Frangi map for the AI...")
    temp_filename = "temp_frangi_isolated.jpg"
    cv2.imwrite(temp_filename, vein_prob_normalized)

    print("Asking Gemini Vision for the optimal threshold...")
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                "You are an image processing assistant. Analyze this Frangi filter probability map. The bright white lines are veins, the dark background is noise. What is the optimal integer threshold value (between 5 and 50) to binarize this image and isolate the veins? Reply with ONLY the integer, nothing else.",
                open(temp_filename, "rb")
            ]
        )
        ai_suggestion = response.text.strip()
        ai_threshold = int(ai_suggestion)
        print(f"Gemini decided on threshold: {ai_threshold}")

    except Exception as e:
        print(f"API Call Failed ({e}). Defaulting to 12.")
        ai_threshold = 12
        
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

    # --- 6. APPLYING THE AI THRESHOLD ---
    _, binary_veins = cv2.threshold(vein_prob_normalized, ai_threshold, 255, cv2.THRESH_BINARY)
    
    # Contour Filtering to kill noise
    contours, _ = cv2.findContours(binary_veins, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(binary_veins)
    for contour in contours:
        if cv2.contourArea(contour) > 30: 
            cv2.drawContours(clean_mask, [contour], -1, 255, thickness=cv2.FILLED)
            
    bridge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    final_veins = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, bridge_kernel)

    # --- 7. VISUALIZATION ---
    overlay = img.copy()
    overlay[final_veins == 255] = [0, 255, 0] 

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title(f"AI Selected Threshold: {ai_threshold}")
    plt.imshow(final_veins, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Final Overlay")
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Run it!
process_veins_api_fixed('hand.jpeg')