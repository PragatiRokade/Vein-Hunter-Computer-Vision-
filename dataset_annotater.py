import cv2
import numpy as np
import glob
import os
from skimage.filters import frangi

class VeinAnnotator:
    def __init__(self):
        self.drawing = False
        self.mode = 'draw' # 'draw' or 'erase'
        self.brush_size = 5
        self.mask = None
        self.img = None
        self.display_img = None

    def auto_detect_veins(self, image):
        print("Running Auto-Detection (Frangi)...")
        # 1. Preprocess
        b, green_channel, r = cv2.split(image)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(green_channel)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

        # 2. Frangi
        vein_probs = frangi(blurred, sigmas=np.arange(3, 11, 2), beta=0.5, gamma=15, black_ridges=True)
        vein_prob_norm = cv2.normalize(vein_probs, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # 3. Threshold and Clean
        _, binary = cv2.threshold(vein_prob_norm, 15, 255, cv2.THRESH_BINARY)
        clean_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        return clean_mask

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.draw_circle(x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.draw_circle(x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.draw_circle(x, y)

    def draw_circle(self, x, y):
        color = 255 if self.mode == 'draw' else 0
        cv2.circle(self.mask, (x, y), self.brush_size, color, -1)
        self.update_display()

    def update_display(self):
        # Create a blended overlay for the user to see what they are doing
        overlay = self.img.copy()
        overlay[self.mask == 255] = [0, 255, 0] # Green veins
        
        # Blend the original image and the green overlay
        self.display_img = cv2.addWeighted(self.img, 0.7, overlay, 0.3, 0)
        
        # Add UI text
        tool = "PEN" if self.mode == 'draw' else "ERASER"
        cv2.putText(self.display_img, f"Tool: {tool} | Size: {self.brush_size}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Dataset Annotator", self.display_img)

def run_annotator(image_folder=".", mask_folder="masks"):
    if not os.path.exists(mask_folder):
        os.makedirs(mask_folder)

    # Grab all jpg/png files in the folder
    image_paths = glob.glob(os.path.join(image_folder, "*.jpg")) + glob.glob(os.path.join(image_folder, "*.png"))
    
    if not image_paths:
        print("No images found in the specified folder.")
        return

    annotator = VeinAnnotator()
    cv2.namedWindow("Dataset Annotator")
    cv2.setMouseCallback("Dataset Annotator", annotator.mouse_callback)

    for img_path in image_paths:
        # Load and resize for easier annotation (optional, but recommended for large phone pics)
        img = cv2.imread(img_path)
        height, width = img.shape[:2]
        new_width = 800
        new_height = int((new_width / width) * height)
        annotator.img = cv2.resize(img, (new_width, new_height))

        # Generate initial guess
        annotator.mask = annotator.auto_detect_veins(annotator.img)
        annotator.update_display()

        print(f"\nEditing: {os.path.basename(img_path)}")
        print("CONTROLS: [d] Pen | [e] Eraser | [+] Brush Up | [-] Brush Down | [s] Save & Next | [n] Skip")

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('d'):   # Draw
                annotator.mode = 'draw'
                annotator.update_display()
            elif key == ord('e'): # Erase
                annotator.mode = 'erase'
                annotator.update_display()
            elif key == ord('=') or key == ord('+'): # Increase brush
                annotator.brush_size = min(30, annotator.brush_size + 2)
                annotator.update_display()
            elif key == ord('-'): # Decrease brush
                annotator.brush_size = max(1, annotator.brush_size - 2)
                annotator.update_display()
            elif key == ord('s'): # Save Mask and move to next image
                filename = os.path.basename(img_path)
                mask_filename = os.path.splitext(filename)[0] + "_mask.png"
                mask_path = os.path.join(mask_folder, mask_filename)
                
                # Save the PURE binary mask (0 and 255) for the Deep Learning model
                cv2.imwrite(mask_path, annotator.mask)
                print(f"Saved: {mask_filename}")
                break
            elif key == ord('n'): # Skip to next image without saving
                print("Skipped.")
                break
            elif key == ord('q'): # Quit entirely
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print("Dataset annotation complete!")

# Run the tool on the current directory
run_annotator()