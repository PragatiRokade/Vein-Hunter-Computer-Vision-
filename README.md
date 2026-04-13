
# Vein Hunter: Real-Time Subcutaneous Vein Extraction

[](https://www.python.org/downloads/)
[](https://pytorch.org/get-started/locally/)
[](https://opensource.org/licenses/MIT)

**Vein Hunter** is an attention-guided deep learning pipeline designed to extract subcutaneous vascular structures from standard RGB imagery. By bypassing expensive Near-Infrared (NIR) hardware and computationally heavy classical filters, this project enables real-time vein detection for autonomous robotic venipuncture using ubiquitous optical sensors (like smartphones).

-----

## Key Features

  * **Attention-Guided Architecture:** Utilizes Convolutional Block Attention Modules (**CBAM**) to dynamically suppress epidermal noise (hair, wrinkles, glare) while highlighting subtle venous gradients.
  * **Real-Time Performance:** Purely neural inference path optimized for GPU, achieving high frame rates on live video streams.
  * **Mobile Integration:** Supports wireless live-streaming via iPhone/Android using IP Camera protocols.
  * **Dual-Phase Training:** Leverages transfer learning from high-contrast retinal datasets (DRIVE/FIVES) fine-tuned on a custom dataset of 116 human arm images.
  * **Robotic-Ready:** Outputs precise 2D (X, Y) spatial coordinates for initial acquisition systems in medical robotics.

-----

## Tech Stack

  * **Deep Learning:** PyTorch, Torchvision
  * **Computer Vision:** OpenCV, Albumentations
  * **Visualization:** Matplotlib, PIL
  * **Hardware Integration:** IP Camera (Mobile-to-PC Stream)
  * **Documentation:** LaTeX (NeurIPS Formatting)

-----

## Architecture

The model uses a **U-Net** backbone enhanced with **CBAM** (Channel and Spatial Attention) blocks in the decoder path. This allow the network to perform "semantic rejection" of surface artifacts that typically confuse classical filters like the Frangi vesselness filter.

> *[Insert your TikZ Architecture Diagram or a screenshot of Figure 2 here]*

-----

## Getting Started

### 1\. Installation

```bash
git clone https://github.com/pragatirokade/vein-hunter.git
cd vein-hunter
pip install -r requirements.txt
```

### 2\. Live Stream Setup (Mobile)

1.  Install **IP Camera Lite** on your iPhone/Android.
2.  Start the server and note the IPv4 address (e.g., `http://172.x.x.x:8081`).
3.  Update the `url` variable in `live_vein_hunter.py` with your credentials:
    ```python
    url = "http://admin:admin@your_ip_address:8081/video"
    ```

### 3\. Run Inference

```bash
python live_vein_hunter.py
```

-----

## Results

| Metric | Score |
| :--- | :--- |
| **Dice Coefficient (F1)** | 0.XX |
| **IoU** | 0.XX |
| **Inference Speed (FPS)** | \~XX FPS |

> *[Insert your `results_plot.jpg` here showing Original vs. Mask vs. Overlay]*

-----

## Academic Report

The full methodology, ablation studies, and comparative results are detailed in our technical report: **"Vein Hunter: An Attention-Guided Deep Learning Pipeline for Real-Time Subcutaneous Vein Extraction"** (Formatted for NeurIPS).

-----

## Contributors

  * **Pragati Rokade** (B23CM1055) 
  * **Radhika Agarwal** (B23ES1027)
  * **Mahi Upadhyay** (B23ES1022)

-----

## License

Distributed under the MIT License. See `LICENSE` for more information.

-----

