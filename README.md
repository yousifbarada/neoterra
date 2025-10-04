# 🌍 Satellite Image ChatBot

## Overview
A **stateful agentic pipeline** for working with satellite imagery (Sentinel-1 SAR and Sentinel-2-like RGB).  
It combines:
- A **deep-learning segmentation model** (oil-spill detector)  
- Preprocessing & postprocessing utilities  
- File download and metadata-extraction tools  
- An **LLM** (Google Gemini via `langchain-google-genai`) bound to those tools  
- A **LangGraph StateGraph agent** for streaming tool calls  
- A **Gradio UI** for chat + image preview + file download  

---

## 📑 Table of Contents
- [Quick Summary](#quick-summary)  
- [Security & Deployment Notes](#security--deployment-notes)  
- [Requirements](#requirements)  
- [Configuration](#configuration--environment-variables)  
- [Architecture & Components](#architecture--components)  
- [How to Run](#how-to-run)  
- [Usage Examples](#usage-examples)  
- [Output Conventions](#output-conventions--file-naming)  
- [Troubleshooting](#troubleshooting--common-pitfalls)  
- [Suggestions & Improvements](#suggestions--improvements)  
- [TODOs & Refactors](#todo--refactors)  
- [Appendix: Function Index](#appendix--function--file-index)  

---

## ⚡ Quick Summary
**What it does:**  
- Accepts user prompts (chat)  
- Downloads/reads satellite images  
- Runs preprocessing (equalization, denoising)  
- Runs an **oil-spill segmentation model**  
- Returns images/metadata  
- Explains results with an LLM  

**Core Tech:**  
`Keras/TensorFlow`, `rasterio`, `opencv-python`, `langchain`, `langchain-google-genai`, `langgraph`, `Gradio`

---

## 🔒 Security & Deployment Notes
- **API Key Management**:  
  Remove hard-coded API keys. Use environment variables or secret managers:
  ```bash
  export GOOGLE_API_KEY="your-real-key"
  ```
- **Model Path**:  
  `OIL_MODEL_PATH` points to a Kaggle dataset path. Replace or configure for local runs.
- **Workdir**:  
  Default = `/kaggle/working`. Use `./data` or `./outputs` for portability.  
- **URL Validation**:  
  Always validate download links before invoking downloads.

---

## 📦 Requirements
Install via `requirements.txt` or `pip`.

- Python 3.9+  
- `tensorflow`, `keras` (matching training version)  
- `rasterio`, `numpy`, `opencv-python`, `pillow`, `scipy`, `matplotlib`  
- `gradio`  
- `langchain`, `langchain-core`, `langchain-google-genai`  
- `langgraph`  
- `gdown` (optional, for Google Drive)  

**Example install:**
```bash
pip install rasterio numpy opencv-python pillow scipy matplotlib gradio gdown keras tensorflow langchain langchain-google-genai langgraph
```

---

## ⚙️ Configuration / Environment Variables
- `GOOGLE_API_KEY` — required for Google GenAI access  
- `OIL_MODEL_PATH` — path to segmentation model  
- `IMAGE_PATH` — optional default test image  
- `WORKDIR` — default = `/kaggle/working`  

---

## 🏗 Architecture & Components

### OilSpillDetector Class
- `__init__(model_path, model_input_shape)` → loads model  
- `load_image(path)` → reads Sentinel-1 VV band  
- `preprocess_image(vv)` → resize, normalize, equalize for visualization  
- `predict_oil_spill(...)` → inference + post-processing + mask output  
- `visualize_result(image, title)` → inline matplotlib display  
- `run_pipeline(...)` → full pipeline  

**Notes:**
- Supports both **softmax (multi-class)** & **sigmoid (binary)** outputs  
- Post-processing merges oil look-alikes into genuine detections  

---

### Tools (bound to LLM via `@tool`)
- `store_satellite_metadata(path)` → metadata JSON  
- `plot_sentinel1_image(path)` → pseudo-RGB preview  
- `plot_rgb_image(path)` → RGB preview  
- `download_image_from_url(url)` → HTTP/GDrive download  
- `oil_spill_segmentation(path)` → segmentation pipeline  
- `hist_equalize_sentinel1(path)` → VV/VH histogram equalization  
- `noise_filtering(path, filter_type, kernel_size)` → median/gaussian filter  
- `get_segmented_metadata(path)` → output image metadata  
- `equalization_Rgb(path)` → RGB equalization  

---

### Agent & Graph (LangGraph)
- **State** carried via `TypedDict`  
- LLM created with `ChatGoogleGenerativeAI(...)`  
- Tools bound with `.bind_tools()`  
- Graph flow:  
  1. `our_agent` → runs `model_call`  
  2. `should_continue` → check for tool calls  
  3. `tools` → executes requested tools  
  4. Loops back for reasoning  

---

## 🚀 How to Run

### Kaggle (Recommended)
1. Place `deeplabv3_model.keras` in a Kaggle dataset → set `OIL_MODEL_PATH`.  
2. Place input TIFF(s) in dataset or working directory.  
3. Run notebook → launch Gradio cell.  

### Local
```bash
export GOOGLE_API_KEY="your-key"
python app.py
```

---

## 💬 Usage Examples
```text
"/kaggle/working/sentinel1_image.tif display the image information"
"/kaggle/working/sentinel1_image.tif segment the image"
"Download this file: https://.../sentinel.tif and run segmentation"
"Apply median noise filter with kernel_size=5 on this image"
"Explain segmentation results in plain English"
```

---

## 📂 Output Conventions & File Naming
- Equalized: `<image>_equalized.tif`  
- Filtered: `<image>_<filter>_filtered.tif`  
- Preview: `<image>_plot.png`  
- Prediction mask: `<image>_prediction.png`  
- Downloads: `/kaggle/working` or `WORKDIR`  

---

## 🛠 Troubleshooting & Common Pitfalls
- **Model load errors** → check TensorFlow/Keras version  
- **Missing bands** → some Sentinel-1 only have VV  
- **Large TIFFs** → downsample or use `rasterio.windows`  
- **gdown missing** → install with `pip install gdown`  

---

## 💡 Suggestions & Improvements
- Config file (YAML/JSON) for paths & settings  
- Structured tool outputs (`{"path": "...", "type": "image"}`)  
- Gradio authentication for deployment  
- Unit tests (`pytest`) with mocked rasterio/model  
- Add **GeoJSON outputs** with `rasterio.transform`  
- Streaming downloads with progress  

---

## ✅ TODO & Refactors
- Remove hard-coded `GOOGLE_API_KEY` & `OIL_MODEL_PATH`  
- Consolidate duplicate helpers (e.g. `normalize_to_uint8`)  
- Improve file-path parsing & structured returns  
- Expand agent unit tests & error handling  
- Add TTA/uncertainty for segmentation  

---

## 📘 Appendix — Function / File Index
- **OilSpillDetector (class)**  
  - load, preprocess, predict, visualize, run_pipeline  
- **Tools**  
  - store_satellite_metadata  
  - plot_sentinel1_image  
  - plot_rgb_image  
  - download_image_from_url  
  - oil_spill_segmentation  
  - hist_equalize_sentinel1  
  - noise_filtering  
  - get_segmented_metadata  
  - equalization_Rgb  

---
