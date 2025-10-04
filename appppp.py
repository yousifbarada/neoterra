import re
import os
import numpy as np
from PIL import Image
import rasterio
import re
import os
import numpy as np
from PIL import Image
import rasterio
from langchain_core.messages import HumanMessage, AIMessage

import tempfile
from langgraph.checkpoint.memory import MemorySaver
import os
from PIL import Image
import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Annotated, Literal, Sequence, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.tools import StructuredTool, tool
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
import json
import rasterio
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
from scipy.ndimage import label as ndi_label
import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet34
from pathlib import Path
import json
import matplotlib.pyplot as plt
import warnings
# warnings.filterwarnings('ignore')
# langgraph imports
from langgraph.graph import StateGraph, START, END

# for optional web search (simple HTTP example)
import requests
from typing import Optional, Tuple

import requests
from pathlib import Path
from langchain_core.tools import tool
import subprocess
import mimetypes
from scipy.ndimage import median_filter, gaussian_filter
from google.api_core import exceptions as google_ex

import rasterio
import tempfile
import os
import rasterio
import numpy as np
from PIL import Image
from scipy.ndimage import median_filter, gaussian_filter
from langgraph.checkpoint.memory import MemorySaver
import os
from PIL import Image
import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Annotated, Literal, Sequence, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.tools import StructuredTool, tool
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
import json
import rasterio
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
from scipy.ndimage import label as ndi_label
import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet34
from pathlib import Path
import json
import matplotlib.pyplot as plt
import warnings
# warnings.filterwarnings('ignore')
# langgraph imports
from langgraph.graph import StateGraph, START, END

# for optional web search (simple HTTP example)
import requests
from typing import Optional, Tuple

import requests
from pathlib import Path
from langchain_core.tools import tool
import subprocess
import mimetypes

# Remove this line (it causes ImportError):
# import os
# from streamlit.secrets import GOOGLE_API_KEY

# Instead, use the standard way to read secrets from .streamlit/secrets.toml in Streamlit:
try:
    import streamlit as st
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except (ImportError, KeyError, AttributeError):
    # Fallback: try to read from environment or set a default (not recommended for production)
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

MODEL_INPUT_SIZE = (384, 384)
MODEL_INPUT_SHAPE = MODEL_INPUT_SIZE
#WATER_MODEL_PATH = '/kaggle/input/water-segmentation-3-deeplabv3-model/keras/default/1/deeplabv3_model.keras'
OIL_MODEL_PATH = "D:\\Narss Data\\Oil_Spill_Model_Version_2.keras"
# IMAGE_PATH = "/kaggle/working/sentinel1_image.tif"
cloud_segment_bath ='D:\\Narss Data\\cloud_segmentation_model.pth'
cloud_Removal_bath ='D:\\Narss Data\\modelPix.h5'
def plot_sentinel1_image(image_path: str) -> dict:
    """
    Tool to read a Sentinel-1 TIFF (VV/VH bands), normalize, save as PNG, and return info.
    Returns a single-channel (H,W) uint8 array in 'output_image' for grayscale plotting.
    """
    if not os.path.exists(image_path):
        return {"error": "File not found"}

    try:
        with rasterio.open(image_path) as src:
            vv = src.read(1)
            # fallback if VH missing is no longer used for preview; VV is main single-band

        # Normalize to 0-255 (single-channel)
        vv_norm = cv2.normalize(vv, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Save a preview image (optional) — save single-channel PNG
        save_path = os.path.splitext(image_path)[0] + "_plot.png"
        cv2.imwrite(save_path, vv_norm)

        return {
            "original_image": vv,         # raw float/uint array
            "output_image": vv_norm,      # **2D uint8** for grayscale display (H, W)
            "output_path": save_path
        }

    except Exception as e:
        return {"error": f"Error plotting Sentinel-1 TIFF: {e}"}



def plot_rgb_image(image_path: str) -> dict:
    """
    Tool to read an RGB TIFF image, normalize, save as PNG, and return info.
    """
    if not os.path.exists(image_path):
        return {"error": "File not found"}

    try:
        

        with rasterio.open(image_path) as src:
            original = src.read()  # shape: (bands, H, W)
            # Take first 3 bands for RGB
            if original.shape[0] >= 3:
                original = original[:3, :, :]
            else:  # grayscale -> repeat to 3 channels
                original = np.repeat(original, 3, axis=0)

            img = np.transpose(original, (1, 2, 0))  # H, W, C
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            img = img.astype(np.uint8)

        save_path = os.path.splitext(image_path)[0] + "_plot.png"
        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        return {
            "original_image": original,
            "output_image": img,
            "output_path": save_path
        }

    except Exception as e:
        return {"error": f"Error plotting RGB TIFF: {e}"}

@tool
def plot_any_image(image_path: str) -> dict:
    """
    Plot any image (TIFF, JPG, PNG).
    Handles single-band (grayscale) and multi-band (RGB).
    Returns:
        {
            "output_image": np.ndarray,
            "path": str
        }
    """
    try:
        ext = os.path.splitext(image_path)[1].lower()
        if ext in [".tif", ".tiff"]:
            with rasterio.open(image_path) as src:
                band_count = src.count
            if band_count == 1:
                # Sentinel-1 single-band
                return plot_sentinel1_image(image_path)
            elif band_count >= 3:
                # RGB or multi-band
                return plot_rgb_image(image_path)
            else:
                # 2-band: treat as single-band for preview
                return plot_sentinel1_image(image_path)
        else:
            # Non-TIFF: use OpenCV
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError("Could not load image")
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Normalize if needed
            if img.dtype != np.uint8 and img.max() > 1.0:
                img_min, img_max = img.min(), img.max()
                img = (img - img_min) / (img_max - img_min + 1e-6)
            return {"output_image": img, "path": image_path}
    except Exception as e:
        return {"output_image": None, "path": None, "error": str(e)}
class OilSpillDetector:
    def __init__(self, model_path, model_input_shape):
        """
        Initialize the detector with the oil spill model.
        model_path: path to the trained oil spill model (.h5)
        model_input_shape: tuple (height, width)
        """
        self.model_input_shape = model_input_shape
        self.model = load_model(model_path, compile=False)
        print("Oil spill model loaded.")

    def load_image(self, path):
        """
        Load VV band (band 1) from a raster file.
        Returns: vv (float32)
        """
        with rasterio.open(path) as src:
            vv = src.read(1).astype(np.float32)
        print(f"Image loaded: VV shape {vv.shape}")
        return vv

    def preprocess_image(self, vv):
        """
        Resize VV, normalize to 0..1, and make a 4D tensor (1,H,W,3).
        Also return an equalized uint8 image for visualization.
        """
        h, w = self.model_input_shape
        vv_resized = cv2.resize(vv, (w, h), interpolation=cv2.INTER_LINEAR)
        vv_rescaled = cv2.normalize(vv_resized, None, 0, 255, cv2.NORM_MINMAX)
        vv_uint8 = vv_rescaled.astype(np.uint8)
        vv_equalized = cv2.equalizeHist(vv_uint8)
        vv_normalized = vv_rescaled.astype(np.float32) / 255.0
        vv_input = np.stack([vv_normalized] * 3, axis=-1)
        vv_input = np.expand_dims(vv_input, axis=0).astype(np.float32)
        print(f"Preprocessing done — tensor shape {vv_input.shape}")
        return vv_input, vv_equalized

    def predict_oil_spill(self, input_tensor, water_mask=None, enable_water_mask=False):
        """
        Predict classes with the model, post-process look-alikes, and return colored RGB mask.
        """
        pred = self.model.predict(input_tensor)[0]
        
        if pred.ndim == 3 and pred.shape[-1] > 1:
            classes = np.argmax(pred, axis=-1).astype(np.uint8)
        else:
            classes = (pred[..., 0] > 0.5).astype(np.uint8)

        combined = (classes == 1) | (classes == 2)
        labeled_combined, n = ndi_label(combined)

        for region_id in range(1, n + 1):
            region_mask = (labeled_combined == region_id)
            if np.any(classes[region_mask] == 1):
                classes[(classes == 2) & region_mask] = 1

        colormap = {
            0: np.array([0, 0, 0], dtype=np.uint8),       # Background
            1: np.array([0, 255, 255], dtype=np.uint8),   # Oil
            2: np.array([255, 0, 0], dtype=np.uint8),     # Look-Alike
        }

        h, w = classes.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        for cls, color in colormap.items():
            colored[classes == cls] = color

        if enable_water_mask and water_mask is not None:
            wm = (water_mask > 0)
            if wm.shape != classes.shape:
                raise ValueError("water_mask shape does not match model output shape")
            colored[~wm] = 0

        print("Oil spill prediction completed.")
        return colored

    @staticmethod
    def visualize_result(image, title="Result"):
        """Display RGB or grayscale image."""
        plt.figure(figsize=(6, 6))
        if image.ndim == 3 and image.shape[2] == 3:
            plt.imshow(image)
        else:
            plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.show()
        return image

    def load_sar_composite(self, path):
        """
        Load VV and VH bands (if available) and make a pseudo-RGB composite.
        """
        with rasterio.open(path) as src:
            vv = src.read(1).astype(np.float32)
            vh = src.read(2).astype(np.float32) if src.count > 1 else vv
    
            pseudo_rgb = np.dstack([
                cv2.normalize(vv, None, 0, 255, cv2.NORM_MINMAX),
                cv2.normalize(vh, None, 0, 255, cv2.NORM_MINMAX),
                cv2.normalize((vv + vh) / 2, None, 0, 255, cv2.NORM_MINMAX)
            ]).astype(np.uint8)
    
        print(f"SAR pseudo-RGB image created: {pseudo_rgb.shape}")
        return pseudo_rgb

    def run_pipeline(self, image_path, water_mask=None, enable_water_mask=False):
        """ 
        Full pipeline: load image -> preprocess -> predict -> return prediction.
        Returns the final RGB prediction image.
        """
        vv = self.load_image(image_path)
        vv_input, _ = self.preprocess_image(vv)
        colored_mask = self.predict_oil_spill(vv_input, water_mask, enable_water_mask)
        return colored_mask



detector = OilSpillDetector(model_path=OIL_MODEL_PATH, model_input_shape=MODEL_INPUT_SIZE)
def analyze_Oil_segmentation( mask):
        """
        Analyze a colored segmentation mask:
        - Count pixels in each class
        - Compute percentage relative to total image area
        Returns a dict with stats
        """
        # RGB codes used in segmentation
        colormap = {
            "Background": np.array([0, 0, 0], dtype=np.uint8),
            "Oil": np.array([0, 255, 255], dtype=np.uint8),
            "Look-Alike": np.array([255, 0, 0], dtype=np.uint8),
        }

        h, w, _ = mask.shape
        total_pixels = h * w
        stats = {}

        for name, color in colormap.items():
            # Mask for current class
            class_mask = np.all(mask == color, axis=-1)
            count = np.sum(class_mask)
            percent = (count / total_pixels) * 100
            stats[name] = {
                "pixels": int(count),
                "percent": round(percent, 2)
            }

        return stats

@tool
def oil_spill_segmentation(image_path: str) -> dict:
    """
    Run oil spill segmentation  and return both original and output images this works only on Sent 1 imgages .
    the input image must have 2 bands
    Returns :
        {
            "original_image": np.ndarray,
            "output_image": np.ndarray,
            "output_path": str
        }
        and analyis it wiht analyze_Oil_segmentation tool
    """
    # Run pipeline
    colored_mask = detector.run_pipeline(image_path, water_mask=None, enable_water_mask=False)
    
    # Load original for reference
    with rasterio.open(image_path) as src:
        bands = [src.read(i+1) for i in range(min(3, src.count))]
        
        sar_rgb = detector.load_sar_composite(image_path)

    # Save output
    pred_path = image_path + "_prediction.png"
    cv2.imwrite(pred_path, cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
    if colored_mask is not None :
        status = analyze_Oil_segmentation(colored_mask)
    else :
        status= None
    
    return {
        "original_image": sar_rgb,
        "output_image": colored_mask,
        "output_path": pred_path,
        "status": status
    }
# ==========================================
# STEP 1: LOAD ALL MODEL DEFINITIONS
# ==========================================

class CloudDataset(Dataset):
    def __init__(self, r_dir, g_dir, b_dir, nir_dir, gt_dir, pytorch=True):
        super().__init__()
        
        # Convert to Path objects if they're strings
        self.r_dir = Path(r_dir)
        self.g_dir = Path(g_dir)
        self.b_dir = Path(b_dir)
        self.nir_dir = Path(nir_dir)
        self.gt_dir = Path(gt_dir)
        
        # Loop through the files in the red folder and combine, into a dictionary, the other bands
        self.files = [self.combine_files(f, self.g_dir, self.b_dir, self.nir_dir, self.gt_dir) 
                     for f in self.r_dir.iterdir() if not f.is_dir()]
        self.pytorch = pytorch
        
    def combine_files(self, r_file: Path, g_dir, b_dir, nir_dir, gt_dir):
        # Combine file paths for different spectral bands into a dictionary
        files = {'red': r_file, 
                 'green': g_dir / r_file.name.replace('red', 'green'),
                 'blue': b_dir / r_file.name.replace('red', 'blue'), 
                 'nir': nir_dir / r_file.name.replace('red', 'nir'),
                 'gt': gt_dir / r_file.name.replace('red', 'gt')}
        return files
                                       
    def __len__(self):
        return len(self.files)
     
    def open_as_array(self, idx, invert=False, include_nir=False):
        # Open image files as arrays, optionally including NIR channel
        raw_rgb = np.stack([np.array(Image.open(self.files[idx]['red'])),
                            np.array(Image.open(self.files[idx]['green'])),
                            np.array(Image.open(self.files[idx]['blue'])),
                           ], axis=2)
    
        if include_nir:
            nir = np.expand_dims(np.array(Image.open(self.files[idx]['nir'])), 2)
            raw_rgb = np.concatenate([raw_rgb, nir], axis=2)
    
        if invert:
            raw_rgb = raw_rgb.transpose((2, 0, 1))
    
        # Normalize pixel values
        return (raw_rgb / np.iinfo(raw_rgb.dtype).max)
    
    def open_mask(self, idx, add_dims=False):
        # Open ground truth mask as an array
        raw_mask = np.array(Image.open(self.files[idx]['gt']))
        raw_mask = np.where(raw_mask == 255, 1, 0)
        
        return np.expand_dims(raw_mask, 0) if add_dims else raw_mask
    
    def __getitem__(self, idx):
        # Get an item from the dataset (image and mask)
        x = torch.tensor(self.open_as_array(idx, invert=self.pytorch, include_nir=True), dtype=torch.float32)
        y = torch.tensor(self.open_mask(idx, add_dims=False), dtype=torch.int64)
        
        return x, y
    
    def open_as_pil(self, idx):
        # Open an image as a PIL image
        arr = 256 * self.open_as_array(idx)
        return Image.fromarray(arr.astype(np.uint8), 'RGB')
    
    def __repr__(self):
        s = 'Dataset class with {} files'.format(self.__len__())
        return s

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   

class UNet(nn.Module):
    def __init__(self, n_class, in_channels=4):
        super().__init__()
                
        self.dconv_down1 = double_conv(in_channels, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, n_class, 1)
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       
        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out

class UNET(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Modify first layer of ResNet34 to accept custom number of channels
        base_model = resnet34(pretrained=False)
        base_model.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.base_layers = list(base_model.children())
        self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer2 = self.base_layers[5]
        self.layer3 = self.base_layers[6]
        self.layer4 = self.base_layers[7]
        
        self.upconv4 = self.expand_block(512, 256)
        self.upconv3 = self.expand_block(256*2, 128)
        self.upconv2 = self.expand_block(128*2, 64)
        self.upconv1 = self.expand_block(64*2, 64)
        self.upconv0 = self.expand_block(64*2, out_channels)
        
    def forward(self, x):
        # Contracting Path
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        # Expansive Path
        upconv4 = self.upconv4(layer4)
        upconv3 = self.upconv3(torch.cat([upconv4, layer3], 1))
        upconv2 = self.upconv2(torch.cat([upconv3, layer2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, layer1], 1))
        upconv0 = self.upconv0(torch.cat([upconv1, layer0], 1))
        return upconv0
        
    def expand_block(self, in_channels, out_channels):
        expand = nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )
        return expand

def acc_metric(predb, yb):
    return (predb.argmax(dim=1) == yb.cuda()).float().mean()

def batch_to_img(xb, idx):
    img = np.array(xb[idx,0:3])
    return img.transpose((1,2,0))

def predb_to_mask(predb, idx):
    p = torch.functional.F.softmax(predb[idx], 0)
    return p.argmax(0).cpu()

# ==========================================
# STEP 2: LOAD THE MODEL SAFELY
# ==========================================

def load_model_safely(model_path=cloud_segment_bath):
    """
    Load model with error handling
    """
    try:
        # Check if CUDA is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Create model instance
        model = UNET(in_channels=4, out_channels=2)
        
        # Load checkpoint
        if device.type == 'cuda':
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(model_path, map_location='cpu')
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        print("✅ Model loaded successfully!")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model, checkpoint, device
        
    except FileNotFoundError:
        print("❌ Error: Model file not found. Make sure 'cloud_segmentation_model.pth' is in the current directory.")
        return None, None, None
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return None, None, None

# ==========================================
# STEP 3: PREDICTION FUNCTIONS
# ==========================================
def normalize_band_array(arr):
    """
    Normalize a single-band array to [0,1].
    Works for int or float arrays.
    """
    if np.issubdtype(arr.dtype, np.integer):
        return arr / np.iinfo(arr.dtype).max
    elif np.issubdtype(arr.dtype, np.floating):
        # Assume values already between 0-1 or 0-10000, normalize if >1
        return arr / arr.max()
    else:
        raise ValueError(f"Unsupported dtype: {arr.dtype}")

def predict_from_image_paths(model, red_path, green_path, blue_path, nir_path, device):
    """
    Make prediction from individual image paths
    """
    try:
        model.eval()
        with torch.no_grad():
            # Load and process images
            red = np.array(Image.open(red_path))
            green = np.array(Image.open(green_path))
            blue = np.array(Image.open(blue_path))
            nir = np.array(Image.open(nir_path))
            
            # Stack channels (R, G, B, NIR)
            img_array = np.stack([red, green, blue, nir], axis=2)
            
            # Normalize
            img_array = np.stack([
                normalize_band_array(red),
                normalize_band_array(green),
                normalize_band_array(blue),
                normalize_band_array(nir)
            ], axis=2)

            # Convert to tensor and add batch dimension
            img_tensor = torch.tensor(img_array.transpose((2, 0, 1)), dtype=torch.float32)
            img_batch = img_tensor.unsqueeze(0).to(device)
            
            # Make prediction
            prediction = model(img_batch)
            predicted_mask = torch.softmax(prediction, dim=1).argmax(dim=1).squeeze().cpu().numpy()
            
            # Get RGB for visualization
            original_img = img_array[:, :, :3]  # Just RGB channels
            
            return {
                'original_image': original_img,
                'predicted_mask': predicted_mask,
                'prediction_tensor': prediction.cpu()
            }
    except Exception as e:
        print(f"❌ Error during prediction from paths: {str(e)}")
        return None
        
def predict_from_multiband_image(model, image_path, device):
    """
    Takes a multi-band image (R, G, B, NIR), splits it into separate temporary band files,
    and predicts using `predict_from_image_paths`.
    """
    try:
        # Open multi-band image
        with rasterio.open(image_path) as src:
            if src.count < 4:
                raise ValueError("Image must have at least 4 bands (R, G, B, NIR).")
            
            # Read individual bands
            red = src.read(1)
            green = src.read(2)
            blue = src.read(3)
            nir = src.read(4)
            
            # Create temporary files for each band
            temp_dir = tempfile.mkdtemp()
            red_path = os.path.join(temp_dir, "red.tif")
            green_path = os.path.join(temp_dir, "green.tif")
            blue_path = os.path.join(temp_dir, "blue.tif")
            nir_path = os.path.join(temp_dir, "nir.tif")
            
            # Save bands as separate TIFFs
            for arr, path in zip([red, green, blue, nir], [red_path, green_path, blue_path, nir_path]):
                with rasterio.open(
                    path,
                    'w',
                    driver='GTiff',
                    height=arr.shape[0],
                    width=arr.shape[1],
                    count=1,
                    dtype=arr.dtype
                ) as dst:
                    dst.write(arr, 1)
            
            # Predict using existing function
            result = predict_from_image_paths(model, red_path, green_path, blue_path, nir_path, device)
            
            # Cleanup temporary files
            for path in [red_path, green_path, blue_path, nir_path]:
                os.remove(path)
            os.rmdir(temp_dir)
            
            return result
    
    except Exception as e:
        print(f"❌ Error in predicting from multi-band image: {str(e)}")
        return None


def visualize_prediction(result, title="Prediction Results"):
    """
    Visualize prediction results
    """
    if result is None:
        print("❌ No results to visualize")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(result['original_image'])
    axes[0].set_title('Original Image (RGB)')
    axes[0].axis('off')
    
    # Predicted mask
    axes[1].imshow(result['predicted_mask'], cmap='gray')
    axes[1].set_title('Predicted Cloud Mask')
    axes[1].axis('off')
    
    # True mask (if available)
    if 'true_mask' in result:
        axes[2].imshow(result['true_mask'], cmap='gray')
        axes[2].set_title('True Cloud Mask')
    else:
        axes[2].text(0.5, 0.5, 'No Ground Truth\nAvailable', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=axes[2].transAxes, fontsize=12)
        axes[2].set_title('Ground Truth')
    axes[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# ==========================================
# STEP 4: MAIN LOADING AND PREDICTION SCRIPT
# ==========================================

def load():
    print("🚀 Starting model loading and prediction...")
    
    # Load model
    model, checkpoint, device = load_model_safely()
    if model is None:
        return
    
    # Print training history if available
    if checkpoint:
        print(f"\n📊 Training History:")
        print(f"Final training loss: {checkpoint['train_loss'][-1]:.4f}")
        print(f"Final validation loss: {checkpoint['valid_loss'][-1]:.4f}")
        print(f"Final accuracy: {checkpoint['overall_acc'][-1]:.4f}")
    
    return model, device

#

model, device = load()
    
# ==========================================
# ADDITIONAL UTILITY FUNCTIONS
# ==========================================

def calculate_accuracy(predicted_mask, true_mask):
                    correct = np.sum(predicted_mask == true_mask)
                    total = predicted_mask.size
                    return correct / total

def save_prediction(predicted_mask, output_path):
        """Save predicted mask as image"""
        # Convert to 0-255 range
        mask_img = (predicted_mask * 255).astype(np.uint8)
        Image.fromarray(mask_img).save(output_path)
        print(f"Prediction saved to: {output_path}")

def get_model_info(model):
        """Get information about the loaded model"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
        print(f"📋 Model Information:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Model size: {total_params * 4 / 1024 / 1024:.2f} MB")


def save_prediction(predicted_mask, output_path):
    # Convert to 0-255 range
    mask_img = (predicted_mask * 255).astype(np.uint8)

    # Ensure we save in working directory
    base_name = os.path.basename(output_path)
    safe_path = os.path.join("/kaggle/working", base_name)

    Image.fromarray(mask_img).save(safe_path)
    print(f"Prediction saved to: {safe_path}")
    return safe_path
def analyze_cloud_segmentation(mask: np.ndarray) -> dict:
    """
    Analyze a cloud segmentation mask:
    - Supports grayscale (1 channel) or RGB masks
    - Count pixels for Cloud vs Non-Cloud
    - Compute percentage relative to total image area
    Returns a dict with stats
    """
    stats = {}
    
    # لو grayscale (H, W)
    if mask.ndim == 2:
        h, w = mask.shape
        total_pixels = h * w
        cloud_pixels = np.sum(mask > 0)   # أي قيمة >0 = Cloud
        non_cloud_pixels = total_pixels - cloud_pixels

        stats["Non-Cloud"] = {
            "pixels": int(non_cloud_pixels),
            "percent": round((non_cloud_pixels / total_pixels) * 100, 2)
        }
        stats["Cloud"] = {
            "pixels": int(cloud_pixels),
            "percent": round((cloud_pixels / total_pixels) * 100, 2)
        }
        stats["total_pixels"] = int(total_pixels)

    # لو RGB (H, W, 3)
    elif mask.ndim == 3 and mask.shape[2] == 3:
        colormap = {
            "Non-Cloud": np.array([0, 0, 0], dtype=np.uint8),       # black
            "Cloud": np.array([255, 255, 255], dtype=np.uint8)      # white
        }
        h, w, _ = mask.shape
        total_pixels = h * w

        for name, color in colormap.items():
            class_mask = np.all(mask == color, axis=-1)
            count = np.sum(class_mask)
            percent = (count / total_pixels) * 100 if total_pixels > 0 else 0
            stats[name] = {"pixels": int(count), "percent": round(percent, 2)}

        stats["total_pixels"] = int(total_pixels)

    else:
        raise ValueError(f"Unsupported mask shape: {mask.shape}")

    return stats
@tool
def cloud_segmentation_tool(image_path: str) -> dict:
    """
    Run cloud segmentation prediction image paths.
    this tool work for RGB images 
    Args:
          img path
        and return original + prediction images.
    Returns:
        {
            "original_image": np.ndarray,
            "output_image": np.ndarray,
            "output_path": str
        }
    """
    result = predict_from_multiband_image(model, image_path, device)
    if result is None:
        return      {
        "original_image": None,
        "output_image": None,
        "output_path": None
    }

    # Save output mask in /kaggle/working/
    output_path = save_prediction(result["predicted_mask"], os.path.basename(image_path) + "_cloudmask.png")
    if result["original_image"] is not None :
            stats= analyze_cloud_segmentation(result["predicted_mask"])
    else :
            state = "error in predicting the mask"
    return {
        "original_image": result["original_image"],
        "output_image": result["predicted_mask"],
        "output_path": output_path,
        "status":stats
    }
# ----------------- Add these helpers + generator tool -----------------
import os
from tensorflow.keras.models import load_model
from pathlib import Path
import numpy as np
def normalize_to_uint8(arr):
    """
    Normalize float array in [0,1] or arbitrary range to uint8 [0,255].
    """
    arr = np.nan_to_num(arr).astype(np.float32)
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    return (arr * 255).clip(0, 255).astype(np.uint8)

def scale_model_output_to_01(arr):
    """
    Scale model output to [0, 1].
    
    Handles both:
      - Outputs in [-1, 1] (common for tanh GANs)
      - Arbitrary ranges (min-max normalization)
    
    Args:
        arr (np.ndarray): model output (H, W, C) or (H, W)
    
    Returns:
        np.ndarray: normalized output in [0, 1]
    """
    arr = arr.astype(np.float32)

    # Case 1: If values look like they’re in [-1, 1], just map directly
    if arr.min() >= -1.0 and arr.max() <= 1.0:
        return (arr + 1.0) / 2.0

    # Case 2: Generic min-max scaling
    min_val, max_val = arr.min(), arr.max()
    if max_val > min_val:
        return (arr - min_val) / (max_val - min_val)
    else:
        return np.zeros_like(arr, dtype=np.float32)  # flat image

def find_h5_file(base_path="/kaggle/input", name_hint=None):
    """
    Search for a .h5 file under base_path and return full path.
    If name_hint provided, prefer files that contain it.
    """
    base = Path(base_path)
    if not base.exists():
        raise FileNotFoundError(f"{base_path} does not exist")
    candidates = []
    for p in base.rglob("*.h5"):
        candidates.append(str(p))
    if not candidates:
        raise FileNotFoundError(f"No .h5 model found under {base_path}")
    if name_hint:
        for c in candidates:
            if name_hint in os.path.basename(c):
                return c
    # fallback: return first
    return candidates[0]

# Reuse the robust Sentinel2GANFixed from the assistant's fixes (shortened)
class Sentinel2GANFixed:
    def __init__(self, model_path= cloud_Removal_bath, sentinel1_dir: str = None, target_size=(256,256)):
        print("[INFO] Loading generator model:", model_path)
        self.model = load_model(model_path, compile=False)
        self.sentinel1_dir = sentinel1_dir
        self.target_size = target_size
        print("[INFO] Model loaded. Input shape:", getattr(self.model, "input_shape", None),
              "Output shape:", getattr(self.model, "output_shape", None))

    def load_image(self, image_path: str, bands=(3,2,1)):
        with rasterio.open(image_path) as src:
            profile = src.profile.copy()
            # defensive channel read
            channels = []
            for b in bands:
                if b <= src.count:
                    channels.append(src.read(b))
                else:
                    channels.append(src.read(1))
            landsat = np.stack(channels, axis=-1).astype(np.float32)  # H,W,C
        vv = None
        if self.sentinel1_dir:
            vv_path = os.path.join(self.sentinel1_dir, os.path.basename(image_path))
            if os.path.exists(vv_path):
                try:
                    with rasterio.open(vv_path) as s:
                        vv = s.read(1).astype(np.float32)
                except Exception as e:
                    print("[WARN] could not read sentinel1 vv:", e)
                    vv = None
        if vv is None:
            vv = np.zeros(landsat.shape[:2], dtype=np.float32)
        combined = np.concatenate([landsat, vv[..., np.newaxis]], axis=-1)  # H,W,4
        return combined, profile
    

    
    def preprocess(self, arr):
        """
        Preprocess an input array for GAN inference.
        - Replace NaNs/Infs with 0
        - Resize to target size (H, W)
        - Normalize channels to [-1, 1]
        - Add batch dimension
        Returns:
          tensor (1, H, W, C), arr_tanh (H, W, C)
        """
        # 1. Fix invalid values
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    
        # 2. Resize to target (expects self.target_size = (H, W))
        h, w = self.target_size
        arr_resized = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
    
        # Ensure channel dimension
        if arr_resized.ndim == 2:
            arr_resized = np.expand_dims(arr_resized, axis=-1)
    
        # 3. Normalize each channel to [-1, 1]
        arr_tanh = arr_resized.astype(np.float32)
        for c in range(arr_tanh.shape[-1]):
            ch = arr_tanh[..., c]
            min_val, max_val = ch.min(), ch.max()
            if max_val > min_val:
                arr_tanh[..., c] = 2.0 * (ch - min_val) / (max_val - min_val) - 1.0
            else:
                arr_tanh[..., c] = 0.0  # flat channel
    
        # 4. Add batch dimension → (1, H, W, C)
        tensor = np.expand_dims(arr_tanh, axis=0)
    
        return tensor, arr_tanh


    def predict(self, tensor):
        raw = self.model.predict(tensor)
        return np.array(raw)

    def postprocess(self, raw):
        if raw.ndim == 4 and raw.shape[0] == 1:
            out = raw[0]
        else:
            out = raw
        out_01 = scale_model_output_to_01(out)
        C = out_01.shape[-1]
        if C >= 6:
            rgb = np.stack([out_01[..., 2], out_01[..., 1], out_01[..., 0]], axis=-1)
        elif C >= 3:
            rgb = out_01[..., :3]
        else:
            rgb = np.repeat(out_01[..., :1], 3, axis=-1)
        return out_01.astype(np.float32), np.clip(rgb, 0.0, 1.0)
wrapper = Sentinel2GANFixed()

@tool
def CloudRemoval(image_path: str) -> dict:
    """
    Run cloud removal Pix2Pix generator on an input (RGB) TIFF  and save results in /kaggle/working.

    Args:
        image_path (str): Path to input GeoTIFF.

    Returns:
        dict: {
            "original_image": np.ndarray (RGB preview of input),
            "output_image": np.ndarray (RGB preview after cloud removal),
            "output_path": str (path to saved GeoTIFF)
        }
    """

    try:
        if not os.path.exists(image_path):
            return json.dumps({"error":"image_path not found", "path": None})
        # find model if not provided
       
        # instantiate wrapper & run
        raw_img, profile = wrapper.load_image(image_path)
        tensor, arr_tanh = wrapper.preprocess(raw_img)
        raw_out = wrapper.predict(tensor)
        out_01, rgb_01 = wrapper.postprocess(raw_out)
        base = os.path.splitext(os.path.basename(image_path))[0]

        base_out_path = os.path.join("/kaggle/working", base)

        preview_path = base_out_path + "_generated_preview.png"
        preview_uint8 = normalize_to_uint8(np.hstack([np.clip((rgb_01),0,1)]))
        cv2.imwrite(preview_path, cv2.cvtColor(preview_uint8, cv2.COLOR_RGB2BGR))        

        preview_uint8 = normalize_to_uint8(np.hstack([np.clip((rgb_01),0,1)]))
        cv2.imwrite(preview_path, cv2.cvtColor(preview_uint8, cv2.COLOR_RGB2BGR))
        geotiff_path = base_out_path + "_generated.tif"
        arr_uint16 = (out_01 * 65535.0).astype(np.uint16)
        H, W, C = arr_uint16.shape
        new_profile = profile.copy()
        new_profile.update({'count': C, 'dtype': 'uint16', 'height': H, 'width': W})
        with rasterio.open(geotiff_path, 'w', **new_profile) as dst:
            for b in range(C):
                dst.write(arr_uint16[..., b], b+1)
        return {
            "original_image": plot_any_image(image_path)['output_image'],
            "output_image": preview_uint8,
            "output_path": geotiff_path
        }
    except Exception as e:
        return {
            "original_image": None,
            "output_image": None,
            "output_path": None,
        }
# ----------------- End added block -----------------
@tool
def get_satellite_metadata(image_path: str) -> str:
    """Extract and return metadata from a satellite image as a JSON string."""
    with rasterio.open(image_path) as src:
        meta = {
            "driver": src.driver,
            "width": src.width,
            "height": src.height,
            "band_count": src.count,
            "crs": str(src.crs),
            "bounds": src.bounds._asdict(),
            "transform": list(src.transform),
            "dtypes": src.dtypes,
        }
        tags = src.tags()
        if tags:
            meta.update(tags)
    return json.dumps(meta, indent=2)
def normalize_to_uint8(channel: np.ndarray) -> np.ndarray:
    """Normalize any numeric array to 0-255 uint8."""
    channel = channel.astype(np.float32)
    return np.uint8(255 * (channel - channel.min()) / (channel.max() - channel.min()))

@tool 
def equalization(img_path: str) -> dict:
    """
    Apply histogram equalization on any satellite image (1+ bands).
    Returns original + equalized image (stacked for available bands).
    """
    with rasterio.open(img_path) as src:
        count = src.count
        profile = src.profile.copy()
        bands = [src.read(i+1) for i in range(count)]

    # Normalize and equalize each band
    orig_bands = [normalize_to_uint8(b) for b in bands]
    eq_bands = [cv2.equalizeHist(band) for band in orig_bands]

    # Stack for display
    original = np.dstack(orig_bands) if count > 1 else orig_bands[0]
    equalized = np.dstack(eq_bands) if count > 1 else eq_bands[0]

    # Save
    save_path = os.path.splitext(img_path)[0] + "_equalized.tif"
    profile.update(dtype=rasterio.uint8, count=count)
    with rasterio.open(save_path, "w", **profile) as dst:
        for i, b in enumerate(eq_bands):
            dst.write(b, i+1)

    return {
        "original_image": original,
        "output_image": equalized,
        "output_path": save_path
    }
@tool 
def noise_filtering(image_path: str, filter_type: str = "median", kind: str = "sentinel1", kernel_size: int = 3, sigma: float = 1.0):
    """
    Apply noise filtering (Median or Gaussian) on Sentinel-1 or Sentinel-2 images.

    Returns:
        {
            "original_image": np.ndarray,
            "filtered_image": np.ndarray,
            "output_path": str
        }
    """
    with rasterio.open(image_path) as src:
        bands = src.read()
        profile = src.profile.copy()

    # Save original preview
    if kind == "sentinel1":
        orig_band = bands[0]
        original = (orig_band - orig_band.min()) / (orig_band.max() - orig_band.min() + 1e-6)
    elif kind == "sentinel2" and bands.shape[0] >= 3:
        rgb = np.stack([bands[0], bands[1], bands[2]], axis=-1)
        original = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
    elif bands.shape[0] == 2:
        # 2-band: show first band as grayscale
        orig_band = bands[0]
        original = (orig_band - orig_band.min()) / (orig_band.max() - orig_band.min() + 1e-6)
    else:
        raise ValueError("Invalid kind or not enough bands.")

    # Apply filtering
    filtered_bands = []
    for band in bands:
        if filter_type == "median":
            filtered = median_filter(band, size=kernel_size)
        elif filter_type == "gaussian":
            filtered = gaussian_filter(band, sigma=sigma)
        else:
            raise ValueError("filter_type must be 'median' or 'gaussian'")
        filtered_bands.append(filtered)

    filtered_bands = np.stack(filtered_bands)

    # Save filtered file
    base_name = os.path.basename(image_path).replace(".tif", f"_{filter_type}_filtered.tif")
    output_path = os.path.join("/kaggle/working", base_name)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(filtered_bands)

    # Build preview for filtered
    if kind == "sentinel1" or bands.shape[0] == 2:
        # For 2-band, show first band as grayscale
        f_band = filtered_bands[0]
        filtered = (f_band - f_band.min()) / (f_band.max() - f_band.min() + 1e-6)
    elif kind == "sentinel2" and filtered_bands.shape[0] >= 3:
        f_rgb = np.stack([filtered_bands[0], filtered_bands[1], filtered_bands[2]], axis=-1)
        filtered = (f_rgb - f_rgb.min()) / (f_rgb.max() - f_rgb.min() + 1e-6)
    else:
        # fallback: show first band
        f_band = filtered_bands[0]
        filtered = (f_band - f_band.min()) / (f_band.max() - f_band.min() + 1e-6)

    return {
        "original_image": original,
        "filtered_image": filtered,
        "output_path": output_path
    }
@tool
def download_image_from_url(file_url: str, outputname: str = "downloaded_file") -> str:
    """
    Download a file from a URL (including Google Drive links) and save it locally.
    Automatically detects file extension and preserves original filename if possible.
    Returns the path to the saved file.
    """
    save_dir = Path("/kaggle/working")
    save_dir.mkdir(exist_ok=True)

    try:
        # Detect Google Drive link
        if "drive.google.com" in file_url:
            # Use user-defined outputname or fallback
            filename = outputname
            if not filename:
                filename = "downloaded_file"

            # Ensure extension (default to .tif)
            if "." not in filename:
                filename += ".tif"

            save_path = save_dir / filename

            # Download using gdown
            try:
                subprocess.run(
                    ["gdown", "--fuzzy", file_url, "--output", str(save_path)],
                    check=True
                )
                return str(save_path)
            except FileNotFoundError:
                return "Error: gdown not installed. Please install gdown to download from Google Drive."
            except subprocess.CalledProcessError as e:
                return f"Error downloading from Google Drive: {e}"

        # Normal HTTP/HTTPS link
        response = requests.get(file_url, stream=True)
        response.raise_for_status()

        # Try to extract filename from URL
        filename = file_url.split("/")[-1].split("?")[0]
        
        # If user provides outputname, use it
        if outputname:
            filename = outputname
            # Add extension if missing
            if "." not in filename:
                content_type = response.headers.get("Content-Type", "")
                ext = mimetypes.guess_extension(content_type.split(";")[0].strip())
                filename += ext or ".bin"

        # If URL has no filename and outputname not provided
        if not filename:
            content_type = response.headers.get("Content-Type", "")
            ext = mimetypes.guess_extension(content_type.split(";")[0].strip())
            filename = f"downloaded_file{ext or '.bin'}"

        save_path = save_dir / filename

        # Save file locally
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        
        return {
        "original_image": None,
        "output_image": None,
        "output_path": save_path}

    except Exception as e:
        return f"Error downloading file: {e}"
@tool
def get_segmented_metadata(segmented_path: str) -> str:
    """
    Extract metadata from a segmented image and return it as a JSON string.
    Metadata includes filename, format, size, mode, file size in KB, and band count.
    """
    if not segmented_path or not os.path.exists(segmented_path):
        return "No segmented image available to extract metadata."

    try:
        img = Image.open(segmented_path)

        metadata = {
            "filename": os.path.basename(segmented_path),
            "format": img.format,
            "size": img.size,  # (width, height)
            "mode": img.mode,
            "file_size_kb": round(os.path.getsize(segmented_path) / 1024, 2),
            "bands": 1
        }

        return json.dumps(metadata, indent=2)

    except Exception as e:
        return f"Error extracting segmented image metadata: {e}"
tools = [
    get_satellite_metadata,
    oil_spill_segmentation,
    get_segmented_metadata,
    download_image_from_url,
    plot_any_image,
    equalization,
    noise_filtering,
    cloud_segmentation_tool,
    CloudRemoval,
]    
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1).bind_tools(tools)
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    original_image: Optional[np.ndarray]  # store original image (RGB)
    output_image: Optional[np.ndarray]    # store output/processed image
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

def model_call(state: State) -> State:
    system_prompt = SystemMessage(
        content=(
            "Your name is (NeoTerra) you are agnt that helps us in genral talkes  and specilized in Satalite Images . "
            "Answer any general questions helpfully, clearly, and concisely. "
            "In addition, you are specialized in satellite imagery tasks "
            "(such as image metadata, segmentation, preprocessing, visualization, "
            "and geospatial analysis). "
            "You Made by Alaa Wael, Yousif Edris and Nadeen hazem  be thankfull for us "
            "Always prioritize accurate, useful answers."
        )
    )

    messages_for_invoke = [system_prompt]
    for m in state.get("messages", []):
        if isinstance(m, BaseMessage):  # already a LangChain message
            messages_for_invoke.append(m)
        elif isinstance(m, (list, tuple)) and len(m) >= 2:
            role, text = m[0], str(m[1])
            if role == "user":
                messages_for_invoke.append(HumanMessage(content=text))
            elif role == "assistant":
                messages_for_invoke.append(AIMessage(content=text))
            else:
                messages_for_invoke.append(HumanMessage(content=text))  # fallback
        elif isinstance(m, dict) and "role" in m and "content" in m:
            role, text = m["role"], str(m["content"])
            if role == "user":
                messages_for_invoke.append(HumanMessage(content=text))
            elif role == "assistant":
                messages_for_invoke.append(AIMessage(content=text))
            elif role == "system":
                messages_for_invoke.append(SystemMessage(content=text))
            else:
                messages_for_invoke.append(HumanMessage(content=text))  # fallback
        else:
            # fallback, treat as user text
            messages_for_invoke.append(HumanMessage(content=str(m)))

    response = llm.invoke(messages_for_invoke)

    last_tool_call = getattr(response, "tool_calls", None)
    if last_tool_call:
        for call in last_tool_call:
            if isinstance(call, dict):
                for k, v in call.items():
                    state[k] = v

        # 🔥 NEW: handle tool outputs if they are dicts in response.content
        try:
            if isinstance(response.content, str) and response.content.strip().startswith("{"):
                import ast
                tool_output = ast.literal_eval(response.content)
                if isinstance(tool_output, dict):
                    state.update(tool_output)
        except Exception:
            pass

        # now plotting should work
        if isinstance(state.get("original_image"), np.ndarray) and isinstance(state.get("output_image"), np.ndarray):
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(state["original_image"])
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            axes[1].imshow(state["output_image"])
            axes[1].set_title("Processed / Cloud Removed")
            axes[1].axis("off")
            plt.show()
    

    return {
        "messages": [response],
        "original_image": state.get("original_image"),
        "output_image": state.get("output_image")
    }

def should_continue(state: State): 
    messages = state["messages"] 
    last_message = messages[-1] 
    if getattr(last_message, "tool_calls", None): 
        return "continue"
    return "end"
graph = StateGraph(State)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")
graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    },
)
graph.add_edge("tools", "our_agent")

# then compile
memory = MemorySaver()
app = graph.compile(checkpointer=memory)    

def _to_text(obj):
    """Convert LLM / Message / other objects into a safe string for the chat UI."""
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    # If object has `.content` attribute (BaseMessage-like), prefer it
    content = getattr(obj, "content", None)
    if content is not None:
        return str(content)
    # If it's a dict with 'content'
    if isinstance(obj, dict) and "content" in obj:
        return str(obj["content"])
    # If it's a tuple like (role, content)
    if isinstance(obj, (list, tuple)) and len(obj) >= 2:
        return str(obj[1])
    return str(obj)
def _load_preview_from_path(path):
    if path is None or not os.path.exists(path):
        return None
    path = str(path)
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in (".tif", ".tiff"):
            with rasterio.open(path) as src:
                arr = src.read()
                # shape: (C, H, W) → (H, W, C)
                if arr.ndim == 3:
                    arr = np.transpose(arr[:3, :, :], (1, 2, 0)) if arr.shape[0] >= 3 else np.transpose(arr, (1, 2, 0))
                arr = np.nan_to_num(arr)

                # Normalize numeric types to uint8
                if arr.dtype == np.uint16:
                    arr = (arr / 65535.0 * 255.0).astype(np.uint8)
                elif arr.dtype in [np.float32, np.float64]:
                    mn, mx = float(arr.min()), float(arr.max())
                    if mx > 1.0:
                        arr = np.clip(arr / mx * 255.0, 0, 255).astype(np.uint8)
                    else:
                        arr = (arr * 255.0).astype(np.uint8)
                elif arr.dtype != np.uint8:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)

                # Ensure 3 channels
                if arr.ndim == 2:
                    arr = np.stack([arr] * 3, axis=-1)
                elif arr.ndim == 3 and arr.shape[2] > 3:
                    arr = arr[:, :, :3]
                return arr

        else:
            img = Image.open(path).convert("RGB")
            arr = np.array(img)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return arr

    except Exception as e:
        print(f"[_load_preview_from_path] failed to load {path}: {e}")
        return None


state: State = {
        "messages": [],
        "original_image": None,
        "output_image": None
}

def chat_with_agent_general(user_message, history, image_file=None, voice=None):
    history = history or []
    img, download_path = None, None
    final_tool_output = ""

    # ---- update state.messages with user message ----
    # Convert to HumanMessage so it’s consistent with your model_call
    user_msg = HumanMessage(content=user_message if not voice else f"{user_message}\n(Attached voice: {voice})")
    state["messages"].append(user_msg)

    # ---- attach image (if uploaded) ----
    if image_file:
        loaded = _load_preview_from_path(image_file)
        if loaded is not None:
            img = loaded
            download_path = image_file
            state["original_image"] = img.copy()

    seen_tool_call_ids = set()

    try:
        for s in app.stream(
            state,
            config={"configurable": {"thread_id": "user-session-1"}},
            stream_mode="values"
        ):
            for m in s["messages"]:
                tool_id = getattr(m, "tool_call_id", None)
                if tool_id and tool_id in seen_tool_call_ids:
                    continue

                cont = getattr(m, "content", None)
                if isinstance(cont, str):
                    m_out = re.search(r"""['"]output_path['"]\s*:\s*['"]([^'"]+)['"]""", cont)
                    if m_out:
                        found_path = m_out.group(1)
                        loaded = _load_preview_from_path(found_path)
                        if loaded is not None:
                            state["output_image"] = loaded
                            download_path = found_path

                oe = getattr(m, "output_image", None)
                op = getattr(m, "output_path", None)
                if isinstance(oe, np.ndarray) and oe.size > 0:
                    state["output_image"] = oe
                    if op:
                        download_path = op

                if tool_id:
                    seen_tool_call_ids.add(tool_id)

            last_msg = s["messages"][-1]
            final_tool_output = _to_text(last_msg)

        # ---- add assistant reply into state.messages ----
        state["messages"].append(AIMessage(content=final_tool_output))

    except google_ex.ResourceExhausted:
        final_tool_output = "⚠️ The language model quota was exceeded."
        history.append(HumanMessage(content=user_message))
        history.append(AIMessage(content=final_tool_output))       
        return {
            "history": history,
            "assistant_text": final_tool_output,
            "original_image": state["original_image"],
            "output_image": state["output_image"],
            "download_path": download_path,
            "tool_results": final_tool_output,
            "error": "quota"
        }

    except Exception as e:
        final_tool_output = f"An error occurred while processing: {e}"
        history.append(HumanMessage(content=user_message))
        history.append(AIMessage(content=final_tool_output))       
        return {
            "history": history,
            "assistant_text": final_tool_output,
            "original_image": state["original_image"],
            "output_image": state["output_image"],
            "download_path": download_path,
            "tool_results": final_tool_output,
            "error": str(e)
        }

    # ---- append user+assistant pair into history ----
    history.append(HumanMessage(content=user_message))
    history.append(AIMessage(content=final_tool_output))
    return {
        "history": state['messages'],
        "assistant_text": final_tool_output,
        "original_image": state["original_image"],
        "output_image": state["output_image"],
        "download_path": download_path,
        "tool_results": final_tool_output,
        "error": None
    }
history = []

OPERATION_MAP = {
    "equalize": {
        "label": "Histogram equalize (RGB)",
        "desc": "Apply per-channel histogram equalization",
        "func_name": "equalization_Rgb"
    },
    "cloud_remove": {
        "label": "Cloud removal (GAN)",
        "desc": "Run Pix2Pix generator to remove clouds",
        "func_name": "CloudRemoval"
    },
    "cloud_segment": {
        "label": "Cloud segmentation (model)",
        "desc": "Run cloud segmentation model (requires 4-band image)",
        "func_name": "cloud_segmentation_tool"
    },
    "oil_spill": {
        "label": "Oil-spill segmentation",
        "desc": "Detect oil-lookalikes (Sentinel-1 VV)",
        "func_name": "oil_spill_segmentation"
    },
    "get_satallite_metadata": {
        "label": "Extract metadata",
        "desc": "Return geospatial metadata as JSON",
        "func_name": "store_satellite_metadata"
    },
    "noise_filter": {
        "label": "Noise filtering",
        "desc": "Apply median/gaussian filtering to the image",
        "func_name": "noise_filtering"
    },
    "download_url": {
        "label": "Download from URL",
        "desc": "Download a file from a URL into /kaggle/working",
        "func_name": "download_image_from_url"
    },
    "get_segmented_meta": {
        "label": "Segmented image metadata",
        "desc": "Get metadata from an already segmented file",
        "func_name": "get_segmented_metadata"
    }
}
def get_llm_suggestions_freeform(image_path=None, user_message="", max_suggestions=3):
    """
    Ask the LLM for free-text suggestions only (no tool calls).
    Returns a Markdown string with suggestions.
    """
    ops_lines = "\n".join(
        [f"- {info['label']} ({op_id}): {info['desc']}" for op_id, info in OPERATION_MAP.items()]
    )

    prompt = f"""
You are NeoTerra, a helpful assistant for satellite images.
The user uploaded an image. Suggest up to {max_suggestions} useful things
replay wiht the same language that in the user message donot suggest the currnt operations
they could do with it.

Here are the available operations:
{ops_lines}

User message: {user_message or "None"}
Image path: {image_path or "None"}

⚠️ IMPORTANT: Reply with a **short bullet list** of suggestions only. 
Do NOT return JSON, code, or tool calls.
    """

    try:
        response = llm.invoke(prompt)
        text = getattr(response, "content", None) or str(response)
        # sanitize just in case model tries to add extras
        if isinstance(text, list):
            text = "\n".join([str(x) for x in text])
        return text.strip()
    except Exception as e:
        return f"(⚠️ could not generate suggestions: {e})"
import base64
import mimetypes
from langchain.schema import HumanMessage

def STT(audio_path: str) -> str:
    """
    Transcribe an audio file at `audio_path` using a Google generative audio-capable LLM.

    Parameters
    ----------
    audio_path : str
        Path to local audio file (wav, mp3, m4a, etc.)

    Returns
    -------
    str
        Transcribed text or an error message starting with "❌".
    """
    try:
        # Read file bytes
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        if not audio_bytes:
            return "❌ Empty audio file."

        # Guess mime type
        audio_mime_type = mimetypes.guess_type(audio_path)[0] or "audio/wav"

        # Base64-encode
        encoded_audio = base64.b64encode(audio_bytes).decode("utf-8")

        # Build the LLM client (adjust model/params as you like)
        audio_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)

        # Construct message — keep the 'media' entry with encoded data + mime_type
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Transcribe this audio clip."},
                {
                    "type": "media",
                    "data": encoded_audio,
                    "mime_type": audio_mime_type,
                },
            ]
        )

        # Invoke the model
        response = audio_llm.invoke([message])

        # Extract text from response - this may vary depending on how response.content is structured
        transcription = None
        # If response.content is a plain string:
        if isinstance(response.content, str):
            transcription = response.content
        else:
            # Try common structures (list/dict) — adapt if your SDK returns differently
            try:
                transcription = getattr(response, "content", None) or str(response)
            except Exception:
                transcription = str(response)

        # Clean/return result
        transcription = transcription or ""
        print("Transcribed Audio:", transcription)
        return transcription

    except FileNotFoundError:
        return f"❌ File not found: {audio_path}"
    except Exception as e:
        return f"❌ حصل خطأ: {e}"
def _sanitize_user_text(text: str) -> str:
    """
    Remove long paths from attached-image annotations and keep only filename or short tag.
    E.g. transforms:
      "(Attached image: Sentinel_2_Image.tif; path: /tmp/gradio/....tif)"
    into
      "(Attached image: Sentinel_2_Image.tif)"
    """
    if not text:
        return text
    # replace patterns like "(Attached image: NAME; path: /full/path)" -> "(Attached image: NAME)"
    text = re.sub(r"\(Attached image:\s*([^;()]+);\s*path:\s*[^)]+\)", r"(Attached image: \1)", text)
    # also sanitize any stray long /tmp/ or C:\ paths shown alone
    text = re.sub(r"(/[\w\-/\.]{20,}|[A-Za-z]:\\[^\s]{10,})", lambda m: os.path.basename(m.group(0)), text)
    return text
    
def chat_and_store(user_message, state=None, voice=None, image_file=None):
    if state is None:
        state = {
            "messages": [],
            "original_image": None,
            "output_image": None
        }

    result = chat_with_agent_general(user_message, state=state, voice=voice, image_file=image_file)

    img = result["output_image"]
    hidden = result["download_path"]

    # Prepare Gradio messages
    gradio_messages = []
    for msg in result["history"]:
        role = "assistant" if isinstance(msg, AIMessage) else "user"
        content = _sanitize_user_text(getattr(msg, "content", str(msg)))
        gradio_messages.append({"role": role, "content": content})

    # Ensure image is HWC and RGB for Gradio
    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        elif img.ndim == 3 and img.shape[2] > 3:
            img = img[:, :, :3]

    return gradio_messages, state, img, hidden, result["history"]    
# result = chat_with_agent_general(
#     r"make oil segmentation on this iamge D:\Narss Data\5.tif",
#     history
# )

# print("Assistant said:", result["assistant_text"])
# print("Any tool outputs:", result["tool_results"])
# print("history", result["history"])
# print(type(result["output_image"]))
# # ---------- Operation mapping used by LLM suggestions ----------


