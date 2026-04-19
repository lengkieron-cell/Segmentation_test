import torch
import segmentation_models_pytorch as smp
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import glob
import _pickle
import cv2
from scipy import ndimage
from skimage import measure
import json

# --- Configuration for your model ---
n_cls = 2
model_path = "rooftop_best_model.pt"
# OR
# model_path = "C:/Users/Kylek/Downloads/Kieron Stuff/saved_models/rrooftop_new_guy_best_model.pt"

# --- 1. Define the device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 2. Instantiate the model architecture ---
# CRITICAL CHANGE: Match the encoder used in your training script!
model = smp.DeepLabV3Plus(
    encoder_name="resnet34", # <-- Use this!
    classes=n_cls
)

# --- 3. Load the saved model weights ---
import dill
import sys, types
import torch.nn as nn
import torch.nn.functional as F # Keeping F import for compatibility

# =========================================================================
# === CRITICAL FIX: Full Serialization Patch for EfficientNet Encoder ===
# Defines ALL the missing classes the PyTorch unpickler needs.
# =========================================================================

# 1. Define the missing classes (They only need to exist as placeholders)
class Conv2dStaticSamePadding(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    @classmethod
    def from_name(cls, *args, **kwargs):
        return cls()
        
class MBConvBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
class GlobalParams:
    def __init__(self, *args, **kwargs):
        pass
        
class BlockArgs:
    def __init__(self, *args, **kwargs):
        pass

# 2. Define the missing module and inject ALL the classes into it
MISSING_MODULE_NAME = 'segmentation_models_pytorch.encoders._efficientnet'
dummy_module = types.ModuleType(MISSING_MODULE_NAME)

setattr(dummy_module, 'Conv2dStaticSamePadding', Conv2dStaticSamePadding)
setattr(dummy_module, 'MBConvBlock', MBConvBlock) 
setattr(dummy_module, 'GlobalParams', GlobalParams) 
setattr(dummy_module, 'BlockArgs', BlockArgs)      
sys.modules[MISSING_MODULE_NAME] = dummy_module

# =========================================================================
# === LOGIC FIX: Handle Full Model Object vs. State Dict ===

try:
    # Load whatever is saved in the .pt file (state_dict or full model object)
    loaded_obj = torch.load(model_path, map_location=device, pickle_module=dill)
    
    if isinstance(loaded_obj, dict):
        # Case 1: The file contains only the state_dict (a dictionary)
        state_dict_to_load = loaded_obj
        print("Model file detected as state_dict.")
        
    elif hasattr(loaded_obj, 'state_dict'):
        # Case 2: The file contains the entire model object (a DeepLabV3Plus instance)
        # Extract the state dictionary from the loaded model object
        state_dict_to_load = loaded_obj.state_dict()
        print("Model file detected as full model object, extracting state_dict.")
        
    else:
        # Fallback for unexpected format
        raise TypeError("Loaded object is neither a state dictionary nor a PyTorch Module.")
        
    # Load the extracted state_dict into the model instantiated at step 2
    model.load_state_dict(state_dict_to_load)
    print("Model weights loaded successfully.")

except Exception as e:
    print(f"CRITICAL ERROR: Model failed to load due to: {e}")
    exit()

# Final setup steps
model.to(device)
model.eval()
print(f"Model loaded and prepared successfully from {model_path}")

# =========================================================================

try:
    # Attempt 1: Load state_dict using dill
    state_dict = torch.load(model_path, map_location=device, pickle_module=dill)
    # ... rest of your code
    # ... rest of your code
    # ... rest of your code
    state_dict = torch.load(model_path, map_location=device, pickle_module=dill)
    
    print("Model weights (state_dict) loaded successfully.")

except (RuntimeError, _pickle.UnpicklingError) as e:
    print(f"Error loading state_dict directly: {e}")
    print("Attempting to load the entire model object directly with weights_only=False...")
    try:
        # Attempt 2: Load entire model object using dill
        model = torch.load(model_path, map_location=device, weights_only=False, pickle_module=dill)
        model.to(device)
        print("Full model object loaded successfully.")

    except Exception as e_full:
        print(f"Failed to load full model object: {e_full}")
        print("CRITICAL ERROR: Model failed to load.")
        exit()

# These lines must run AFTER the loading attempts are complete
model.to(device) # Ensure model is on CPU, even if loaded as full object
model.eval() # Set the model to evaluation mode
print(f"Model loaded and prepared successfully from {model_path}")

# --- Define transformations for input images ---
im_transform = transforms.Compose([
    # IMPORTANT: Resize the image to the exact dimensions your model was trained on.
    # If your model was trained on 640x640, use transforms.Resize((640, 640)).
    # If it was trained on 256x256, use transforms.Resize((256, 256)), etc.
    # I'm keeping 256x256 as a placeholder; please adjust if your model expects 640x640 or another size.
    transforms.Resize((400, 400)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Utility function to convert PyTorch tensor to NumPy array for display ---
def tn_2_np(tensor):
    np_array = tensor.numpy()
    if np_array.ndim == 3 and np_array.shape[0] in [1, 3]:
        if np_array.shape[0] == 3:
            np_array = np.transpose(np_array, (1, 2, 0))
        else:
            np_array = np.squeeze(0)
    if np_array.ndim == 3 and np_array.shape[2] == 3:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        np_array = std * np_array + mean
        np_array = np.clip(np_array, 0, 1)
    return np_array

# --- NEW: Structured Detection Output Functions ---
def extract_rooftop_detections(mask, min_area=100, pixel_to_meter_ratio=0.25, simplify_epsilon=0.005):
    """
    Extract structured detection information from segmentation mask with tight polygon fitting.
    
    Args:
        mask: Binary mask (0 for background, 1 for rooftop)
        min_area: Minimum area in pixels to consider a detection
        pixel_to_meter_ratio: Conversion factor from pixels to meters
        simplify_epsilon: Contour simplification factor (lower = more detailed, higher = more simplified/straight)
                          This value is a percentage of the contour's perimeter.
    
    Returns:
        List of detection dictionaries with tight polygons, areas, centroids, etc.
    """
    detections = []
    
    # Find connected components (individual rooftop instances)
    labeled_mask, num_features = ndimage.label(mask)
    
    for i in range(1, num_features + 1):
        # Extract individual rooftop region
        rooftop_region = (labeled_mask == i)
        
        # Calculate area in pixels
        area_pixels = np.sum(rooftop_region)
        
        # Skip small detections
        if area_pixels < min_area:
            continue
            
        # Calculate area in square meters
        area_m2 = area_pixels * (pixel_to_meter_ratio ** 2)
        
        # Calculate centroid
        coords = np.where(rooftop_region)
        centroid_y = np.mean(coords[0])
        centroid_x = np.mean(coords[1])
        
        # Extract tight contour from mask
        contours, _ = cv2.findContours(rooftop_region.astype(np.uint8), 
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        polygon_points = []
        detailed_contour = []
        bbox = {'x_min': 0, 'y_min': 0, 'x_max': 0, 'y_max': 0, 'width': 0, 'height': 0}
        
        if contours:
            # Get the largest contour (main rooftop outline)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get detailed contour points (every pixel on the boundary)
            detailed_contour = largest_contour.reshape(-1, 2).tolist()
            
            # Create simplified polygon for easier processing using cv2.approxPolyDP
            # The 'simplify_epsilon' parameter controls the level of simplification.
            # A smaller value retains more detail, a larger value creates straighter lines.
            epsilon = simplify_epsilon * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            polygon_points = approx.reshape(-1, 2).tolist()
            
            # Calculate tight bounding box from actual contour
            x_coords = [point[0] for point in detailed_contour]
            y_coords = [point[1] for point in detailed_contour]
            
            bbox = {
                'x_min': int(min(x_coords)),
                'y_min': int(min(y_coords)),
                'x_max': int(max(x_coords)),
                'y_max': int(max(y_coords)),
                'width': int(max(x_coords) - min(x_coords)),
                'height': int(max(y_coords) - min(y_coords))
            }
        
        # Calculate confidence based on area and shape regularity
        perimeter = cv2.arcLength(largest_contour, True) if contours else 0
        compactness = (4 * np.pi * area_pixels) / (perimeter ** 2) if perimeter > 0 else 0
        confidence = min(1.0, (area_pixels / 1000) * compactness)
        
        detection = {
            'id': i,
            'class': 'rooftop',
            'confidence': float(confidence),
            'bbox': bbox,
            'centroid': {
                'x': float(centroid_x),
                'y': float(centroid_y)
            },
            'area': {
                'pixels': int(area_pixels),
                'square_meters': float(area_m2)
            },
            'polygon_simplified': polygon_points,   # Simplified polygon for easy processing
            'polygon_detailed': detailed_contour,   # Detailed contour following mask exactly
            'perimeter_pixels': float(perimeter),
            'compactness': float(compactness),
            'estimated_energy_kwh_per_day': float(area_m2 * (100 / 365))
        }
        
        detections.append(detection)
    
    return detections

def visualize_structured_detections(image, detections, title="Rooftop Detections", show_bbox=True, show_polygon=True):
    """
    Visualize detections with tight polygons, optional bounding boxes, and labels.
    
    IMPORTANT CHANGE: This function now draws the 'polygon_simplified' to show
    the geometrically precise, straight-line segments, as requested.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']
    
    for i, detection in enumerate(detections):
        color = colors[i % len(colors)]
        
        # Draw simplified polygon outline
        # Changed from 'polygon_detailed' to 'polygon_simplified' for cleaner lines
        if show_polygon and 'polygon_simplified' in detection: 
            polygon_points = np.array(detection['polygon_simplified'])
            if polygon_points.size > 0: # Check if polygon_points is not empty
                # Close the polygon by adding the first point at the end
                polygon_points = np.vstack([polygon_points, polygon_points[0]])
                ax.plot(polygon_points[:, 0], polygon_points[:, 1], 
                        color=color, linewidth=2, alpha=0.8)
                
                # Fill the polygon with low alpha
                ax.fill(polygon_points[:, 0], polygon_points[:, 1], 
                        color=color, alpha=0.2)
        
        # Draw bounding box (optional, mainly for reference)
        if show_bbox:
            bbox = detection['bbox']
            rect = plt.Rectangle((bbox['x_min'], bbox['y_min']), 
                                 bbox['width'], bbox['height'],
                                 fill=False, color=color, linewidth=1, 
                                 linestyle='--', alpha=0.5)
            ax.add_patch(rect)
        
        # Draw centroid
        centroid = detection['centroid']
        ax.plot(centroid['x'], centroid['y'], 'o', color=color, markersize=6)
        
        # Add label with more info
        label = f"ID: {detection['id']}\n"
        label += f"Area: {detection['area']['square_meters']:.1f} m²\n"
        label += f"Energy: {detection['estimated_energy_kwh_per_day']:.1f} kWh/day\n"
        label += f"Confidence: {detection['confidence']:.2f}"
        
        # Position label at the top of the detection
        label_x = detection['centroid']['x']
        label_y = detection['bbox']['y_min'] - 15
        
        ax.text(label_x, label_y, label, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                fontsize=8, color='white', weight='bold', ha='center')
    
    ax.set_title(f"{title} - {len(detections)} rooftops detected")
    ax.axis('off')
    plt.tight_layout()
    plt.show()

def save_detections_to_json(detections, image_path, output_dir="detections"):
    """
    Save detections to JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get image filename without extension
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    json_path = os.path.join(output_dir, f"{image_name}_detections.json")
    
    # Create output structure
    output_data = {
        'image_path': image_path,
        'image_name': image_name,
        'total_detections': len(detections),
        'total_rooftop_area_m2': sum(d['area']['square_meters'] for d in detections),
        'total_estimated_energy_kwh_per_day': sum(d['estimated_energy_kwh_per_day'] for d in detections),
        'detections': detections
    }
    
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Detections saved to: {json_path}")
    return json_path

# --- Modified Dataset Class ---
class RooftopTestDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, transform=None):
        print(f"Initializing RooftopTestDataset for image_dir: {image_dir}")
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.tif")))
        self.image_paths.extend(sorted(glob.glob(os.path.join(image_dir, "*.png"))))
        self.image_paths = sorted(list(set(self.image_paths))) # Remove duplicates and sort
        print(f"Found {len(self.image_paths)} image files in {image_dir}:")
        for p in self.image_paths:
            print(f"  - {p}")

        self.mask_paths = None
        if mask_dir:
            print(f"Initializing RooftopTestDataset for mask_dir: {mask_dir}")
            self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.tif")))
            self.mask_paths.extend(sorted(glob.glob(os.path.join(mask_dir, "*.png"))))
            self.mask_paths = sorted(list(set(self.mask_paths))) # Remove duplicates and sort
            print(f"Found {len(self.mask_paths)} mask files in {mask_dir}:")
            for p in self.mask_paths:
                print(f"  - {p}")

            if len(self.image_paths) != len(self.mask_paths):
                raise ValueError(f"Mismatch: {len(self.image_paths)} images found, but {len(self.mask_paths)} masks found. "
                                 f"Ensure all images have corresponding masks or process unseen images separately.")
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        mask = None
        if self.mask_paths:
            mask_path = self.mask_paths[idx]
            mask = Image.open(mask_path).convert("L")
            mask = np.array(mask)
            mask = (mask > 0).astype(np.uint8)
            mask = torch.from_numpy(mask).long()

        if self.transform:
            image = self.transform(image)

        if mask is not None:
            return image, mask, img_path   # Return image path for saving detections
        else:
            return image, img_path

# --- NEW: Structured Inference Function ---
def structured_inference(data_loader, model, device, output_dir="detections", visualize=True):
    """
    Perform inference and extract structured detections.
    """
    print("\n--- Starting Structured Inference ---")
    model.eval()
    
    all_detections = []
    
    with torch.no_grad():
        for batch_data in data_loader:
            # Unpack batch data
            if len(batch_data) == 3:   # images, masks, paths
                images, gts, image_paths = batch_data
                has_ground_truth = True
            else:   # images, paths
                images, image_paths = batch_data
                gts = None
                has_ground_truth = False

            images = images.to(device)
            preds = model(images)
            pred_masks = torch.argmax(preds, dim=1).cpu().numpy()

            # Process each image in the batch
            for j in range(images.shape[0]):
                image_path = image_paths[j]
                predicted_mask = pred_masks[j]
                original_image_tensor = images[j].cpu() # Keep tensor for original image display
                original_image_display = tn_2_np(original_image_tensor) # For matplotlib display

                # Extract structured detections with tighter fitting
                # You can adjust 'simplify_epsilon' here to control the smoothness of the polygons.
                # A smaller value (e.g., 0.005) will result in more vertices and less simplification.
                # A larger value (e.g., 0.02 or 0.05) will result in fewer vertices and straighter lines.
                detections = extract_rooftop_detections(predicted_mask, min_area=50, simplify_epsilon=0.01)
                
                # Save detections to JSON
                json_path = save_detections_to_json(detections, image_path, output_dir)
                
                # Visualize if requested
                if visualize and len(detections) > 0:
                    visualize_structured_detections(original_image_display, detections, 
                                                    f"Detections for {os.path.basename(image_path)}")
                
                # Add to all detections
                all_detections.extend(detections)
                
                # Print summary
                total_area = sum(d['area']['square_meters'] for d in detections)
                total_energy = sum(d['estimated_energy_kwh_per_day'] for d in detections)
                
                print(f"Image: {os.path.basename(image_path)}")
                print(f"   Detections: {len(detections)}")
                print(f"   Total Area: {total_area:.2f} m²")
                print(f"   Total Energy: {total_energy:.2f} kWh/day")
                print("-" * 50)
    
    return all_detections

# --- Paths for your data (Relative Paths) ---
# This looks for the folders inside your 'Repo' folder
test_image_dir = "images"
test_mask_dir = "masks"
unseen_image_dir = "unseen_images"

# For the model path at the top of your script:
model_path = "rooftop_best_model.pt"

# --- Instantiate the Test Dataset and DataLoader (with masks) ---
print("\n--- Processing Original Test Set (with masks) ---")
test_dataset = RooftopTestDataset(image_dir=test_image_dir, mask_dir=test_mask_dir, transform=im_transform)
test_dl = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
print(f"Test DataLoader created with {len(test_dataset)} images.")

# Run Structured Inference on Test Set
all_detections_test_set = structured_inference(test_dl, model, device, output_dir="detections_test_set", visualize=True)

print(f"\n--- Final Summary for Original Test Set ---")
print(f"Total detections across all images: {len(all_detections_test_set)}")
print(f"Total rooftop area: {sum(d['area']['square_meters'] for d in all_detections_test_set):.2f} m²")
print(f"Total estimated energy: {sum(d['estimated_energy_kwh_per_day'] for d in all_detections_test_set):.2f} kWh/day")


# --- Instantiate a NEW Dataset and DataLoader for UNSEEN IMAGES (without masks) ---
# IMPORTANT: Notice mask_dir is NOT provided here.
print("\n--- Processing Unseen Images (without masks) ---")
if os.path.exists(unseen_image_dir) and len(os.listdir(unseen_image_dir)) > 0:
    unseen_dataset = RooftopTestDataset(image_dir=unseen_image_dir, transform=im_transform)
    unseen_dl = DataLoader(unseen_dataset, batch_size=1, shuffle=False, num_workers=0) # Batch size 1 for single unseen image
    print(f"Unseen DataLoader created with {len(unseen_dataset)} images.")

    # Run Structured Inference on Unseen Images
    all_detections_unseen = structured_inference(unseen_dl, model, device, output_dir="detections_unseen", visualize=True)

    print(f"\n--- Final Summary for Unseen Images ---")
    print(f"Total detections across all unseen images: {len(all_detections_unseen)}")
    print(f"Total rooftop area: {sum(d['area']['square_meters'] for d in all_detections_unseen):.2f} m²")
    print(f"Total estimated energy: {sum(d['estimated_energy_kwh_per_day'] for d in all_detections_unseen):.2f} kWh/day")
else:
    print(f"Warning: Unseen image directory '{unseen_image_dir}' does not exist or is empty. Skipping unseen image processing.")

# --- Example: Single Image Structured Prediction (kept for specific file testing) ---
# This part is still useful if you want to explicitly test one specific image file.
# Make sure the image_path variable points to an actual image file.
# IMPORTANT: Replace "tile_6400_4000.tif" with the ACTUAL filename of your unseen image.
single_image_test_path = os.path.join(unseen_image_dir, "Tile04_1.tif") # Corrected to point to a file

print(f"\n--- Verifying single image path: {single_image_test_path} ---")
if os.path.exists(single_image_test_path):
    print(f"Success: Single image found at {single_image_test_path}")
    print(f"\n--- Single Image Structured Prediction (Specific File) ---")
    print(f"Processing: {single_image_test_path}")
    
    # Load and process image
    image = Image.open(single_image_test_path).convert("RGB")
    input_tensor = im_transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        predicted_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    
    # Extract structured detections
    detections = extract_rooftop_detections(predicted_mask, min_area=50, simplify_epsilon=0.01)
    
    # Save detections
    json_path = save_detections_to_json(detections, single_image_test_path, output_dir="detections_single_image")
    
    # Visualize
    original_image_display = tn_2_np(input_tensor.squeeze(0).cpu())
    visualize_structured_detections(original_image_display, detections, f"Single Image Detections: {os.path.basename(single_image_test_path)}")
    
    # Print summary
    print(f"\nDetection Summary for {os.path.basename(single_image_test_path)}:")
    print(f"Number of rooftops detected: {len(detections)}")
    for i, det in enumerate(detections):
        print(f"   Rooftop {i+1}: {det['area']['square_meters']:.2f} m², "
              f"{det['estimated_energy_kwh_per_day']:.2f} kWh/day")
    
    total_area = sum(d['area']['square_meters'] for d in detections)
    total_energy = sum(d['estimated_energy_kwh_per_day'] for d in detections)
    print(f"Total area: {total_area:.2f} m²")
    print(f"Total energy: {total_energy:.2f} kWh/day")

else:
    print(f"Error: Single image not found at {single_image_test_path}. Please double-check the path and filename (including extension).")
    256

    state_dict
    