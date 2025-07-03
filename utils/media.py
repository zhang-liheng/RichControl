import numpy as np
import torch
import torchvision.transforms.functional as vF
import cv2
import PIL
from PIL import Image
import base64


JPEG_QUALITY = 95


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def preprocess(image, processor, **kwargs):
    if isinstance(image, PIL.Image.Image):
        pass
    elif isinstance(image, np.ndarray):
        image = PIL.Image.fromarray(image)
    elif isinstance(image, torch.Tensor):
        image = vF.to_pil_image(image)
    else:
        raise TypeError(f"Image must be of type PIL.Image, np.ndarray, or torch.Tensor, got {type(image)} instead.")
    
    image = processor.preprocess(image, **kwargs)
    return image


def erode_image(image_path, min_width=25, max_width=50, kernel_size=10, output_path=None):
    if isinstance(image_path, np.ndarray):
        original_img = image_path
    elif isinstance(image_path, Image.Image):
        original_img = cv2.cvtColor(np.array(image_path),cv2.COLOR_RGB2BGR)
    else:
        original_img = cv2.imread(image_path)
    
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    
    is_thick_original, original_max_line_width = check_line_thickness(gray, min_width, max_width)
    is_thick_inverted, inverted_max_line_width = check_line_thickness(255 - gray, min_width, max_width)
    
    result_img = original_img.copy()
    
    if is_thick_original:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        result_img = cv2.dilate(original_img, kernel, iterations=1)

    elif is_thick_inverted:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        result_img = cv2.erode(original_img, kernel, iterations=1)
    
    if output_path is not None:
        cv2.imwrite(output_path, result_img)

    if isinstance(image_path, Image.Image) or isinstance(image_path, str):
        result_img = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            
    return result_img


def check_line_thickness(gray_img, min_width, max_width):
    _, binary = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
    binary_inv = cv2.bitwise_not(binary)
    dist_transform = cv2.distanceTransform(binary_inv, cv2.DIST_L2, 3)
    distances = dist_transform[dist_transform > 0]
    if distances.size == 0:
        return False
    
    line_widths = distances * 2
    max_line_width = np.max(line_widths)
    return min_width <= max_line_width <= max_width, max_line_width


def unsharp_mask_image(image_path, contrast_factor=50, blur_radius=3):
    if isinstance(image_path, Image.Image):
        image = cv2.cvtColor(np.array(image_path), cv2.COLOR_RGB2BGR)
    elif isinstance(image_path, str):
        image = cv2.imread(image_path)
    
    blurred = cv2.GaussianBlur(image, (0, 0), blur_radius)
    sharpened = cv2.addWeighted(image, 1.0 + contrast_factor, blurred, -contrast_factor, 0)
    
    if isinstance(image_path, Image.Image) or isinstance(image_path, str):
        sharpened = Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))

    return sharpened