import os
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
from model import UNetDeblurImproved
import numpy as np
from itertools import product
import cv2
import sys

class TextScanner:
    def __init__(self):
        self.ambiguous_chars = {
            '8': ['B'],
            'B': ['8'],
            'U': ['LI'],
            'l': ['I', '1'],
            'I': ['l', '1'],
            '1': ['l', 'I'],
            'O': ['0'],
            '0': ['O'],
            'S': ['5'],
            '5': ['S'],
            'Z': ['2'],
            '2': ['Z']
        }
        
        # Initialize the deblurring model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNetDeblurImproved().to(self.device)
        self.model.load_state_dict(torch.load("deblur_net.pth", map_location=self.device))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((36, 1100)),
            transforms.ToTensor(),
        ])

    def deblur_image(self, image_path):
        """Deblur the image using the trained model"""
        input_img = Image.open(image_path).convert("L")
        input_tensor = self.transform(input_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output_tensor = self.model(input_tensor)
        
        output_tensor = output_tensor.squeeze(0).cpu().clamp(0, 1)
        return transforms.ToPILImage()(output_tensor)

    def recognize_text(self, image, filename=None):
        """OCR function that segments and recognizes individual characters"""
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Image dimensions
        height, width = img_array.shape
        
        # For OpenSans-Light, characters have variable widths
        # We'll use adaptive segmentation based on connected components
        _, binary = cv2.threshold(img_array, 128, 255, cv2.THRESH_BINARY)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            (255 - binary).astype(np.uint8), connectivity=8
        )
        
        # Sort components by x-coordinate
        char_regions = []
        for i in range(1, num_labels):  # Skip background (label 0)
            x, y, w, h, area = stats[i]
            if area > 20:  # Filter out noise
                char_regions.append((x, y, w, h, i))
        
        char_regions.sort(key=lambda x: x[0])  # Sort by x-coordinate
        
        recognized_text = ""
        debug_image = image.copy()
        debug_draw = ImageDraw.Draw(debug_image)
        
        for x, y, w, h, label_idx in char_regions:
            # Extract character region
            char_mask = (labels == label_idx).astype(np.uint8) * 255
            char_binary = char_mask[y:y+h, x:x+w]
            
            # Recognize character
            char = self.recognize_character(char_binary)
            recognized_text += char
            
            # Draw debug visualization
            debug_draw.rectangle([x, y, x+w, y+h], outline='red')
        
        # Save debug visualization
        debug_image.save('debug_ocr.png')
        
        return recognized_text

    def recognize_character(self, char_image):
        """Recognize a single character from its binary image"""
        # Calculate basic features
        total_black = np.sum(char_image < 128)  # Count dark pixels
        
        # Calculate vertical profile (sum along rows)
        vertical_profile = np.sum(char_image < 128, axis=0)
        
        # Calculate horizontal profile (sum along columns)
        horizontal_profile = np.sum(char_image < 128, axis=1)
        
        # Calculate center of mass
        y_indices, x_indices = np.where(char_image < 128)
        if len(y_indices) > 0 and len(x_indices) > 0:
            center_y = np.mean(y_indices)
            center_x = np.mean(x_indices)
        else:
            return ' '  # Empty character
        
        # Number of connected components
        num_components = self.count_connected_components(char_image)
        
        # Decision tree for character recognition
        # This is a simplified version - you'll need to tune these rules
        if total_black < 50:  # Threshold for empty space
            return ' '
        
        # Identify common patterns
        if num_components == 2:
            if self.has_dot_above(horizontal_profile):
                return 'i'
            if self.is_equal_sign(horizontal_profile):
                return '='
            
        if self.is_vertical_line(vertical_profile):
            if center_y < char_image.shape[0] * 0.4:  # Upper half
                return 'l'
            return '1'
        
        if self.is_zero_like(horizontal_profile, vertical_profile):
            if self.has_diagonal_line(char_image):
                return '0'
            return 'O'
        
        # Add more character recognition rules...
        
        # Return most likely character based on features
        return self.classify_character(char_image, total_black, center_y, center_x, 
                                     vertical_profile, horizontal_profile, num_components)

    def count_connected_components(self, binary_image):
        """Count number of connected components in binary image"""
        return cv2.connectedComponents(binary_image.astype(np.uint8))[0] - 1

    def has_dot_above(self, horizontal_profile):
        """Check if there's a dot in the upper portion"""
        upper_third = len(horizontal_profile) // 3
        return np.any(horizontal_profile[:upper_third] > 0)

    def is_equal_sign(self, horizontal_profile):
        """Detect if pattern matches '=' character"""
        peaks = np.where(horizontal_profile > np.max(horizontal_profile) * 0.5)[0]
        return len(peaks) == 2 and (peaks[1] - peaks[0]) > 3

    def is_vertical_line(self, vertical_profile):
        """Detect if pattern is a vertical line (I, l, 1)"""
        return np.std(vertical_profile) < 5 and np.mean(vertical_profile) > 0

    def is_zero_like(self, horizontal_profile, vertical_profile):
        """Detect if pattern matches 0 or O"""
        h_std = np.std(horizontal_profile)
        v_std = np.std(vertical_profile)
        return h_std < 10 and v_std < 10

    def has_diagonal_line(self, char_image):
        """Detect diagonal line (helps distinguish 0 from O)"""
        edges = cv2.Canny(char_image.astype(np.uint8), 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=15)
        if lines is not None:
            for rho, theta in lines[0]:
                if 0.25 < theta/np.pi < 0.75:  # ~45 degree angle
                    return True
        return False

    def classify_character(self, char_image, total_black, center_y, center_x,
                          vertical_profile, horizontal_profile, num_components):
        """Final classification based on all features"""
        # You can expand this with more sophisticated classification
        # Could use a trained classifier (SVM, Random Forest, etc.)
        # For now, using simple rules
        
        features = {
            'total_black': total_black,
            'center_y': center_y,
            'center_x': center_x,
            'v_profile_std': np.std(vertical_profile),
            'h_profile_std': np.std(horizontal_profile),
            'num_components': num_components
        }
        
        # Add more character recognition rules here
        # This is where you'd implement the main character recognition logic
        # based on the features we've calculated
        
        # For now, return a default character (you'll want to improve this)
        return 'X'

    def generate_combinations(self, text):
        """Generate all possible combinations based on ambiguous characters"""
        positions = []
        chars = []
        
        # Find all ambiguous characters and their positions
        for i, char in enumerate(text):
            if char in self.ambiguous_chars:
                positions.append(i)
                chars.append([char] + self.ambiguous_chars[char])
        
        if not positions:
            return [text]
        
        # Generate all possible combinations
        combinations = []
        for combination in product(*chars):
            new_text = list(text)
            for pos, new_char in zip(positions, combination):
                new_text[pos] = new_char
            combinations.append(''.join(new_text))
            
        return combinations

    def scan_and_generate(self, image_path):
        """Main function to scan image and generate possible combinations"""
        print(f"\nProcessing file: {image_path}")
        
        # First deblur the image
        deblurred_img = self.deblur_image(image_path)
        print("Image deblurred successfully")
        
        # Recognize text from deblurred image
        text = self.recognize_text(deblurred_img, filename=os.path.basename(image_path))
        
        # Generate all possible combinations
        combinations = self.generate_combinations(text)
        
        result = {
            'original_text': text,
            'combinations': combinations,
            'num_combinations': len(combinations),
            'ambiguous_positions': [i for i, char in enumerate(text) if char in self.ambiguous_chars]
        }
        
        print(f"Generated {result['num_combinations']} possible combinations")
        print(f"Found {len(result['ambiguous_positions'])} ambiguous characters")
        
        return result

def main():
    scanner = TextScanner()
    
    # Get command line argument
    if len(sys.argv) != 2:
        print("Usage: python scanner.py <image_file>")
        sys.exit(1)
        
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} not found")
        sys.exit(1)
    
    result = scanner.scan_and_generate(image_path)
    
    print(f"\nResults for {image_path}:")
    print(f"Original text: {result['original_text']}")
    print(f"Number of possible combinations: {result['num_combinations']}")
    print(f"Ambiguous positions: {result['ambiguous_positions']}")
    print("\nPossible combinations:")
    for combo in result['combinations']:
        print(combo)

if __name__ == "__main__":
    main() 