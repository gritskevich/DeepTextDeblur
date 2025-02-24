import os
from PIL import Image

def shift_image_right(input_path, output_path, shift_pixels=5):
    """
    Shifts an image right by specified pixels, adds white space on left,
    and crops right side to maintain original dimensions.
    """
    # Read image
    with Image.open(input_path) as img:
        img = img.convert('L')  # Convert to grayscale
        width, height = img.size
        
        # Create new white image
        shifted = Image.new('L', (width, height), color=255)
        
        # Paste original image with shift, excluding rightmost pixels
        shifted.paste(img.crop((0, 0, width - shift_pixels, height)), 
                     (shift_pixels, 0))
        
        # Save the result
        shifted.save(output_path)
        return True

def process_directory(input_dir="./vb/license", shift_pixels=5):
    """Process all PNG files in the specified directory"""
    # Create output directory if it doesn't exist
    output_dir = os.path.join(input_dir, "shifted")
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each PNG file
    count = 0
    for filename in os.listdir(input_dir):
        if filename.endswith('.png'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            if shift_image_right(input_path, output_path, shift_pixels):
                count += 1
                print(f"Processed: {filename}")
            
    print(f"\nCompleted! Processed {count} images.")
    print(f"Shifted images are saved in: {output_dir}")

if __name__ == "__main__":
    process_directory() 