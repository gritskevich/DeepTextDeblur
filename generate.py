import os
import random
import string
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ---------- Configuration ----------
NUM_EXAMPLES = 10000
TEXT_LENGTH = 70

# Large font so text is BIG
FONT_SIZE = 24

# Base64-like chars (letters, digits, plus, slash)
ALLOWED_CHARS = string.ascii_letters + string.digits + '+/'

# Absolute path to a real .ttf font on your Mac
# e.g. "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_PATH = "./static/OpenSans-Light.ttf"

BG_COLOR = (255, 255, 255)  # White background
TEXT_COLOR = (0, 0, 0)      # Black text
BLUR_RADIUS = 5         # Gaussian blur radius

# Very large dummy canvas to avoid clipping
DUMMY_WIDTH, DUMMY_HEIGHT = 1100, 36

# Output directories
PATH_SHARP = 'data/Deblur/Sharp/'
PATH_BLUR = 'data/Deblur/Blur/'

os.makedirs(PATH_SHARP, exist_ok=True)
os.makedirs(PATH_BLUR, exist_ok=True)

# ---------- Load the Font ----------
try:
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    print(f"Loaded font from: {FONT_PATH}")
except IOError:
    raise SystemExit(
        f"ERROR: Could not load font at {FONT_PATH}. "
        "Please specify a valid .ttf file path."
    )

# ---------- Generate Examples ----------
for i in range(NUM_EXAMPLES):
    # 1) Generate a random "base64-like" string (70 chars)
    text = ''.join(random.choices(ALLOWED_CHARS, k=TEXT_LENGTH))
#    text = '3D3VC2QP9askfGHzc/tmHHIdQQ+q94P7VWxXuils9tRAhO77bUzrwrfQs90NPit/FgLveV'

    # 2) Create a large dummy canvas
    dummy_img = Image.new("RGB", (DUMMY_WIDTH, DUMMY_HEIGHT), BG_COLOR)
    draw_dummy = ImageDraw.Draw(dummy_img)

    # Draw text at (0,0). We'll crop out the whitespace next.
    draw_dummy.text((4, -2), text, font=font, fill=TEXT_COLOR)

    # 3) Detect bounding box of all non-background pixels
    bbox = dummy_img.getbbox()
    if bbox is None:
        # If text is empty or something went wrong, skip
        print(f"Warning: No bounding box found for text: {text}")
        continue

    # 4) Crop the dummy image to the bounding box
    sharp_img = dummy_img.crop(bbox)

    # 5) Save the "sharp" image
    sharp_filename = os.path.join(PATH_SHARP, f"sharp_{i:04d}.png")
    sharp_img.save(sharp_filename)

    # 6) Create the blurred image
#    blur_img = sharp_img.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))
#    blur_filename = os.path.join(PATH_BLUR, f"blur_{i:04d}.png")
#    blur_img.save(blur_filename)

    print(f"[{i+1}/{NUM_EXAMPLES}] Saved sharp. Text: {text}")

print("All done! Check your output folders for results.")