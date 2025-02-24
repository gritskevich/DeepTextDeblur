from PIL import Image
import numpy as np

def analyze_black_distribution(image_path):
    """
    Inverts the grayscale image so that black pixels are 255
    and white pixels are 0, then sums row intensities.
    """
    img = Image.open(image_path).convert('L')
    data = np.array(img)  # shape: (36, 1100)

    # Invert intensities: black becomes 255, white becomes 0
    blackness = 255 - data

    # Sum across each row
    row_sums = np.sum(blackness, axis=1)

    total_blackness = np.sum(row_sums)
    row_indices = np.arange(row_sums.shape[0])

    center_of_mass = np.sum(row_indices * row_sums) / total_blackness

    cdf = np.cumsum(row_sums)
    cdf_normalized = cdf / total_blackness

    line_10_percent = np.searchsorted(cdf_normalized, 0.10)
    line_90_percent = np.searchsorted(cdf_normalized, 0.90)

    return {
        "row_sums_black": row_sums,
        "total_blackness": total_blackness,
        "center_of_mass": center_of_mass,
        "line_10_percent": line_10_percent,
        "line_90_percent": line_90_percent
    }

if __name__ == "__main__":
    image_path = "manual.png"
    results = analyze_black_distribution(image_path)
    print("Row sums (blackness):", results["row_sums_black"])
    print("Total blackness:", results["total_blackness"])
    print("Center of mass (row):", results["center_of_mass"])
    print("10% cutoff row:", results["line_10_percent"])
    print("90% cutoff row:", results["line_90_percent"])

# Center of mass (row): 16.051223790161842
# 10% cutoff row: 7
# 90% cutoff row: 26
# Total blackness: 1182656
