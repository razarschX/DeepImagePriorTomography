import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.util import img_as_float, random_noise
from skimage.restoration import denoise_nl_means, estimate_sigma
import cv2


# Step 1: Load and preprocess the input image
def load_image(image_path):
    """Load an image and preprocess it for further operations."""
    image = imread(image_path)
    if len(image.shape) == 3:  # If RGB, convert to grayscale
        image = rgb2gray(image)
    image = img_as_float(image)  # Ensure image is float in range [0, 1]
    image = cv2.resize(image, (256, 256))  # Resize image to 256x256 for consistency
    return image


# Step 2: Add noise
def add_noise(image, noise_level=0.1):
    """Add Gaussian noise to an image."""
    noisy_image = random_noise(image, mode='gaussian', var=noise_level ** 2)
    return noisy_image


# Step 3: Advanced Denoising Methods
def advanced_denoise(image):
    """Perform advanced denoising using Non-Local Means filter."""
    sigma_est = np.mean(estimate_sigma(image, channel_axis=None))
    denoised_image = denoise_nl_means(image, h=1.15 * sigma_est, fast_mode=True, patch_size=7, patch_distance=11)
    return denoised_image


# Step 4: Visualize results
def visualize_results(original_image, noisy_image, denoised_image):
    """Visualize the original, noisy, and denoised images."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Noisy image
    axes[1].imshow(noisy_image, cmap='gray')
    axes[1].set_title("Noisy Image")
    axes[1].axis('off')

    # Denoised image
    axes[2].imshow(denoised_image, cmap='gray')
    axes[2].set_title("Denoised Image")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


# Step 5 (Optional): Calculate and display the error
def calculate_comparison_error(original_image, denoised_image):
    """Compare the original image and the denoised image."""
    error = np.mean((original_image - denoised_image) ** 2)
    print(f"Denoising Error (MSE): {error}")


# Main Execution
if __name__ == "__main__":
    # Provide the image path
    image_path = input("Enter the path to an image: ").strip()
    original_image = load_image(image_path)

    # Add noise to the image
    noisy_image = add_noise(original_image, noise_level=0.1)

    # Perform advanced denoising (Non-Local Means)
    denoised_image = advanced_denoise(noisy_image)

    # Visualize the results
    visualize_results(original_image, noisy_image, denoised_image)

    # Calculate and display the error
    calculate_comparison_error(original_image, denoised_image)
