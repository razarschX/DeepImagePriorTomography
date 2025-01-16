import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import radon, iradon, resize
from skimage.metrics import mean_squared_error as mse
import torch
import torch.nn as nn
import torch.optim as optim


# Step 1: Load a user-provided image
def load_user_image():
    """
    Prompt user for image file path and load the image.
    Converts to grayscale if the image is RGB.
    """
    image_path = input("Enter the path to the corrupted image file: ").strip()
    try:
        image = imread(image_path)
        if len(image.shape) == 3:  # Convert RGB to grayscale
            image = rgb2gray(image)
        return image
    except FileNotFoundError:
        print("Error: File not found. Please provide a valid image path.")
        return None


# Step 2: Perform the Radon Transform
def perform_radon_transform(image, theta_values):
    """
    Perform the Radon transform on the image.
    Args:
        image: 2D numpy array of the image.
        theta_values: Array of angles (degrees) for the Radon transform.
    Returns:
        sinogram: The 2D sinogram resulting from the Radon transform.
    """
    sinogram = radon(image, theta=theta_values, circle=True)
    return sinogram


# Step 3: Perform the Inverse Radon Transform
def perform_inverse_radon_transform(sinogram, theta_values):
    """
    Perform the Inverse Radon transform to reconstruct the image.
    Args:
        sinogram: The 2D sinogram.
        theta_values: Array of angles (degrees) that were used in the Radon transform.
    Returns:
        reconstructed_image: The reconstructed image from the sinogram.
    """
    reconstructed_image = iradon(sinogram, theta=theta_values, circle=False)
    return reconstructed_image


# DIP Model: A simple U-Net-like architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Deep Image Prior for Reconstruction
def deep_image_prior(sinogram, corrupted_sinogram, theta_values, epochs=500, lr=0.01):
    """
    Use Deep Image Prior to reconstruct the image from the sinogram.
    Args:
        sinogram: The ground-truth sinogram (for loss reference).
        corrupted_sinogram: The noisy input sinogram.
        theta_values: Projection angles.
        epochs: Number of epochs to optimize.
        lr: Learning rate for optimizer.
    Returns:
        reconstructed_image: Image reconstructed by DIP.
    """
    # Normalize sinograms
    sinogram = sinogram / np.max(sinogram)
    corrupted_sinogram = corrupted_sinogram / np.max(corrupted_sinogram)

    # Convert to PyTorch tensors
    corrupted_sinogram_tensor = torch.from_numpy(corrupted_sinogram).float().unsqueeze(0).unsqueeze(
        0)  # Add batch & channel dimensions
    target_sinogram_tensor = torch.from_numpy(sinogram).float().unsqueeze(0).unsqueeze(0)

    # Initialize DIP model (CNN)
    model = SimpleCNN()
    model = model.train()

    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.MSELoss()

    print("Training DIP...")
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        output_sinogram = model(corrupted_sinogram_tensor)

        # Compute loss (compare to original clean sinogram)
        loss = loss_function(output_sinogram, target_sinogram_tensor)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")

    # Get the denoised sinogram (output of the model)
    denoised_sinogram = model(corrupted_sinogram_tensor).detach().squeeze().numpy()

    # Perform Inverse Radon Transform
    reconstructed_image = perform_inverse_radon_transform(denoised_sinogram, theta_values)

    return reconstructed_image


# Step 4: Visualize and Compare Results
def visualize_results(original_image, noisy_sinogram, denoised_sinogram, original_sinogram, reconstructed_image_dip):
    """
    Visualize the original image, noisy sinogram, denoised sinogram, original sinogram,
    and the DIP reconstructed image.
    Args:
        original_image: The original image.
        noisy_sinogram: The noisy sinogram from Radon transform (corrupted sinogram).
        denoised_sinogram: The denoised sinogram from Deep Image Prior.
        original_sinogram: The clean original sinogram from Radon transform.
        reconstructed_image_dip: The reconstructed image with DIP.
    """
    # Plot the results
    fig, axes = plt.subplots(1, 5, figsize=(24, 6))

    # Original image
    axes[0].imshow(original_image, cmap='gray', aspect='equal')
    axes[0].set_title("Original Image")
    axes[0].axis('off')  # Turn off axis ticks for better visualization

    # Noisy (Corrupted) Sinogram
    axes[1].imshow(noisy_sinogram, cmap='gray', aspect='auto')  # Keep aspect auto for projection data
    axes[1].set_title("Noisy Sinogram (Corrupted)")
    axes[1].axis('on')  # Keep axis ticks for sinogram visualization

    # Denoised Sinogram
    axes[2].imshow(denoised_sinogram, cmap='gray', aspect='auto')  # Display denoised sinogram
    axes[2].set_title("Denoised Sinogram (DIP)")
    axes[2].axis('on')  # Keep axis ticks for visualization

    # Original (Clean) Sinogram
    axes[3].imshow(original_sinogram, cmap='gray', aspect='auto')  # Display clean original sinogram
    axes[3].set_title("Original Sinogram (Clean)")
    axes[3].axis('on')  # Keep axis ticks for visualization

    # Reconstructed image (DIP)
    axes[4].imshow(reconstructed_image_dip, cmap='gray', aspect='equal')
    axes[4].set_title("Reconstructed Image (DIP)")
    axes[4].axis('off')  # Turn off axis ticks for better visualization

    plt.tight_layout()  # Automatically adjust spacings
    plt.show()


# Main Execution
if __name__ == "__main__":
    # 1. Load user-provided image
    original_image = load_user_image()
    if original_image is None:
        print("Exiting program because no valid image was provided.")
        exit(1)

    # 2. Define theta (projection angles for Radon transform)
    theta = np.linspace(0., 180., max(original_image.shape), endpoint=False)  # Angles from 0 to 180 degrees

    # 3. Perform Radon Transform (get the sinogram)
    original_sinogram = perform_radon_transform(original_image, theta)

    # 4. Corrupt the sinogram by adding noise
    noisy_sinogram = original_sinogram + np.random.normal(0, 0.1, original_sinogram.shape)

    # 5. Reconstruct using Deep Image Prior
    reconstructed_dip_image = deep_image_prior(original_sinogram, noisy_sinogram, theta, epochs=500, lr=0.01)

    # 6. Extract the denoised sinogram
    denoised_sinogram = perform_radon_transform(reconstructed_dip_image, theta)

    # 7. Visualize the results
    visualize_results(original_image, noisy_sinogram, denoised_sinogram, original_sinogram, reconstructed_dip_image)