import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def detect_edges(image_path):
    """
    Reads an image and applies the Sobel operator using 2D convolution
    to detect horizontal and vertical edges.
    """
    1. Load the image in grayscale
    Grayscale simplifies the image to a single 2D matrix of intensities
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Image not found. Please ensure 'sample_image.jpg' is in the directory.")
        return

    2. Define the Sobel kernels
    The X kernel detects vertical edges (horizontal gradients)
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    The Y kernel detects horizontal edges (vertical gradients)
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]])

    3. Apply 2D Convolution
    'mode=same' ensures the output image is the same size as the input
    'boundary=symm' handles the edges of the image smoothly
    grad_x = convolve2d(img, sobel_x, mode='same', boundary='symm')
    grad_y = convolve2d(img, sobel_y, mode='same', boundary='symm')

    4. Calculate Gradient Magnitude
    This combines both X and Y gradients to find the overall edge strength
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    5. Normalize the result
    Scale the values back to the standard 0-255 pixel range
    gradient_magnitude = (gradient_magnitude / gradient_magnitude.max()) * 255
    gradient_magnitude = gradient_magnitude.astype(np.uint8)

    6. Plotting the results side-by-side for comparison
    g, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original Grayscale')
    axes[0].axis('off')

    axes[1].imshow(np.abs(grad_x), cmap='gray')
    axes[1].set_title('Vertical Edges (Sobel X)')
    axes[1].axis('off')

    axes[2].imshow(gradient_magnitude, cmap='gray')
    axes[2].set_title('Combined Edge Magnitude')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run the function
    detect_edges('sample_image.jpg')
