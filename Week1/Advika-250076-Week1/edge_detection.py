import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def detect_edges(image_path):
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Image not found. Please put a file named 'sample_image.jpg' in the same folder.")
        return

    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]])

    
    scharr_x = np.array([[-3, 0, 3],
                         [-10, 0, 10],
                         [-3, 0, 3]])

    scharr_y = np.array([[-3, -10, -3],
                         [ 0,   0,  0],
                         [ 3,  10,  3]])

    
    grad_x_sobel = convolve2d(img, sobel_x, mode='same', boundary='symm')
    grad_y_sobel = convolve2d(img, sobel_y, mode='same', boundary='symm')
    
   
    grad_x_scharr = convolve2d(img, scharr_x, mode='same', boundary='symm')
    grad_y_scharr = convolve2d(img, scharr_y, mode='same', boundary='symm')

    sobel_magnitude = np.sqrt(grad_x_sobel**2 + grad_y_sobel**2)
    scharr_magnitude = np.sqrt(grad_x_scharr**2 + grad_y_scharr**2)


    sobel_magnitude = (sobel_magnitude / sobel_magnitude.max()) * 255
    scharr_magnitude = (scharr_magnitude / scharr_magnitude.max()) * 255
    
    sobel_magnitude = sobel_magnitude.astype(np.uint8)
    scharr_magnitude = scharr_magnitude.astype(np.uint8)

    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original Grayscale')
    axes[0].axis('off')

    axes[1].imshow(sobel_magnitude, cmap='gray')
    axes[1].set_title('Sobel Edge Detection')
    axes[1].axis('off')

    axes[2].imshow(scharr_magnitude, cmap='gray')
    axes[2].set_title('Scharr Edge Detection')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
  
    detect_edges('sample_image.jpg')
    

