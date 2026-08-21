# Edge Detection Experimentation

# What this code does
For this task, I wrote a Python script to detect edges in an image using 2D convolutions. I decided to implement and compare two different mathematical operators: Sobel and Scharr.

# Methods Tested
1. I used OpenCV to read the test image in grayscale. This makes the math easier since I only have to deal with one matrix of pixel intensities instead of three color channels.
2. I defined 3 x 3 matrices for the X and Y directions. Sobel uses smaller weights (like 1 and 2), while Scharr uses larger weights (3 and 10) to create a stronger response.
3. Used `scipy.signal.convolve2d` to pass the X and Y kernels over the image, which calculates the vertical and horizontal gradients.
4. To get the final edge map, I combined the X and Y results using the magnitude formula: $G = \sqrt{G_x^2 + G_y^2}$.

# Conclusion
The Sobel operator is the most practical choice for basic edge detection on standard images due to its better noise-smoothing characteristics .
