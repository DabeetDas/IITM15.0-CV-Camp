# Edge Detection Experimentation

# Objective
To test and compare convolution-based edge detection techniques.

# Methods Tested
1. Sobel Operator (Primary): Used a 3x3 kernel. It provided a good balance between edge emphasis and noise suppression because the center weights are slightly heavier, which creates a slight smoothing effect.
2. Scharr Operator: I experimented with Scharr kernels (weights of 3 and 10 instead of 1 and 2). 
   Result: Scharr picked up finer details and much higher gradient responses, but it also amplified background noise more than Sobel.
3. Prewitt Operator: Swapped the kernel weights to uniform values (1s and -1s). 
   Result: The edges were slightly blurrier compared to Sobel, proving that Sobel's heavier center weighting is superior for standard images.

# Conclusion
The Sobel operator is the most practical choice for basic edge detection on standard images due to its inherent noise-smoothing characteristics during the gradient calculation.
