Code overview:-
1. The image was first convereted to grey scale image
2. Applied seperately X any Y direction kernels
3. Calculated overall edge_detection magnitude using sqrt(Gx^2 + Gy^2)
4. Plotted final images from all three filters

Kernels

Prewitt
Gx = [-1  0  1
      -1  0  1
      -1  0  1]

Gy = [-1 -1 -1
       0  0  0
       1  1  1]

Sobel
Gx = [-1  0  1
      -2  0  2
      -1  0  1]

Gy = [-1 -2 -1
       0  0  0
       1  2  1]

Scharr
Gx = [-3   0   3
      -10  0  10
       -3  0   3]

Gy = [ 3  10  3
       0   0  0
      -3 -10 -3]

Comparison

Prewitt: simplest; detects edges well but is relatively less robust to noise.

Sobel: gives more weight to the central pixels, making it somewhat more robust to noise.

Scharr: uses stronger central weighting and has better rotational symmetry, generally giving sharper/more accurate gradient estimates.

