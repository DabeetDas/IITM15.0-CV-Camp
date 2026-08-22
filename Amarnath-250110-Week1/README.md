Firstly for image selection, i tried with different images. Images with a dark background gave better results.

For Grayscale conversion i used the standard RGB to Grayscale conversion formula-
(0.299 * R + 0.587 * G+ 0.114 * B)

I have used sobel filter for convulation

I tried different values to decide the value of threshold, a higher number like 180 was ignoring edges while low number included background noise
so i took 110

i also tried using the code given in the resources (blog on edge detection) which used cv2 library. (I have also added that code in the notebook)
Edge detection result was better there.
