import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature

# Load the image
image = cv2.imread("cat.jpg")
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Edge Detection using Canny
edges = cv2.Canny(image_gray, 100, 200)

# Edge Detection using Sobel
sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=5)  # Horizontal edges
sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=5)  # Vertical edges
sobel_combined = cv2.magnitude(sobel_x, sobel_y)

# Texture Extraction using Gabor Filters
kernels = []
for theta in range(4):
    theta = theta / 4. * np.pi
    for sigma in (1, 3):
        for lamda in np.pi / 4. * np.array([0, 1]):
            gabor = cv2.getGaborKernel((21, 21), sigma, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
            kernels.append(gabor)

# Apply Gabor filters
filtered_images = []
for kernel in kernels:
    filtered = cv2.filter2D(image_gray, cv2.CV_8UC3, kernel)
    filtered_images.append(filtered)

# Texture Extraction using Local Binary Patterns (LBP)
lbp = feature.local_binary_pattern(image_gray, P=8, R=1, method="uniform")

# Plotting the results
plt.figure(figsize=(15, 10))

# Original Image
plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# Canny Edges
plt.subplot(2, 3, 2)
plt.imshow(edges, cmap='gray')
plt.title('Canny Edges')
plt.axis('off')

# Sobel Combined Edges
plt.subplot(2, 3, 3)
plt.imshow(sobel_combined, cmap='gray')
plt.title('Sobel Edges')
plt.axis('off')

# Gabor Filter Results
for i in range(len(filtered_images)):
    plt.subplot(2, 3, 4 + i)
    plt.imshow(filtered_images[i], cmap='gray')
    plt.title(f'Gabor Filter {i+1}')
    plt.axis('off')

# LBP Plot
plt.subplot(2, 3, 7)
plt.imshow(lbp, cmap='gray')
plt.title('Local Binary Pattern')
plt.axis('off')

plt.tight_layout()
plt.show()
