# My contribution:
## Code for Poisson filter
```
def apply_poisson_filter(img):
    # Create a Poisson filter kernel
    kernel_size = 3
    poisson_kernel = poisson.pmf(np.arange(-kernel_size // 2, kernel_size // 2 + 1), 25)
    poisson_kernel = poisson_kernel.reshape(-1, 1)  # Reshape to a column vector
    poisson_kernel = poisson_kernel / poisson_kernel.sum()
    # Apply Poisson filter using convolution
    poisson_filtered_image = convolve(img.astype(float), poisson_kernel, mode='constant', cval=0.0)
    return poisson_filtered_image.astype(np.uint8)
```

## Explanation

### Create Poisson Filter Kernel:

- **Kernel Size:** Set to 3, which defines the size of the filter.
- **Poisson Kernel:** Probability mass function (pmf) of a Poisson distribution is calculated for values within the specified range.
- Reshape the poisson_kernel to a column vector.
- Normalize the kernel by dividing it by its sum.

### Convolution:

- Use the `convolve` function to apply the Poisson filter using convolution.
- The input image (`img`) is converted to float for precision.
- The result is then converted back to uint8 for image representation.


## code for Speckle Filter 

```
def speckle_filter_bilateral(image_path, d=9, sigma_color=75, sigma_space=75):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply bilateral filter for speckle noise reduction
    filtered_image = cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    cv2_imshow(image)
    cv2_imshow(filtered_image)

    # Wait for a key press and close all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

```
## Explanation

### Read Image:

- Read the input image using OpenCV (`cv2`).

### Bilateral Filter:

- Use `cv2.bilateralFilter` to apply a bilateral filter.
- **Parameters:**
  - `d`: Diameter of each pixel neighborhood.
  - `sigma_color`: Filter sigma in the color space.
  - `sigma_space`: Filter sigma in the coordinate space.

### Display Images:

- Display the original and filtered images using `cv2_imshow`.
- Wait for a key press and close all windows.

![1](https://github.com/Kevin-Benny/IA-ImageFilteringTech-Flask/assets/90462533/25010f6b-bc55-478e-afe7-dc359a935241)
![image](https://github.com/Kevin-Benny/IA-ImageFilteringTech-Flask/assets/90462533/12b60868-726a-4059-b9c7-61805a479d5d)
![image](https://github.com/Kevin-Benny/IA-ImageFilteringTech-Flask/assets/90462533/9bd8d380-14a1-4563-9f39-ed4f4f8f7112)


# All Filters Used:
- Speckle Filter 
- Poisson Filter
- Laplace Filter
- Non-Local Means Filter
- Mean Filter
- Gaussian Filter
- Median Filter
- Inverse Filter
- Weiner Filter
- Canny Edge Detection


## Speckle Filter

The Speckle Filter is a specialized tool designed to mitigate speckle noise in images. This form of granular noise commonly occurs in images acquired through various imaging processes.

## Poisson Filter

The Poisson Filter is employed to diminish Poisson noise in images. Poisson noise is inherent in images obtained through processes that involve the counting of photons, such as in medical imaging or low-light photography.

## Laplace Filter

The Laplace Filter serves as an edge detection mechanism, highlighting regions of rapid intensity change within an image. It is particularly useful for identifying and enhancing edges.

## Non-Local Means Filter

The Non-Local Means Filter is an advanced denoising tool that leverages comparisons of non-local image patches to estimate noise levels. This technique effectively removes noise while preserving intricate image details.

## Mean Filter

The Mean Filter, a fundamental smoothing tool, replaces each pixel value with the average of its neighboring pixels. This technique is valuable for reducing noise and creating a subtle blur in the image.

## Gaussian Filter

The Gaussian Filter, a sophisticated smoothing method, utilizes a Gaussian function to assign weights to neighboring pixels. It excels in reducing noise and blurring while maintaining edge preservation.

## Median Filter

The Median Filter, a non-linear approach, replaces each pixel value with the median value of its neighboring pixels. This technique is particularly effective in eliminating impulse noise while retaining sharp edges.

## Inverse Filter

The Inverse Filter is employed in image restoration to recover the original image from a degraded version. It involves dividing the Fourier transform of the degraded image by the Fourier transform of the point spread function.

## Wiener Filter

The Wiener Filter, a powerful signal processing tool, is utilized for image restoration. It minimizes mean square error between the original image and the estimated image, considering noise and degradation characteristics.

## Canny Edge Detection

Canny Edge Detection is a sophisticated algorithm for identifying edges in an image by detecting points with sharp intensity changes. This process involves multiple stages, including gradient calculation, non-maximum suppression, and edge tracking by hysteresis.

