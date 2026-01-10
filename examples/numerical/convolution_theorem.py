"""
Experiment 20: Convolution theorem demo - blur kernel in spatial vs frequency domain.

Demonstrates that convolution in spatial domain equals
multiplication in frequency domain.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def create_test_image(size=128):
    """Create a test image with sharp features."""
    img = np.zeros((size, size))

    # Add some rectangles
    img[30:50, 30:50] = 1.0
    img[70:100, 20:60] = 0.7
    img[50:80, 80:110] = 0.5

    # Add some lines
    img[20, :] = 1.0
    img[:, 60] = 0.8

    # Add a circle
    y, x = np.ogrid[:size, :size]
    center = (80, 90)
    radius = 15
    mask = (x - center[1])**2 + (y - center[0])**2 <= radius**2
    img[mask] = 1.0

    return img


def gaussian_kernel(size, sigma):
    """Create a Gaussian blur kernel."""
    x = np.arange(size) - size // 2
    kernel_1d = np.exp(-x**2 / (2 * sigma**2))
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    return kernel_2d / kernel_2d.sum()


def box_kernel(size):
    """Create a box (average) blur kernel."""
    kernel = np.ones((size, size))
    return kernel / kernel.sum()


def convolve_spatial(image, kernel):
    """Convolve image with kernel in spatial domain (direct)."""
    from scipy.ndimage import convolve
    return convolve(image, kernel, mode='wrap')


def convolve_frequency(image, kernel):
    """Convolve using FFT (frequency domain)."""
    # Pad kernel to image size
    padded_kernel = np.zeros_like(image)
    kh, kw = kernel.shape
    ih, iw = image.shape

    # Center kernel
    start_h = (ih - kh) // 2
    start_w = (iw - kw) // 2
    padded_kernel[start_h:start_h+kh, start_w:start_w+kw] = kernel

    # Shift kernel to corner (for FFT)
    padded_kernel = np.roll(padded_kernel, -kh//2, axis=0)
    padded_kernel = np.roll(padded_kernel, -kw//2, axis=1)

    # FFT convolution
    image_fft = np.fft.fft2(image)
    kernel_fft = np.fft.fft2(padded_kernel)
    result_fft = image_fft * kernel_fft
    result = np.real(np.fft.ifft2(result_fft))

    return result


def main():
    # Create test image
    size = 128
    image = create_test_image(size)

    # Create kernels
    kernel_gauss = gaussian_kernel(15, 3)
    kernel_box = box_kernel(9)

    # Convolve in both domains
    result_spatial_gauss = convolve_spatial(image, kernel_gauss)
    result_freq_gauss = convolve_frequency(image, kernel_gauss)

    result_spatial_box = convolve_spatial(image, kernel_box)
    result_freq_box = convolve_frequency(image, kernel_box)

    # Create figure
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    # Row 1: Original and Gaussian blur
    ax = axes[0, 0]
    ax.imshow(image, cmap='gray')
    ax.set_title('Original Image')
    ax.axis('off')

    ax = axes[0, 1]
    ax.imshow(kernel_gauss, cmap='hot')
    ax.set_title('Gaussian Kernel')
    ax.axis('off')

    ax = axes[0, 2]
    ax.imshow(result_spatial_gauss, cmap='gray')
    ax.set_title('Spatial Convolution')
    ax.axis('off')

    ax = axes[0, 3]
    ax.imshow(result_freq_gauss, cmap='gray')
    ax.set_title('FFT Convolution')
    ax.axis('off')

    # Row 2: Box blur and difference
    ax = axes[1, 0]
    diff_gauss = np.abs(result_spatial_gauss - result_freq_gauss)
    ax.imshow(diff_gauss, cmap='hot')
    ax.set_title(f'Difference (max={np.max(diff_gauss):.2e})')
    ax.axis('off')

    ax = axes[1, 1]
    ax.imshow(kernel_box, cmap='hot')
    ax.set_title('Box Kernel')
    ax.axis('off')

    ax = axes[1, 2]
    ax.imshow(result_spatial_box, cmap='gray')
    ax.set_title('Box Spatial')
    ax.axis('off')

    ax = axes[1, 3]
    ax.imshow(result_freq_box, cmap='gray')
    ax.set_title('Box FFT')
    ax.axis('off')

    # Row 3: Frequency domain visualization
    ax = axes[2, 0]
    image_fft = np.fft.fftshift(np.fft.fft2(image))
    ax.imshow(np.log(np.abs(image_fft) + 1), cmap='viridis')
    ax.set_title('Image Spectrum (log)')
    ax.axis('off')

    ax = axes[2, 1]
    # Pad and shift kernel for visualization
    padded_kernel = np.zeros_like(image)
    kh, kw = kernel_gauss.shape
    start_h = (size - kh) // 2
    start_w = (size - kw) // 2
    padded_kernel[start_h:start_h+kh, start_w:start_w+kw] = kernel_gauss
    kernel_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded_kernel)))
    ax.imshow(np.abs(kernel_fft), cmap='viridis')
    ax.set_title('Kernel Spectrum')
    ax.axis('off')

    ax = axes[2, 2]
    result_fft = np.fft.fftshift(np.fft.fft2(result_freq_gauss))
    ax.imshow(np.log(np.abs(result_fft) + 1), cmap='viridis')
    ax.set_title('Result Spectrum (log)')
    ax.axis('off')

    # Summary
    ax = axes[2, 3]
    ax.axis('off')

    summary = """Convolution Theorem
==================
Time/Space Domain:
  g(x) = f(x) * h(x)

Frequency Domain:
  G(ω) = F(ω) · H(ω)

Key Properties:
• Convolution ↔ Multiplication
• Spatial blur = low-pass filter
• FFT complexity: O(N log N)
• Direct: O(N²) for each pixel

Numerical Agreement:
  Gaussian: max diff = {:.2e}
  Box:      max diff = {:.2e}

Applications:
• Image filtering/denoising
• Signal processing
• Solving PDEs (e.g., diffusion)
• Deconvolution/deblurring""".format(
        np.max(diff_gauss),
        np.max(np.abs(result_spatial_box - result_freq_box))
    )

    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Convolution Theorem: Spatial vs Frequency Domain',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'convolution_theorem.png'), dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir}/convolution_theorem.png")


if __name__ == "__main__":
    main()
