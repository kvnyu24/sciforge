import numpy as np
from numba import jit

@jit(nopython=True)
def _mandelbrot_kernel(
    width: int,
    height: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    max_iter: int,
) -> np.ndarray:
    """Numba-accelerated kernel for Mandelbrot set calculation."""
    image = np.zeros((height, width), dtype=np.uint16)
    pixel_width = (x_max - x_min) / width
    pixel_height = (y_max - y_min) / height

    for i in range(height):
        for j in range(width):
            c = complex(x_min + j * pixel_width, y_min + i * pixel_height)
            z = 0
            n = 0
            while abs(z) <= 2 and n < max_iter:
                z = z * z + c
                n += 1
            image[i, j] = n

    return image

def generate_mandelbrot_set(
    width: int,
    height: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    max_iter: int,
) -> np.ndarray:
    """
    Generates the Mandelbrot set for a given view and resolution.

    Args:
        width: The width of the output image in pixels.
        height: The height of the output image in pixels.
        x_range: A tuple (min, max) for the real axis.
        y_range: A tuple (min, max) for the imaginary axis.
        max_iter: The maximum number of iterations to determine if a point
                  has escaped. Higher values increase detail and computation time.

    Returns:
        A 2D numpy array of integers, where each value represents the
        number of iterations before the point escaped (or max_iter if it did not).
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    return _mandelbrot_kernel(width, height, x_min, x_max, y_min, y_max, max_iter)

@jit(nopython=True)
def generate_mandelbrot_tile(
    x_min: float, y_min: float,
    pixel_width: float, pixel_height: float,
    tile_width: int, tile_height: int, max_iter: int
) -> np.ndarray:
    """Numba-accelerated function to compute a single tile."""
    image = np.zeros((tile_height, tile_width), dtype=np.uint16)
    for i in range(tile_height):
        for j in range(tile_width):
            c = complex(x_min + j * pixel_width, y_min + i * pixel_height)
            z = 0
            n = 0
            while abs(z) <= 2 and n < max_iter:
                z = z * z + c
                n += 1
            image[i, j] = n
    return image

@jit(nopython=True)
def _julia_kernel(
    c: complex,
    width: int,
    height: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    max_iter: int,
) -> np.ndarray:
    """Numba-accelerated kernel for Julia set calculation."""
    image = np.zeros((height, width), dtype=np.uint16)
    pixel_width = (x_max - x_min) / width
    pixel_height = (y_max - y_min) / height

    for i in range(height):
        for j in range(width):
            z = complex(x_min + j * pixel_width, y_min + i * pixel_height)
            n = 0
            while abs(z) <= 2 and n < max_iter:
                z = z * z + c
                n += 1
            image[i, j] = n
    return image

def generate_julia_set(
    c: complex,
    width: int,
    height: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    max_iter: int,
) -> np.ndarray:
    """
    Generates a Julia set for a given complex constant 'c'.

    Args:
        c: The complex constant that defines the Julia set.
        width: The width of the output image in pixels.
        height: The height of the output image in pixels.
        x_range: A tuple (min, max) for the real axis.
        y_range: A tuple (min, max) for the imaginary axis.
        max_iter: The maximum number of iterations.

    Returns:
        A 2D numpy array of escape times, similar to the Mandelbrot set.
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    return _julia_kernel(c, width, height, x_min, x_max, y_min, y_max, max_iter) 