"""
Utility functions for image processing and visualization.
"""
import cv2
import numpy as np
import os


def load_image(image_path):
    """
    Load an image from the given path with error handling.

    Args:
        image_path (str): Path to the image file

    Returns:
        numpy.ndarray: The loaded image, or None if loading failed
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return None

    print(f"Successfully loaded image from {image_path}")
    return image


def get_image_info(image):
    """
    Get basic information about an image.

    Args:
        image (numpy.ndarray): The image to analyze

    Returns:
        dict: Dictionary containing image information
    """
    height, width = image.shape[0], image.shape[1]
    channels = image.shape[2] if len(image.shape) > 2 else 1

    info = {
        'width': width,
        'height': height,
        'channels': channels,
        'dtype': image.dtype,
        'size_bytes': image.nbytes
    }

    return info


def create_pixel_grid(image, grid_size=3, cell_size=100, randomize=False):
    """
    Create a visualization grid of pixels from an image.

    Args:
        image (numpy.ndarray): Source image
        grid_size (int): Size of the grid (3 means 3x3)
        cell_size (int): Size of each cell in pixels
        randomize (bool): If True, select random pixels from the image

    Returns:
        numpy.ndarray: Grid visualization of pixels
    """
    # Create a blank canvas
    total_size = grid_size * cell_size
    grid = np.zeros((total_size, total_size, 3), dtype=np.uint8)

    # Generate random pixel positions if randomize is True
    if randomize:
        # Get random coordinates within image dimensions
        height, width = image.shape[:2]
        y_coords = np.random.randint(0, height, size=(grid_size, grid_size))
        x_coords = np.random.randint(0, width, size=(grid_size, grid_size))

    # Fill in the grid with pixels from the image
    for row in range(grid_size):
        for col in range(grid_size):
            # Calculate cell boundaries
            y_start = row * cell_size
            y_end = y_start + cell_size
            x_start = col * cell_size
            x_end = x_start + cell_size

            if randomize:
                # Get a random pixel from the image
                y_pos, x_pos = y_coords[row, col], x_coords[row, col]
                pixel_color = image[y_pos, x_pos]

                # Add a small label to show the coordinates
                grid[y_start:y_end, x_start:x_end] = pixel_color
            else:
                # Get the pixel from the original image, if available
                if row < image.shape[0] and col < image.shape[1]:
                    # Fill the cell with the pixel color
                    grid[y_start:y_end, x_start:x_end] = image[row, col]

    return grid


def save_image(image, output_path):
    """
    Save an image to the specified path.

    Args:
        image (numpy.ndarray): Image to save
        output_path (str): Path where to save the image

    Returns:
        bool: True if successful, False otherwise
    """
    # Ensure the directory exists
    directory = os.path.dirname(output_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    result = cv2.imwrite(output_path, image)

    if result:
        print(f"Image successfully saved to {output_path}")
        return True
    else:
        print(f"Error: Failed to save image to {output_path}")
        return False


def display_image(image, window_name="Image", wait=True):
    """
    Display an image in a window.

    Args:
        image (numpy.ndarray): Image to display
        window_name (str): Name of the window
        wait (bool): If True, wait for a key press

    Returns:
        int: Key code pressed if wait=True, 0 otherwise
    """
    cv2.imshow(window_name, image)

    if wait:
        key = cv2.waitKey(0)
        cv2.destroyWindow(window_name)
        return key

    return 0
