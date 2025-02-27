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


def extract_dominant_colors(image, num_colors=6):
    """
    Extract the dominant colors from an image using K-means clustering.
    
    Args:
        image (numpy.ndarray): Source image
        num_colors (int): Number of dominant colors to extract
        
    Returns:
        list: List of (color, percentage) tuples where color is a BGR array
    """
    # Reshape the image to be a list of pixels
    pixels = image.reshape(-1, 3).astype(np.float32)
    
    # Define criteria and apply kmeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert centers to uint8
    centers = np.uint8(centers)
    
    # Count labels to find most frequent colors
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Sort by frequency
    sorted_indices = np.argsort(-counts)  # Negative for descending order
    
    # Calculate percentages
    total_pixels = len(pixels)
    dominant_colors = []
    
    for idx in sorted_indices:
        color = centers[idx].tolist()
        percentage = counts[idx] / total_pixels * 100
        dominant_colors.append((color, percentage))
    
    return dominant_colors

def bgr_to_hex(bgr):
    """
    Convert BGR color to hex color string.
    
    Args:
        bgr (list/tuple): BGR color values as [B, G, R]
        
    Returns:
        str: Hex color string in format '#RRGGBB'
    """
    b, g, r = bgr
    return f'#{r:02X}{g:02X}{b:02X}'

def rgb_to_hex(rgb):
    """
    Convert RGB color to hex color string.
    
    Args:
        rgb (list/tuple): RGB color values as [R, G, B]
        
    Returns:
        str: Hex color string in format '#RRGGBB'
    """
    r, g, b = rgb
    return f'#{r:02X}{g:02X}{b:02X}'

def hex_to_bgr(hex_color):
    """
    Convert hex color string to BGR.
    
    Args:
        hex_color (str): Hex color string in format '#RRGGBB' or 'RRGGBB'
        
    Returns:
        tuple: BGR color values as (B, G, R)
    """
    # Remove '#' if present
    if hex_color.startswith('#'):
        hex_color = hex_color[1:]
    
    # Convert hex to RGB then to BGR
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    
    return (b, g, r)

def hex_to_rgb(hex_color):
    """
    Convert hex color string to RGB.
    
    Args:
        hex_color (str): Hex color string in format '#RRGGBB' or 'RRGGBB'
        
    Returns:
        tuple: RGB color values as (R, G, B)
    """
    # Remove '#' if present
    if hex_color.startswith('#'):
        hex_color = hex_color[1:]
    
    # Convert hex to RGB
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    
    return (r, g, b)

def create_color_palette(colors, cell_size=100):
    """
    Create a visualization of color palette with hex values.
    
    Args:
        colors (list): List of (color, percentage) tuples where color is a BGR array
        cell_size (int): Size of each color cell in pixels
        
    Returns:
        numpy.ndarray: Image showing the color palette
    """
    num_colors = len(colors)
    width = cell_size * num_colors
    height = cell_size
    
    # Create blank image
    palette = np.zeros((height, width, 3), dtype=np.uint8)
    
    for i, (color, percentage) in enumerate(colors):
        # Fill the rectangle with the color
        start_x = i * cell_size
        end_x = start_x + cell_size
        
        # Get hex representation
        hex_color = bgr_to_hex(color)
        
        # Fill rectangle with color
        palette[:, start_x:end_x] = color
        
        # Determine text color for better contrast
        brightness = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
        text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
        
        # Add hex value text
        text_position = (start_x + 10, height // 2)
        cv2.putText(palette, hex_color, text_position, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
        
        # Add percentage below
        percentage_text = f'{percentage:.1f}%'
        percentage_position = (start_x + 10, height - 20)
        cv2.putText(palette, percentage_text, percentage_position, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
    
    return palette
