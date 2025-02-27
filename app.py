# A simple program to visualize image pixels using utility functions
from utils.image_utils import load_image, get_image_info, create_pixel_grid, save_image, display_image

# Step 1: Load the image
print("Step 1: Loading the image...")
image_path = './image/image.jpeg'
image = load_image(image_path)

if image is None:
    print("Exiting due to image loading error.")
    exit()

# Step 2: Get image information
print("\nStep 2: Getting image information")
info = get_image_info(image)
# Print out the width, height, channels, data type, and memory size use this syntax: info['key']


# Step 3: Look at some pixel values
# Print out the top-left pixel, one pixel to the right, and one pixel down from the top-left pixel use this syntax: image[y, x]
# where y is the row index and x is the column index
# For example, to access the top-left pixel, you would use image[0, 0]
print("\nStep 3: Let's examine the first few pixels:")


# Step 4: Create a visualization grid
# Create a visualization grid of the first 9 pixels using the create_pixel_grid function
# Set grid_size=3, cell_size=100, and randomize=False
print("\nStep 4: Creating a visualization grid of the first 9 pixels...")


# Step 5: Save and display the result
# Save the pixel visualization to a file using the save_image function
print("\nStep 5: Saving and showing the pixel visualization")
output_path = './image/pixels_enlarged.jpg'


# Display the pixel visualization using the display_image function
# Use the output_path './image/pixels_enlarged.jpg' and window name 'First 9 Pixels Enlarged'
print("\nDisplaying the visualization (press any key to close)")
