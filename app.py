# A simple program to visualize image pixels using utility functions
from utils.image_utils import load_image, get_image_info, create_pixel_grid, save_image, display_image, create_color_palette

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



# Extra: You can comment out the code we have written so far from line 13, and we can continue with the following code.
# Here we will extract the dominant colors from the image and create a color palette visualization

# Extract dominant colors (default: 6 colors) using the extract_dominant_colors function
# Set num_colors=6
print("\nExtracting dominant colors...")
num_colors = 6


# Create a visualization of the color palette using the create_color_palette function
# Set cell_size=100
print("\nCreating color palette visualization...")

# Save the color palette visualization to a file using the save_image function
output_path = './image/color_palette.jpg'


# Display the color palette visualization using the display_image function
# Use the output_path './image/color_palette.jpg' and window name 'Dominant Color Palette'
print("\nDisplaying the color palette (press any key to close)")