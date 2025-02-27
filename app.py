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
print(
    f"Image size: {info['width']} pixels wide by {info['height']} pixels tall")
print(
    f"Color channels: {info['channels']} (This means it's a color image with BGR values)")
print(f"Data type: {info['dtype']}")
print(f"Memory size: {info['size_bytes']} bytes")

# Step 3: Look at some pixel values
print("\nStep 3: Let's examine the first few pixels:")
print(f"Top-left pixel: {image[0, 0]}")
print(f"One pixel to the right: {image[0, 1]}")
print(f"One pixel down: {image[1, 0]}")

# Step 4: Create a visualization grid
print("\nStep 4: Creating a visualization grid of the first 9 pixels...")
pixel_grid = create_pixel_grid(
    image, grid_size=3, cell_size=100, randomize=False)

# Step 5: Save and display the result
print("\nStep 5: Saving and showing the pixel visualization")
output_path = './image/pixels_enlarged.jpg'
save_image(pixel_grid, output_path)

print("\nDisplaying the visualization (press any key to close)")
display_image(pixel_grid, "First 9 Pixels Enlarged")
