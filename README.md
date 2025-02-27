# Encoding/Decoding Image

This is a simple introduction to understanding how images are encoded and decoded by computers. We will be using the OpenCV library and some utility functions I have made available to you in this repository to inspect images.

## Getting Started

### Clone the Repository

Open your terminal and navigate to a directory where you want to store this project. Then, run the following commands:

```bash
git clone https://github.com/Code-Interfaces/encode_decode_images.git
cd encode_decode_images
```

### Prerequisites

Make sure you have the following installed:

- Python 3.6 or higher
- Set up a virtual environment (optional but recommended)
  - Create: `python -m venv env`
  - Activate:
    - Windows: `.\env\Scripts\activate`
    - macOS/Linux: `source env/bin/activate`
  - Deactivate: `deactivate`
- Required libraries:
  - OpenCV: `pip install opencv-python`
  - NumPy: `pip install numpy`

### Project Structure

The repository has the following structure:

```tree
encode_decode_images/
├── app.py                   # Main application file
├── image/                   # Folder for images
│   └── image.jpeg           # Sample image (you should add this)
├── utils/                   # Utility functions
│   ├── __init__.py          # Makes utils a proper package
│   └── image_utils.py       # Utility functions for image processing
└── README.md                # This file
```

### Add a Test Image

Before running the code, place a JPEG image into the `image` folder and name it `image.jpeg`. You can use any image from your computer.

## Understanding How Images Work

Before diving into the code, let's understand how computers store images:

- **Pixels**: Images are made up of tiny dots called pixels
- **Channels**: Each pixel has color values:
  - Grayscale images: 1 channel (brightness)
  - Color images: 3 channels (Blue, Green, Red in OpenCV)
- **Values**: Each channel has values from 0-255
  - 0 = no intensity of that color
  - 255 = maximum intensity
- **Coordinates**: Pixels are accessed using (y, x) coordinates in OpenCV
  - The top-left corner is (0, 0)

## The Code: Breaking It Down

### Utility Functions

The `utils/image_utils.py` file contains several helper functions that make our main code cleaner and more reusable. Let's explore what each function does:

#### 1. Loading Images

```python
def load_image(image_path):
    """Load an image with error handling"""
    # ...
```

This function loads an image from a file and checks if it was loaded successfully.

#### 2. Getting Image Information

```python
def get_image_info(image):
    """Get basic information about an image"""
    # ...
```

This function extracts details like width, height, and number of color channels from an image.

#### 3. Creating a Pixel Grid Visualization

```python
def create_pixel_grid(image, grid_size=3, cell_size=100, randomize=False):
    """Create a visualization grid of pixels from an image"""
    # ...
```

This is the most interesting function - it creates a visual representation of pixels by:

- Taking pixels from the original image
- Creating enlarged squares filled with each pixel's color
- Arranging them in a grid

#### 4. Saving and Displaying Images

```python
def save_image(image, output_path):
    """Save an image to a file"""
    # ...

def display_image(image, window_name="Image", wait=True):
    """Display an image in a window"""
    # ...
```

These functions handle saving images to disk and displaying them on screen.

### The Main Application

The `app.py` file ties everything together by:

1. Loading an image
2. Getting information about it
3. Examining individual pixel values
4. Creating a visualization of the first 9 pixels
5. Saving and displaying the result

## Running the Application

To run the application:

```bash
python app.py
```

You should see:

1. Console output showing information about your image
2. A window displaying a grid of enlarged pixels
3. A saved image file at `./image/pixels_enlarged.jpg`

## What To Look For

When you run the program:

1. **Notice the pixel values** printed in the console. They're arrays of 3 numbers (Blue, Green, Red values)
2. **Look at the visualization** to see how these values translate to actual colors
3. **Compare adjacent pixels** to see how small changes in values affect the color

## Experimenting with the Code

Now that you understand the basics, try these modifications:

1. **See random pixels**: Change `randomize=False` to `randomize=True` in `app.py`:

   ```python
   pixel_grid = create_pixel_grid(image, grid_size=3, cell_size=100, randomize=True)
   ```

2. **View more pixels**: Increase the grid size (but maybe reduce cell size to fit on screen):

   ```python
   pixel_grid = create_pixel_grid(image, grid_size=5, cell_size=60)
   ```

3. **Try different images**: Replace `image.jpeg` with different images and see how the pixels differ

## Understanding the Output

When you look at the visualization:

- Each large square represents a single pixel from the original image
- The colors you see are exactly what the computer "sees" at each pixel location
- The numbers printed to the console are the actual numeric values that create those colors

## Conclusion

Congratulations! You now understand the basics of how images are represented as pixels in computers. This foundation will help you explore more advanced image processing and computer vision topics.

Remember:

- Images are just arrays of numbers
- OpenCV uses BGR (Blue, Green, Red) order for color values
- Breaking down problems into smaller functions makes code easier to understand
