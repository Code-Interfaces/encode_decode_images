import cv2
import numpy as np
import random
import argparse
from pathlib import Path


class ImageGlitcher:
    """A class to create various glitch effects on images using OpenCV."""

    def __init__(self, image_path):
        """Initialize with an image path."""
        self.original = cv2.imread(image_path)
        if self.original is None:
            raise ValueError(f"Could not load image from {image_path}")
        self.height, self.width = self.original.shape[:2]
        self.glitched = self.original.copy()

    def reset(self):
        """Reset glitched image back to original."""
        self.glitched = self.original.copy()
        return self

    def save(self, output_path):
        """Save the glitched image."""
        cv2.imwrite(output_path, self.glitched)
        print(f"Saved glitched image to {output_path}")
        return self

    def display(self, window_name="Glitched Image"):
        """Display the current glitched image."""
        cv2.imshow(window_name, self.glitched)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return self

    def pixel_sort(self, threshold=100, vertical=False):
        """Sort pixels in rows or columns based on brightness."""
        img = self.glitched.copy()

        # Convert to grayscale for threshold detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if vertical:
            # Process columns
            for x in range(self.width):
                column = img[:, x].copy()
                mask = gray[:, x] > threshold
                # Sort the pixels where mask is True
                sorted_indices = np.argsort(column[mask, 0])
                column[mask] = column[mask][sorted_indices]
                img[:, x] = column
        else:
            # Process rows
            for y in range(self.height):
                row = img[y, :].copy()
                mask = gray[y, :] > threshold
                # Sort the pixels where mask is True
                sorted_indices = np.argsort(row[mask, 0])
                row[mask] = row[mask][sorted_indices]
                img[y, :] = row

        self.glitched = img
        return self

    def channel_shift(self, x_offset=10, y_offset=10, channel=0):
        """Shift a color channel by the specified offset."""
        channels = list(cv2.split(self.glitched))  # Convert tuple to list

        # Create a translation matrix
        M = np.float32([[1, 0, x_offset], [0, 1, y_offset]])

        # Apply the translation to the specified channel
        channels[channel] = cv2.warpAffine(
            channels[channel],
            M,
            (self.width, self.height),
            borderMode=cv2.BORDER_WRAP
        )

        # Merge channels back
        self.glitched = cv2.merge(channels)
        return self

    def data_corruption(self, corruption_amount=0.01):
        """Simulate data corruption by randomly changing byte values."""
        # Create a writable copy of the image array
        img_array = np.frombuffer(
            self.glitched.tobytes(), dtype=np.uint8).copy()

        # Number of bytes to corrupt
        num_to_corrupt = int(corruption_amount * len(img_array))

        # Get random indices
        indices = random.sample(range(len(img_array)), num_to_corrupt)

        # Corrupt the bytes
        for idx in indices:
            img_array[idx] = random.randint(0, 255)

        # Convert back to image
        corrupted = np.reshape(img_array, self.glitched.shape)
        self.glitched = corrupted.copy()  # Ensure we have a writable copy
        return self

    def block_displacement(self, block_size=32, max_offset=20):
        """Randomly displace blocks of the image."""
        result = self.glitched.copy()

        for y in range(0, self.height, block_size):
            for x in range(0, self.width, block_size):
                # Define block boundaries
                y2 = min(y + block_size, self.height)
                x2 = min(x + block_size, self.width)

                # Random displacement
                if random.random() < 0.3:  # Only displace some blocks
                    x_offset = random.randint(-max_offset, max_offset)
                    y_offset = random.randint(-max_offset, max_offset)

                    # Get the block
                    block = self.glitched[y:y2, x:x2].copy()

                    # Calculate destination coordinates
                    dest_x = max(0, min(self.width - (x2 - x), x + x_offset))
                    dest_y = max(0, min(self.height - (y2 - y), y + y_offset))

                    # Place the block
                    result[dest_y:dest_y+(y2-y), dest_x:dest_x+(x2-x)] = block

        self.glitched = result
        return self

    def wave_distortion(self, amplitude=10, frequency=0.1):
        """Apply a wave distortion effect."""
        rows, cols = self.glitched.shape[:2]

        # Create maps for remapping
        map_x = np.zeros((rows, cols), dtype=np.float32)
        map_y = np.zeros((rows, cols), dtype=np.float32)

        for i in range(rows):
            for j in range(cols):
                map_x[i, j] = j + amplitude * np.sin(i * frequency)
                map_y[i, j] = i + amplitude * np.cos(j * frequency)

        # Apply remapping
        self.glitched = cv2.remap(
            self.glitched, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        return self

    def jpeg_artifacts(self, quality=10):
        """Simulate JPEG compression artifacts."""
        # Encode to JPEG in memory
        _, encoded = cv2.imencode('.jpg', self.glitched, [
                                  cv2.IMWRITE_JPEG_QUALITY, quality])

        # Decode back to image
        self.glitched = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return self

    def random_glitch(self):
        """Apply a random combination of glitch effects."""
        effects = [
            lambda: self.channel_shift(random.randint(
                5, 20), random.randint(5, 20), random.randint(0, 2)),
            lambda: self.data_corruption(random.uniform(0.001, 0.03)),
            lambda: self.block_displacement(random.choice(
                [16, 32, 64]), random.randint(10, 40)),
            lambda: self.wave_distortion(
                random.randint(5, 20), random.uniform(0.05, 0.2)),
            lambda: self.jpeg_artifacts(random.randint(5, 30)),
            lambda: self.pixel_sort(random.randint(
                50, 200), random.choice([True, False]))
        ]

        # Apply 2-4 random effects
        for _ in range(random.randint(2, 4)):
            random.choice(effects)()

        return self


def main():
    parser = argparse.ArgumentParser(description='Glitch images using OpenCV')
    parser.add_argument('input', help='Path to the input image')
    parser.add_argument(
        '--output', '-o', help='Path to save the output image', default='glitched.jpg')
    parser.add_argument('--effect', '-e', choices=['channel', 'corrupt', 'block', 'wave', 'jpeg', 'sort', 'random'],
                        default='random', help='Glitch effect to apply')
    args = parser.parse_args()

    # Create glitcher
    glitcher = ImageGlitcher(args.input)

    # Apply selected effect
    if args.effect == 'channel':
        glitcher.channel_shift(15, 10, 2)
    elif args.effect == 'corrupt':
        glitcher.data_corruption(0.02)
    elif args.effect == 'block':
        glitcher.block_displacement(32, 30)
    elif args.effect == 'wave':
        glitcher.wave_distortion(15, 0.1)
    elif args.effect == 'jpeg':
        glitcher.jpeg_artifacts(8)
    elif args.effect == 'sort':
        glitcher.pixel_sort(120)
    else:  # random
        glitcher.random_glitch()

    # Save result
    output_path = args.output
    if Path(output_path).suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
        output_path += '.jpg'

    glitcher.save(output_path)

    # Display result
    glitcher.display()


if __name__ == "__main__":
    main()
