from PIL import Image
import os

def compress_images(directory, quality=60):
    """Compress all PNG images in the given directory."""
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            filepath = os.path.join(directory, filename)
            with Image.open(filepath) as img:
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                # Save with compression
                img.save(filepath, 'PNG', optimize=True, quality=quality)
            print(f"Compressed {filename}")

if __name__ == "__main__":
    compress_images("plots")
