import sys
from PIL import Image

def convert_png_to_pgm(png_file_path):
    # Generate the output PGM file path by replacing the PNG extension with PGM
    pgm_file_path = png_file_path.rsplit('.', 1)[0] + '.pgm'
    
    # Open the PNG image file
    with Image.open(png_file_path) as img:
        # Convert the image to grayscale ('L' stands for Luminance)
        img_gray = img.convert('L')
        
        # Save the grayscale image in PGM format
        img_gray.save(pgm_file_path, 'PPM')
        
        print(f"Converted {png_file_path} to {pgm_file_path} successfully.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_png_file>")
        sys.exit(1)
    
    png_file_path = sys.argv[1]  # Command line argument for the PNG file path
    convert_png_to_pgm(png_file_path)
