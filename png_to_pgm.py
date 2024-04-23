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

def convert_pgm_to_png(pgm_file_path):
    # Generate the output PNG file path by replacing the PGM extension with PNG
    png_file_path = pgm_file_path.rsplit('.', 1)[0] + '.png'
    
    # Open the PGM image file
    with Image.open(pgm_file_path) as img:
        # Convert the image to RGB ('RGB' mode is required for saving as PNG)
        img_rgb = img.convert('RGB')
        
        # Save the RGB image in PNG format
        img_rgb.save(png_file_path, 'PNG')
        
        print(f"Converted {pgm_file_path} to {png_file_path} successfully.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <conversion_type> <input_file>")
        print("Conversion types: png2pgm, pgm2png")
        sys.exit(1)
    
    conversion_type = sys.argv[1]  # Conversion type: png2pgm or pgm2png
    input_file_path = sys.argv[2]  # Input file path
    
    if conversion_type == 'png2pgm':
        convert_png_to_pgm(input_file_path)
    elif conversion_type == 'pgm2png':
        convert_pgm_to_png(input_file_path)
    else:
        print("Invalid conversion type. Supported types: png2pgm, pgm2png")
        sys.exit(1)