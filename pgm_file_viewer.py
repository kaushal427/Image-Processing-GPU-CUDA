import sys
from PIL import Image
import matplotlib.pyplot as plt

def display_pgm(files):
    # Determine the number of files to set the figure width dynamically
    n_files = len(files)
    plt.figure(figsize=(5 * n_files, 5))  # Adjust the figure size based on the number of images

    for i, file_path in enumerate(files):
        try:
            # Open the image file
            with Image.open(file_path) as img:
                plt.subplot(1, n_files, i + 1)  # Arrange subplots in 1 row
                plt.imshow(img, cmap='gray')  # Display image in grayscale
                plt.title(f'File: {file_path}')  # Set the title to the file name
                plt.axis('off')  # Turn off axis numbers and ticks
        except IOError:
            print(f"Failed to open {file_path}. Ensure the file exists and is a valid PGM file.")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        pgm_files = [file for file in sys.argv[1:] if file.lower().endswith('.pgm')]
        if pgm_files:
            display_pgm(pgm_files)
        else:
            print("No valid PGM files provided.")
    else:
        print("Usage: python script_name.py <path_to_pgm_file_1> <path_to_pgm_file_2> ...")
