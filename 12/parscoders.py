import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Function to preprocess the image and extract letters
def extract_letters(image_path, erosion_size=7, dilation_size=3):
    # Check if the file exists
    if not os.path.isfile(image_path):
        print(f"Error: The file at path '{image_path}' does not exist.")
        return

    try:
        # Check the image with PIL (Pillow) to see if it can be opened
        with Image.open(image_path) as pil_image:
            pil_image.verify()  # Verify that it is an image
            print("The file is a valid image and can be opened with PIL.")
            
            # Convert GIF to PNG
            pil_image = Image.open(image_path)  # Reopen the image since verify() closes it
            png_path = image_path.replace('.JPG', '.png')
            pil_image.save(png_path, 'PNG')
            print(f"Image converted to PNG format and saved as {png_path}.")
        
        # Now read the converted PNG with OpenCV
        image = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            print("Error: OpenCV failed to load the converted PNG image.")
            return
        
        # Preprocessing: Apply Gaussian Blur to reduce noise
        blurred_image = cv2.GaussianBlur(image, (7, 7), 0)
        
        # Step 1: Apply adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                cv2.THRESH_BINARY_INV, 11, 2)

        # Step 2: Erode the image to remove noise
        erosion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (erosion_size, erosion_size))
        eroded_image = cv2.erode(adaptive_thresh, erosion_kernel, iterations=1)

        # Step 3: Dilate the image to make contours more distinct
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation_size, dilation_size))
        dilated_image = cv2.dilate(eroded_image, dilation_kernel, iterations=1)

        # Step 4: Connected Component Analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated_image, connectivity=8)
        
        # Extract the sizes of the components and their indices
        sizes = stats[1:, cv2.CC_STAT_AREA]
        indices = np.argsort(sizes)[-6:]  # Get the indices of the 6 largest components

        # Create a directory to save the individual letters
        output_dir = "letters"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save each of the largest components as a separate image
        for i, index in enumerate(indices):
            x, y, w, h, area = stats[index + 1]
            letter_image = image[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(output_dir, f"letter_{i+1}.png"), letter_image)

        # Displaying the binary, eroded, and dilated images using OpenCV
        cv2.imshow('Binary Image', adaptive_thresh)
        cv2.imshow('Eroded Image', eroded_image)
        cv2.imshow('Dilated Image', dilated_image)

        # Wait until a key is pressed then close the windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print(f"Individual letter images saved in directory: {output_dir}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
image_path = '/Users/hossein/Desktop/CV/12/75.JPG'
extract_letters(image_path, erosion_size=4, dilation_size=3)
