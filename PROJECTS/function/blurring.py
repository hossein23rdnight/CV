import numpy as np
import cv2
from extract import extract
from masking import mask


def blur(image: np.ndarray, points: np.ndarray) -> np.ndarray:

    extracted = extract(image, points)

    blurred = cv2.GaussianBlur(extracted, (57, 57), 0)

    masked_image = mask(image, points, blurred)

    return masked_image
