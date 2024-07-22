import numpy as np
import cv2

def mask(image: np.ndarray, points: np.ndarray, cover: np.ndarray) -> np.ndarray:


    plate = np.float32([(x * image.shape[1], y * image.shape[0]) for x, y in points])
    cover_height, cover_width = cover.shape[:2]
    cover_corner = np.float32([[0, 0], [cover_width - 1, 0], [cover_width - 1, cover_height - 1], [0, cover_height - 1]])

    matrix = cv2.getPerspectiveTransform(cover_corner, plate)
    transformed_cover = cv2.warpPerspective(cover, matrix, (image.shape[1], image.shape[0]))

    cover_gray = cv2.cvtColor(transformed_cover, cv2.COLOR_BGR2GRAY)
    _, cover_mask = cv2.threshold(cover_gray, 0, 255, cv2.THRESH_BINARY)

    inverted_mask = cv2.bitwise_not(cover_mask)
    base_image_part = cv2.bitwise_and(image, image, mask=inverted_mask)
    cover_part = cv2.bitwise_and(transformed_cover, transformed_cover, mask=cover_mask)

    result = cv2.add(base_image_part, cover_part)

    return result

