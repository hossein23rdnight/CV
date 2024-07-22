import numpy as np
import cv2

def extract(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    
    
    src_points = np.float32([(x * image.shape[1], y * image.shape[0]) for x, y in points])
    
    
    dst_aspect_ratio = 4.5 #based on the project document assuming the width of the region is 4.5 times its height
    dst_height = 200  
    dst_width = int(dst_aspect_ratio * dst_height)
    
    
    dst_points = np.float32([[0, 0], [dst_width - 1, 0], [dst_width - 1, dst_height - 1], [0, dst_height - 1]])
    
    
    transformation_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    result = cv2.warpPerspective(image, transformation_matrix, (dst_width, dst_height))
    
    
    return result
