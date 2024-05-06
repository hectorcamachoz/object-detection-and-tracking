"""
object-tracking.py

This script is used to detect and object using the ORB algorithm, and track it on a video.

Authors: Héctor Camacho Zamora
Organisation: UDEM
First created on Monday 29 April 2024
"""

# Importing the necessary libraries
import cv2
import argparse
import numpy as np
global passes_to_right
global passes_to_left
global wh
wh = False
passes_to_right = 0 
passes_to_left = 0

def parse_args():
    """
    Parse command line arguments for image paths.
    
    Returns:
        Namespace: Parsed command line arguments with paths to the object and video.
    """
    parser = argparse.ArgumentParser(description='ORB feature matching between two images.')
    parser.add_argument('--img_obj', type=str, help='Path to the first image.')
    parser.add_argument('--video', type=str, help='Path to the second image.')
    return parser.parse_args()

def load_and_resize_image(path: str, scale=0.7) -> np.ndarray:
    """
    Load an image from a file and resize it.
    
    Args:
        path (str): Path to the image file.
        scale (float): Scaling factor for resizing the image.

    Returns:
        np.ndarray: The resized image.
    """
    image = cv2.imread(path) # Load the image
    if image is None:
        raise FileNotFoundError(f"Image at path {path} not found.") # Raise an error if the image is not found
    resized_image = cv2.resize(image, (0, 0), fx=scale, fy=scale) # Resize the image
    return resized_image

def load_video(path: str) -> cv2.VideoCapture:
    """
    Load a video from the given path and return a VideoCapture object.

    Parameters:
        path (str): The path to the video file.

    Returns:
        cv2.VideoCapture: The VideoCapture object representing the loaded video.

    Raises:
        FileNotFoundError: If the video file at the given path is not found.
    """
    vd = cv2.VideoCapture(path)
    if not vd.isOpened():
        raise FileNotFoundError(f"Video at path {path} not found.") # Raise an error if the video is not found
    return vd

def detect_features(image: np.ndarray) -> tuple:
    """
    Detect and compute ORB features and descriptors in the image.
    
    Args:
        image (np.ndarray): Image in which to detect features.

    Returns:
        tuple: Keypoints and descriptors of the image.
    """
    orb = cv2.ORB_create(nfeatures=1000) # Create an ORB object, nfetures is the maximum number of features to retain
    keypoints, descriptors = orb.detectAndCompute(image, None) # Detect keypoints and compute descriptors
    
    return keypoints, descriptors

def match_features(desc1: np.ndarray, desc2: np.ndarray) -> list:
    """
    Match ORB features using the Brute Force matcher.

    Args:
        desc1 (np.ndarray): Descriptors of the first image.
        desc2 (np.ndarray): Descriptors of the second image.

    Returns:
        list: Good matches after applying ratio test.
    """
    bf = cv2.BFMatcher() # Create a Brute Force matcher
    matches = bf.knnMatch(desc1, desc2, k=2) # Match descriptors of the two images
    good_matches = [] # List to store good matches
    for m, n in matches: # Apply ratio test
        if m.distance < 0.5 * n.distance: # If the distance is less than 0.5 times the next closest distance
            good_matches.append([m])
    return good_matches

def draw_matches(image1: np.ndarray, keypoints1: list, image2: np.ndarray,
                  keypoints2: list, matches: list) -> np.ndarray:
    """
    Draw matches between two images.

    Args:
        image1 (np.ndarray): First image.
        keypoints1 (list): Keypoints in the first image.
        image2 (np.ndarray): Second image.
        keypoints2 (list): Keypoints in the second image.
        matches (list): Good matches to draw.

    Returns:
        np.ndarray: Image with drawn matches.
    """
    
    return cv2.drawMatchesKnn(image1, keypoints1, image2, 
                              keypoints2, matches, None, flags=2) # Draw matches

def draw_keypoints(image: np.ndarray, keypoints2: list) -> np.ndarray:
    """
    Draw keypoints as circles on the image.
    
    Args:
        image (np.ndarray): Image on which to draw the keypoints.
        keypoints (list): List of keypoints.
    
    Returns:
        np.ndarray: Image with keypoints drawn.
    """
    return cv2.drawKeypoints(image, keypoints2, None, 
                             color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

def find_centroid(matched_image: np.ndarray, good_matches: list, keypoints2: list) -> tuple:
    """
    Find the centroid of the matched keypoints in an image.

    Args:
        matched_image (np.ndarray): The matched image with keypoints.
        good_matches (list): The list of good matches between keypoints.
        keypoints2 (list): The list of keypoints in the matched image.

    Returns:
        tuple: A tuple containing the matched image with the centroid drawn, 
        the x-coordinate of the centroid, and the y-coordinate of the centroid.
    """
    sum_x = 0
    sum_y = 0
    cx_list = []
    cy_list = []
    num_matches = len(good_matches)

    for match in good_matches:
        kp2 = keypoints2[match[0].trainIdx].pt
        sum_x += kp2[0]
        sum_y += kp2[1]
        

    if num_matches > 0:
        centroid_x = sum_x / num_matches
        centroid_y = sum_y / num_matches
        cx_list.append(centroid_x)
        cy_list.append(centroid_y)
        
        if len(cx_list) > 4:
            smoothed_centroid_x = (1.5 * cx_list[-1] + cx_list[-2] + cx_list[-3]
                                    + cx_list[-4]) / 5
            smoothed_centroid_y = (1.5 * cy_list[-1] + cy_list[-2] + cy_list[-3] 
                                   + cy_list[-4]) / 5

            if abs(centroid_x - smoothed_centroid_x) <= 5 and abs(centroid_y - smoothed_centroid_y) <= 5:
                matched_image = cv2.circle(matched_image, (int(smoothed_centroid_x), 
                                                           int(smoothed_centroid_y)), 
                                                           5, (0, 0, 255), -1)
            else:
                return matched_image
        else:
            matched_image = cv2.circle(matched_image, 
                                       (int(centroid_x), int(centroid_y)),
                                         5, (0, 0, 255), -1)
        return matched_image, centroid_x, centroid_y
    else:
        return None, None, None

def draw_rectangle(image: np.ndarray, centroid_x: int, centroid_y: int) -> np.ndarray:
    """
    Draw a rectangle on the input image around the specified centroid coordinates.

    Args:
        image (np.ndarray): The input image on which the rectangle is to be drawn.
        centroid_x (int): The x-coordinate of the centroid around which the rectangle will be drawn.
        centroid_y (int): The y-coordinate of the centroid around which the rectangle will be drawn.

    Returns:
        np.ndarray: The image with the rectangle drawn around the centroid coordinates.
    """
    if centroid_x is None or centroid_y is None:
        return image
    frame = cv2.rectangle(image, (int(centroid_x) - 100, int(centroid_y) - 50),
                           (int(centroid_x) + 100, int(centroid_y) + 50), 
                           (0, 0, 255), 2)
    return frame

def draw_line(img: np.ndarray) -> np.ndarray:
    """
    Draw a line on the input image from the top to the bottom center.
    
    Args:
        img (np.ndarray): The input image on which to draw the line.
        
    Returns:
        np.ndarray: The image with the line drawn from top center to bottom center.
    """
    height, width, _ = img.shape
    center_x = width // 2
    
    frame = cv2.line(img, (center_x, 0), (center_x,height), (0, 255, 0), 2)
    return frame
def count_passes(image: np.ndarray, centroid_x: int, centroid_y: int) -> np.ndarray:
    """
    Count the number of passes made by an object based on its centroid coordinates.

    Parameters:
        image (numpy.ndarray): The input image.
        centroid_x (int): The x-coordinate of the object's centroid.
        centroid_y (int): The y-coordinate of the object's centroid.

    Returns:
        numpy.ndarray: The input image with text displaying the number of passes to the right and left.

    Note:
        - The function uses global variables `passes_to_right`, `passes_to_left`, and `wh` 
        to keep track of the number of passes and the direction of the passes.
        - If `centroid_x` or `centroid_y` is `None`, the function returns 0.
        - If the object's centroid is to the left of the image's center and `wh` is `True`, 
        the function increments `passes_to_right` and sets `wh` to `False`.
        - If the object's centroid is to the right of the image's center and `wh` is `False`, 
        the function increments `passes_to_left` and sets `wh` to `True`.
        - The function also adds text to the input image displaying the number of passes 
        to the right and left.
    """
    global passes_to_right
    global passes_to_left
    global wh
    
    if centroid_x is None or centroid_y is None:
        return 0
    
    if centroid_x < image.shape[1] // 2 and wh == True:
        passes_to_right += 1
        wh = False
        matched_image = cv2.putText(image, f'Passes to right: {passes_to_right}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        matched_image = cv2.putText(image, f'Passes to left: {passes_to_left}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return matched_image
    
    if centroid_x > image.shape[1] // 2 and wh == False:
        passes_to_left += 1 
        matched_image = cv2.putText(image, f'Passes to right: {passes_to_right}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        matched_image = cv2.putText(image, f'Passes to left: {passes_to_left}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        wh = True
        return matched_image

def run_pipeline():
    """
    Run the pipeline for object tracking and display the matched image with the centroid drawn.
"""
    args = parse_args() # Parse command line arguments
    obj = load_and_resize_image(args.img_obj) # Load and resize the image
    video = load_video(args.video)
    keypoints1, descriptors1 = detect_features(obj) # Detect features in the image 1
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame = draw_line(frame)
        keypoints2, descriptors2 = detect_features(frame)
        good_matches = match_features(descriptors1, descriptors2)
        matched_image = draw_keypoints(frame, keypoints2)
        matched_image, centroid_x, centroid_y = find_centroid(frame, good_matches, keypoints2)
        matched_image = draw_rectangle(matched_image, centroid_x, centroid_y)
        matched_image = count_passes(matched_image, centroid_x, centroid_y)
        if matched_image is not None and centroid_x is not None and centroid_y is not None:
            cv2.imshow('Centroid', matched_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
   run_pipeline()

