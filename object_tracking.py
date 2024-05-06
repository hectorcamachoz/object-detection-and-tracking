"""
object-tracking.py

This script is used to detect and object using the ORB algorithm, and track it on a video.

Authors: Héctor Camacho Zamora
Organisation: UDEM
First created on Monday 29 April 2024
"""

# Importing the necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
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

def load_and_resize_image(path, scale=0.7):
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

def load_video(path):
    vd = cv2.VideoCapture(path)
    if not vd.isOpened():
        raise FileNotFoundError(f"Video at path {path} not found.") # Raise an error if the video is not found
    return vd
def detect_features(image):
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

def match_features(desc1, desc2):
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

def draw_matches(image1, keypoints1, image2, keypoints2, matches):
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
    
    return cv2.drawMatchesKnn(image1, keypoints1, image2, keypoints2, matches, None, flags=2) # Draw matches
def draw_keypoints(image, keypoints2):
    """
    Draw keypoints as circles on the image.
    
    Args:
        image (np.ndarray): Image on which to draw the keypoints.
        keypoints (list): List of keypoints.
    
    Returns:
        np.ndarray: Image with keypoints drawn.
    """
    return cv2.drawKeypoints(image, keypoints2, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

def find_centroid(matched_image, good_matches, keypoints2):
    sum_x = 0
    sum_y = 0
    i = 0
    cx_list = []
    cy_list = []
    num_matches = len(good_matches)

    for match in good_matches:
        kp2 = keypoints2[match[0].trainIdx].pt
        sum_x += kp2[0]
        sum_y += kp2[1]
        
        i += 1

    if num_matches > 0:
        centroid_x = sum_x / num_matches
        centroid_y = sum_y / num_matches
        cx_list.append(centroid_x)
        cy_list.append(centroid_y)
        
        if len(cx_list) > 4:
            smoothed_centroid_x = (1.5 * cx_list[-1] + cx_list[-2] + cx_list[-3] + cx_list[-4]) / 5
            smoothed_centroid_y = (1.5 * cy_list[-1] + cy_list[-2] + cy_list[-3] + cy_list[-4]) / 5

            if abs(centroid_x - smoothed_centroid_x) <= 5 and abs(centroid_y - smoothed_centroid_y) <= 5:
                matched_image = cv2.circle(matched_image, (int(smoothed_centroid_x), int(smoothed_centroid_y)), 5, (0, 0, 255), -1)
            
            else:
                return matched_image
        else:
            matched_image = cv2.circle(matched_image, (int(centroid_x), int(centroid_y)), 5, (0, 0, 255), -1)

        return matched_image, centroid_x, centroid_y
    else:
        return None, None, None

def draw_rectangle(image, centroid_x, centroid_y):
    if centroid_x is None or centroid_y is None:
        return image
    frame = cv2.rectangle(image, (int(centroid_x) - 100, int(centroid_y) - 50), (int(centroid_x) + 100, int(centroid_y) + 50), (0, 0, 255), 2)
    return frame
def draw_line(img):
    height, width, _ = img.shape
    center_x = width // 2
    
    frame = cv2.line(img, (center_x, 0), (center_x,height), (0, 255, 0), 2)
    return frame
def count_passes(image, centroid_x, centroid_y):
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
    args = parse_args() # Parse command line arguments
    obj = load_and_resize_image(args.img_obj) # Load and resize the image
    #obj = cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY)
    video = load_video(args.video)
    keypoints1, descriptors1 = detect_features(obj) # Detect features in the image 1
    while True:
        ret, frame = video.read()
        if not ret:
            break
       # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = draw_line(frame)
        keypoints2, descriptors2 = detect_features(frame)
        good_matches = match_features(descriptors1, descriptors2)
        matched_image = draw_keypoints(frame, keypoints2)
        #image_with_matches = draw_matches(obj, keypoints1, frame, keypoints2, good_matches)
        matched_image, centroid_x, centroid_y = find_centroid(frame, good_matches, keypoints2)
        matched_image = draw_rectangle(matched_image, centroid_x, centroid_y)
        matched_image = count_passes(matched_image, centroid_x, centroid_y)
        if matched_image is not None and centroid_x is not None and centroid_y is not None:
            cv2.imshow('Centroid', matched_image)
            #cv2.imshow('matches', image_with_matches)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
   run_pipeline()

