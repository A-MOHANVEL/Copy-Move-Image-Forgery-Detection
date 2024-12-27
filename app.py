from flask import Flask, render_template, request, redirect, url_for
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def detect_copy_move(img1_path, img2_path, match_ratio=0.7):
    # Read images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Convert to grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Create SIFT object
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)

    # Match keypoint descriptors
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = matcher.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < match_ratio * n.distance and kp1[m.queryIdx].pt != kp2[m.trainIdx].pt:
            good_matches.append(m)

    # Draw matches
    img_matches = cv2.drawMatches(img1_gray, kp1, img2_gray, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Save the result image
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.jpg')
    cv2.imwrite(result_path, img_matches)

    return result_path, len(good_matches)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if files are uploaded
        if 'img1' not in request.files or 'img2' not in request.files:
            return "Please upload both images."

        img1 = request.files['img1']
        img2 = request.files['img2']

        # Save uploaded images
        img1_path = os.path.join(app.config['UPLOAD_FOLDER'], 'img1.jpg')
        img2_path = os.path.join(app.config['UPLOAD_FOLDER'], 'img2.jpg')
        img1.save(img1_path)
        img2.save(img2_path)

        # Perform copy-move detection
        result_path, match_count = detect_copy_move(img1_path, img2_path)

        # Redirect to display result
        return render_template('index.html', result_path=result_path, match_count=match_count)

    return render_template('index.html', result_path=None, match_count=None)

if __name__ == '__main__':
    app.run(debug=True)
