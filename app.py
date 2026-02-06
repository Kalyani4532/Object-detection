from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def detect_and_count_objects(image_path):
    """Detect objects in an image and return count + processed image path."""
    image = cv2.imread(image_path)
    if image is None:
        return -1, None   # error case

    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Threshold
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter by area
        min_area = 50
        filtered = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        object_count = len(filtered)

        # Draw contours + text
        cv2.drawContours(image, filtered, -1, (0, 255, 0), 2)
        cv2.putText(image, f"Objects: {object_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Save result
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.jpg')
        cv2.imwrite(result_path, image)

        return object_count, result_path

    except Exception as e:
        print("Error during object detection:", e)
        return -1, None


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)

        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)

        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)

        # Call function
        count, result_path = detect_and_count_objects(image_path)
        if count == -1:
            return "Error processing image."

        return render_template('index.html',
                               count=count,
                               image_url=result_path)

    return render_template('index.html', count=None)


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)

