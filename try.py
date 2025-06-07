
from ultralytics import YOLO
from flask import Flask, request, render_template
import numpy as np
import cv2
import os

app = Flask(__name__)

model = YOLO(r'D:/Kathyayinireddy/PROJECTS/petechiae_detection/runs/classify/train4/weights/best.pt')
#model = YOLO('yolov8-cls.pt')

def count_dark_spots(image_path):
    
    grayscale_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply blur to the grayscale image
    blurred_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)

    # Thresholding to identify darker spots
    _, thresholded_image = cv2.threshold(blurred_image, 100, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Count the number of darker spots
    dark_spots_count = len(contours)

    return dark_spots_count


@app.route('/', methods=['GET','POST'])
def index():
    result = None
    image_url = None
    dark_spots_count = None
    pp=None

    if request.method == 'POST':
        uploaded_image = request.files['image_upload']
        if uploaded_image:
            try:
                # Save the uploaded image to the "static" folder
                image_path = os.path.join("static", "temp_image.jpg")
                uploaded_image.save(image_path)
                results = model(image_path)
                names_dict = results[0].names
                probs = results[0].probs.data.tolist()
                result = names_dict[np.argmax(probs)]
                image_url = 'static/temp_image.jpg'
                print(result)
                if result == "With petechiae":
                    # Count darker spots in the image
                    dark_spots_count = count_dark_spots(image_path)
                    if dark_spots_count>10:
                        pp="NEEDS MEDICATION"

            except Exception as e:
                print(f"Error processing image: {e}")

    return render_template('as.html', result=result, image_url=image_url, dark_spots_count=dark_spots_count, status=pp)

# ...
if __name__ == '__main__':
    app.run(debug=True)
