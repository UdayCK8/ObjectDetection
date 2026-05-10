from flask import Flask, render_template, request
from ultralytics import YOLO
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = YOLO(r'C:\Users\user\Desktop\project\work\work\best.pt')  # your trained best.pt model

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        image = request.files['image']
        if image:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(image_path)

            # Run YOLOv8 inference
            results = model(image_path)
            results[0].save(filename=image_path)  # overwrite with result

            return render_template('index.html', image=image.filename)

    return render_template('index.html', image=None)

if __name__ == '__main__':
    app.run(debug=True)
