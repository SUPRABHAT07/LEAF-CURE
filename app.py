import os
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')

# ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# load model
MODEL_PATH = 'banana_leaf_model.h5'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Trained model not found at {MODEL_PATH} - run train_model.py first.")
model = load_model(MODEL_PATH)

# auto-detect class names from dataset/train folders (sorted for stable order)
dataset_train = os.path.join('dataset', 'train')
if not os.path.exists(dataset_train):
    raise FileNotFoundError(f"Dataset train folder not found at: {dataset_train}")
class_names = sorted([d for d in os.listdir(dataset_train) if os.path.isdir(os.path.join(dataset_train, d))])
print("Class names:", class_names)

IMG_SIZE = (128, 128)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ''
    img_url = ''
    if request.method == 'POST':
        if 'leaf_image' not in request.files:
            result = "No file part"
            return render_template('index.html', result=result, img_url=img_url)

        f = request.files['leaf_image']
        if f.filename == '':
            result = "No selected file"
            return render_template('index.html', result=result, img_url=img_url)

        # save file
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.save(save_path)
        img_url = save_path.replace("\\", "/")

        # preprocess and predict
        img_obj = image.load_img(save_path, target_size=IMG_SIZE)
        img_arr = image.img_to_array(img_obj) / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)

        preds = model.predict(img_arr)
        pred_idx = int(np.argmax(preds, axis=-1)[0])
        result = class_names[pred_idx]

    return render_template('index.html', result=result, img_url=img_url)

if __name__ == '__main__':
    app.run(debug=True)
