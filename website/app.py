# from flask import Flask, request, render_template
# from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
# from keras.preprocessing import image
# import numpy as np

# app = Flask(__name__)

# # Load a pre-trained VGG16 model
# model = VGG16(weights='imagenet')

# # Define a function to preprocess and classify an image
# def classify_image(img_path):
#     img = image.load_img(img_path, target_size=(224, 224))
#     img = image.img_to_array(img)
#     img = np.expand_dims(img, axis=0)
#     img = preprocess_input(img)
#     predictions = model.predict(img)
#     decoded_predictions = decode_predictions(predictions, top=3)[0]
#     return decoded_predictions

# @app.route("/", methods=["GET", "POST"])
# def index():
#     if request.method == "POST":
#         # Check if an image was uploaded
#         if "image" in request.files:
#             img = request.files["image"]
#             img_path = "static/2-rotated1.jpg"
#             img.save(img_path)
#             predictions = classify_image(img_path)
#             return render_template("index.html", image_path=img_path, predictions=predictions)

#     return render_template("index.html", image_path=None, predictions=None)

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the pre-trained CNN model
model = load_model('model/bone_classifier.h5')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/classify', methods=['POST', 'GET'])
def classify():
    if 'image' not in request.files:
        return "No file part"

    image_file = request.files['image']

    if image_file.filename == 'static/dataset/':
        return "No selected file"

    if image_file:
        image_path = f"static/{image_file.filename}"
        image_file.save(image_path)

        img = image.load_img(image_path, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)

        # Preprocess the image
        img = img / 5.0  # Depending on the preprocessing required for your model

        # Make predictions using the loaded model
        predictions = model.predict(img)
        print(predictions)
        ret = ""
        if (predictions < 0.5):
            ret = "Fracture"
        else:
            ret = "Not Fracture"

        # Process predictions as needed
        # You can return the top classes or any other information

        return "Predictions: " + ret


if __name__ == '__main__':
    app.run(debug=True)
