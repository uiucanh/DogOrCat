import os

from flask import Flask, request, redirect, url_for, flash, render_template, jsonify
from werkzeug.utils import secure_filename
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import tensorflow as tf

UPLOAD_FOLDER = './api/dataset/single_prediction/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
classifier = load_model('trained_model.h5')
classifier._make_predict_function()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
        UPLOAD_FOLDER=UPLOAD_FOLDER)

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/upload', methods=['POST'])
    def upload():
        file  = request.files['image']
        if allowed_file(file.filename):
            test_image = image.load_img(file, target_size=(64, 64))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            result = classifier.predict(test_image)
            if result[0][0] == 1:
                prediction = 'dog'
            else:
                prediction = 'cat'
            return jsonify(prediction)
        else:
            return jsonify("Not allowed extension"), 500

    return app