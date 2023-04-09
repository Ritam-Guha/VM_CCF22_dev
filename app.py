import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import os, csv
import io
from .codes.run import run

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = f'codes/data'
model = None


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/prediction', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Check if a file was uploaded
        print(request.files)
        if 'file' not in request.files:
            return 'No file uploaded'

        file = request.files['file']
        print(file)

        # Check if the file has a valid filename
        if file.filename == '':
            return 'Invalid file'

        # Save the file to the server
        filename = file.filename
        # contents = file.read()
        # print(type(contents))
        # file.save(os.path.join(app.config['UPLOAD_FOLDER'], f"{filename}"))

        # Save the contents of the file to a CSV file
        with open(os.path.join(app.config['UPLOAD_FOLDER'], f'{filename}'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            content = file.read()
            string_io = io.StringIO(content.decode('utf-8'))

            reader = csv.reader(string_io)
            print(reader)
            for row in reader:
                writer.writerow(row)

        run(filename)

    # output = round(prediction[0], 2)
    return render_template('result.html')


if __name__ == '__main__':
    app.run(debug=True)
