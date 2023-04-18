import csv
import io
import os

from VM_CCF22_dev.codes.run import run
import VM_CCF22_dev.config as config
from flask import Flask, request, render_template

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = f'{config.BASE_PATH}/codes/data'
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
