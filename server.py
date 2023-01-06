import tempfile
import subprocess

from flask import Flask, request, Response

app = Flask(__name__)

@app.route('/', methods=["GET"])
def hello():
    print("Hello")
    return "hello"
@app.route('/predict', methods=['POST'])
def predict():
    print("Got Request.")
    print(request.files)
    # Save the image to a temporary file
    image_data = request.files['image']
    image_file = tempfile.NamedTemporaryFile(suffix='.jpg')
    image_data.save(image_file)
    image_file.flush()
 
    # Run a Python file using a subprocess
    result = subprocess.run(["bash", "script.sh", image_file.name, "0.7"], stdout=subprocess.PIPE)
 
    r = Response(result.stdout, status=200, mimetype="application/json")
    r.headers["Content-Type"] = "application/json"
    return r

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=80)