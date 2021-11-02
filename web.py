import os
from flask import Flask,request
from Display import Display
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'upload/'
@app.route('/')
def hello_world():
    return 'Hello World!'
@app.route('/api/v2/textupload',methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        text = request.form.get('text')
        print(text)
        path_list = display.run(text)
        return path_list
    elif request.method == 'GET':
        return 'success'
    else:
        return 'error'
if __name__ == '__main__':
    path = './models/11-02-10:03_DCMH_IR/179.pth.tar'
    display = Display(path)
    app.run(host='0.0.0.0', port=5100, debug=True)