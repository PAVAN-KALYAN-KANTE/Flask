from flask import Flask,render_template,Response,request
import cv2
import tensorflow as tf
import numpy as np
import os
import shutil
from detectandstore import detectandupdate
from pushfile import pushIntoFile
from werkzeug.utils import secure_filename
from flask import current_app
from flask import send_file
from getmail import send_mail
from utils import draw_bounding_box
import warnings
warnings.filterwarnings('ignore')


app=Flask(__name__)
model = tf.keras.models.load_model("Models/90_83")
detect_fn = tf.saved_model.load("Models/my_models/saved_model")

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
static_files = ['Aboutb.jpg', 'cam.jpg', 'Classifyb.jpg', 'Classifydoneb.jpg', 'Detectb.jpg', 'Detectdoneb.jpg', 
                'display.css', 'eye.png', 'eye1.png', 'feedbackb.jpg', 'Homeb.jpg', 'loading-page.gif',
                'Picdetectb.jpg', 'Picuploadb.jpg', 'thumbsup.jpg']

def MakeZipFile(filepath):
    shutil.make_archive('./dataset','zip',filepath)

def MakeZipLabel(filepath):
    shutil.make_archive('./labeldata','zip',filepath)

@app.route('/picdelete')
def picdelete():
    for file in os.listdir("static"):
        if file not in static_files:
            os.remove(f"static/{file}")
    return ("nothing")

@app.route('/deletelabel')
def deletelabel():
    os.remove("labeldata.zip")
    return ("nothing")

@app.route('/deletedata')
def deletedata():
    os.remove("dataset.zip")
    return ("nothing")

def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
            
        success, frame = cap.read()
        
        if not success:
            break
        else:
            coordinates = draw_bounding_box(frame, detect_fn)
            for (y, h, x, w) in coordinates:
                cv2.rectangle(frame,(x,y),(w, h),(0, 255, 0),2)
                img = frame[y:h, x:w]
                img = tf.image.resize(img, size = [128, 128])
                pred = model.predict(tf.expand_dims(img, axis=0))
                pred_class = class_names[tf.argmax(pred, axis = 1).numpy()[0]]
                cv2.putText(frame, pred_class, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/webcam')
def webcam():
    return render_template('index.html')

def allowed_file(filename):
    return ('.' in filename and
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)

@app.route('/detectpic', methods=['GET', 'POST'])
def detectpic():
    UPLOAD_FOLDER = 'static'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    if request.method == 'POST':

        file = request.files['file']

        if file and allowed_file(file.filename):

            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            result =detectandupdate(filename)
            return render_template('showdetect.html', orig=result[0], pred=result[1])

@app.route('/picdetect')
def picdetect():
    return render_template('picdetect.html')

@app.route('/bulkdetect', methods=['GET', 'POST'])
def bulkdetect():
    dirs = ["preparedataset","preparedataset/Angry", "preparedataset/Disgust", "preparedataset/Fear", 
    "preparedataset/Happy", "preparedataset/input", "preparedataset/Neutral", "preparedataset/Sad",
    "preparedataset/Surprise"]
    for dir in dirs:
        os.mkdir(dir)
    UPLOAD_FOLDER = './preparedataset/input'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    if request.method == 'POST':
        for file in request.files.getlist('file'):
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                pushIntoFile(filename)
        MakeZipFile('./preparedataset')
        return render_template('donedow.html')

@app.route('/red_to_bulkin')
def red_to_bulkin():
    return render_template('bulkinput.html')


@app.route('/makebound', methods=['GET', 'POST'])
def makebound():
    dirs = ["labeleddata", "labeleddata/input", "labeleddata/output"]
    for dir in dirs:
        os.mkdir(dir)
    UPLOAD_FOLDER = './labeleddata/input'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    if(request.method == 'POST'):
        for file in request.files.getlist('file'):
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                path = "labeleddata/input/" + str(filename)
                image = cv2.imread(path)
                coordinates = draw_bounding_box(image, detect_fn)

                for (y, h, x, w) in coordinates:
                    cv2.rectangle(image,(x,y),(w, h),(0, 255, 0),2)
                    filepath="labeleddata/output/"+ str(filename)[:-4]+'.txt'
                    with open(filepath, 'a') as f:
                        content=str(x) + ' ' + str(y) + ' ' + str(w) + ' '+ str(h) + '\n'
                        f.write(content)
        MakeZipLabel('./labeleddata')          
    return render_template('donelabel.html')

@app.route('/getinlab')
def getinlab():
    return render_template('inputforbound.html')


@app.route('/downloadDS', methods=['GET', 'POST'])
def downloaddataset():
    path='dataset.zip'
    shutil.rmtree("preparedataset")
    return send_file(path,as_attachment=True)

@app.route('/downloadLB', methods=['GET', 'POST'])
def downloadLabel():
    path='labeldata.zip'
    shutil.rmtree("labeleddata")
    return send_file(path,as_attachment=True)

@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

@app.route('/sentsafe',methods=['GET', 'POST'])
def send_sentsafe():
    if request.method == 'POST':
        email = request.form['email']
        comments = request.form['comments']
        name=request.form['name']
        comments=email+"  \n "+name+"+  \n "+comments
        send_mail(email,comments)
    return render_template('sentfeed.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)