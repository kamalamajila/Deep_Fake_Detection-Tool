from flask import flash, redirect, render_template, request, url_for, get_flashed_messages, Flask, session
from flask_sqlalchemy import SQLAlchemy
import os
import psycopg2
import cv2
from mtcnn import MTCNN
import random
import imblearn
import numpy as np
from skimage.metrics import normalized_root_mse, peak_signal_noise_ratio, structural_similarity
import joblib

db_user = "postgres"
db_password = "Post_Nik18"
db_host = "localhost"
db_port = 5432
db_name = "postgres"

app = Flask(__name__, template_folder="templates")

app.config['SQLALCHEMY_DATABASE_URI'] = f'postgresql://{db_user}:{db_password}@{db_host}/{db_name}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False 
app.secret_key = 'detection'
app.config['SECRET_KEY'] = os.urandom(24)

db = SQLAlchemy()
db.init_app(app)

db_config = {
    'host': db_host,
    'port': db_port,
    'database': db_name,
    'user': db_user,
    'password': db_password
}
connection = psycopg2.connect(host=db_host, user=db_user, password=db_password, database=db_name)
cursor = connection.cursor()

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    fullname = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()

        if user and user.password == password:
            session['user_id'] = user.id
            flash('Logged in successfully!', 'success')
            return redirect(url_for('classify'))
        else:
            flash('Invalid email or password!', 'danger')

    get_flashed_messages()

    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        fullname = request.form['fullname']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm-password']


        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return redirect(url_for('signup'))


        new_user = User(fullname=fullname, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()

        flash('Registered successfully! Please login.', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')


def croping_and_scaling(result, img):
    left_eye_kp = result["keypoints"]["left_eye"]
    right_eye_kp = result["keypoints"]["right_eye"]

    eye_loc_percentage = (0.375, 0.375)
    rescaled_img_width = 224

    del_y_org = right_eye_kp[1] - left_eye_kp[1]
    del_x_org = right_eye_kp[0] - left_eye_kp[0]
    dist_org = np.sqrt((del_y_org * 2) + (del_x_org * 2))

    rescaled_img_left_eye_kp = (eye_loc_percentage[0] * rescaled_img_width, eye_loc_percentage[1] * rescaled_img_width)
    rescaled_img_right_eye_kp = ((1 - eye_loc_percentage[0]) * rescaled_img_width, eye_loc_percentage[1] * rescaled_img_width)

    del_y_rescaled = rescaled_img_right_eye_kp[1] - rescaled_img_left_eye_kp[1]
    del_x_rescaled = rescaled_img_right_eye_kp[0] - rescaled_img_left_eye_kp[0]
    dist_rescaled = np.sqrt((del_y_rescaled) * 2 + (del_x_rescaled) * 2)

    scaling_factor = dist_rescaled / dist_org
    del_y = right_eye_kp[1] - left_eye_kp[1]
    del_x = right_eye_kp[0] - left_eye_kp[0]
    eyes_angle = np.degrees(np.arctan2(del_y, del_x))

    eyes_center = ((left_eye_kp[0] + right_eye_kp[0]) / 2.0, (left_eye_kp[1] + right_eye_kp[1]) / 2.0)
    rotation_matrix = cv2.getRotationMatrix2D(center=eyes_center, angle=eyes_angle, scale=scaling_factor)

    translated_x = rescaled_img_width * 0.5
    translated_y = rescaled_img_left_eye_kp[1]
    rotation_matrix[0, 2] = rotation_matrix[0, 2] + (translated_x - eyes_center[0])
    rotation_matrix[1, 2] = rotation_matrix[1, 2] + (translated_y - eyes_center[1])

    rotated_and_scaled_image = cv2.warpAffine(src=img, M=rotation_matrix, dsize=(rescaled_img_width, rescaled_img_width))
    return rotated_and_scaled_image



def extract_faces_from_video(video_path, num_faces=3):
    detector = MTCNN()
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    faces = []
    
    for idx in range(num_faces):
        while True:
            frame_id = random.randint(0, frame_count-1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()

            if not ret:
                continue

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = detector.detect_faces(img)

            if result:
                rotated_and_scaled_image = croping_and_scaling(result[0], img)
                face_filename = f"frame_{idx + 1}.jpg"  # Sequential filename
                face_path = os.path.join('static', face_filename)
                cv2.imwrite(face_path, rotated_and_scaled_image)
                faces.append(face_path)
                break

    cap.release()
    cv2.destroyAllWindows()
    
    return faces



def extract_faces_from_image(image_path):
    detector = MTCNN()
    faces = []

    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return faces

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = detector.detect_faces(img_rgb)
    if results is None:
        print(f"No faces detected in the image: {image_path}")
        return faces

    for i, result in enumerate(results):
        rotated_and_scaled_image = croping_and_scaling(result, img_rgb)
        face_filename = f"image.jpg"
        face_path = os.path.join('static', face_filename)
        cv2.imwrite(face_path, rotated_and_scaled_image)
        faces.append(face_path)

    return faces



@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if request.method == 'POST':
        image = request.files.get('image')
        video = request.files.get('video')

        if image:
            print("Image uploaded")
            image_path = os.path.join("uploads", image.filename)
            image.save(image_path)

            if os.path.exists(image_path):
                faces = extract_faces_from_image(image_path)
                if faces:
                    face_paths = faces[0]
                else:
                    print(f"No face found")
                    return render_template('main_1.html', error_message="No face found")
                return redirect(url_for("process", face_paths=face_paths))
            else:
                print(f"File {image_path} does not exist.")
                return render_template('main_1.html', error_message="Failed to process video.")



        elif video:
            print("Video uploaded")
            video_path = os.path.join("uploads", video.filename)
            video.save(video_path)

            if os.path.exists(video_path):
                faces = extract_faces_from_video(video_path, 3)
                face_paths = ','.join(faces)
                return redirect(url_for("process", face_paths=face_paths))
            else:
                print(f"File {video_path} does not exist.")
                return render_template('main_1.html', error_message="Failed to process video.")

    return render_template('main_1.html')


@app.route('/process', methods=['GET', 'POST'])
def process():
    face_paths = request.args.get('face_paths')
    print(face_paths)
    face_paths = face_paths.split(',') if face_paths else []
    print(face_paths)

    detector = MTCNN()

    model_path = 'svm_model.pkl'
    model = joblib.load(model_path)

    rotated_and_scaled_image = cv2.imread(face_paths[0])
    if rotated_and_scaled_image is None:
        print(f"Failed to read image: {face_paths[0]}")
        return render_template('main_2.html', error_message="Failed to process video.")
    result = detector.detect_faces(rotated_and_scaled_image)

    blurred_img = cv2.GaussianBlur(rotated_and_scaled_image, (3, 3), 0.5)

    nrmse = normalized_root_mse(rotated_and_scaled_image, blurred_img)
    psnr = peak_signal_noise_ratio(rotated_and_scaled_image, blurred_img, data_range=255)
    ssim = structural_similarity(rotated_and_scaled_image, blurred_img, channel_axis=2, gaussian_weights=True,
                                    sigma=1.5, use_sample_covariance=False, data_range=255, win_size=7)

    hist, bins = np.histogram(rotated_and_scaled_image.ravel(), 32, [0, 255], density=True)

    feature_vector = np.concatenate([[nrmse], [psnr], [ssim], hist])

    result = model.predict([feature_vector])
    
    result = result[0]  

    return render_template('main_2.html', face_paths=face_paths, result=result)


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
