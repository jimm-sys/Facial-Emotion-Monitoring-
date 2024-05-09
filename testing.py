from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import json

app = Flask(__name__)

# Load the pre-trained model and cascade classifier
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_classifier = load_model('model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Function to detect emotions and return emotion probabilities
def detect_emotion(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    
    # Process each face detected
    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) containing the face
        roi_gray = gray[y:y+h, x:x+w]
        
        # Resize the ROI to match the input size of the model
        roi_gray = cv2.resize(roi_gray, (48, 48))
        
        # Convert the resized ROI to a format compatible with the model
        roi = roi_gray.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        # Predict the emotion probabilities for the ROI
        preds = emotion_classifier.predict(roi)[0]
        
        # Normalize the probabilities
        prob_sum = np.sum(preds)
        normalized_preds = [pred / prob_sum for pred in preds]
        
        # Return the normalized emotion probabilities
        return normalized_preds

# Function to generate frames and send emotion percentages
def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Detect emotions and get emotion probabilities
            probabilities = detect_emotion(frame)
            if probabilities is not None:
                # Convert frame and probabilities to JSON format
                data = {
                    'frame': frame.tolist(),
                    'probabilities': probabilities  # No need to convert to list
                }
                # Send frame and emotion probabilities to the front-end using SSE
                yield f"data: {json.dumps(data)}\n\n"
    cap.release()



# Route to render the template
@app.route('/')
def index():
    return render_template('vhl.html')

# Route to stream video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='text/event-stream')

# Route to stop the video feed or ongoing process
@app.route('/stop_process', methods=['POST'])
def stop_process():
    global generate_frames_flag
    generate_frames_flag = False
    return 'Process stopped successfully'

# Route to start the video feed or ongoing process
@app.route('/start_process', methods=['POST'])
def start_process():
    global generate_frames_flag
    generate_frames_flag = True
    return 'Process started successfully'

# Route to get test results
@app.route('/get_results')
def get_results():
    # Return test results as JSON
    return jsonify({'results': test_results})

# Route to update test results
@app.route('/update_results', methods=['POST'])
def update_results():
    global test_results
    # Fetch test results from request data
    test_results = request.form.get('results')
    return 'Results updated successfully'

if __name__ == "__main__":
    app.run(debug=True)
