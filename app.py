from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import random
import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Emotion Detection Setup
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_classifier = load_model('model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    emotions = []

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        preds = emotion_classifier.predict(roi)[0]
        emotions.append(preds)
        label = emotion_labels[preds.argmax()]
        probability = preds.max() * 100  # Get the maximum probability
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"{label}: {probability:.2f}%", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    return frame, emotions


def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)
            frame, _ = detect_emotion(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('vhl.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_process', methods=['POST'])
def stop_process():
    # Placeholder implementation for stopping process
    _, emotions = detect_emotion(frame)  # Get emotions from the last processed frame
    avg_probabilities = np.mean(emotions, axis=0)
    return jsonify(probabilities=avg_probabilities.tolist())

@app.route('/emotions/<emotion>')
def show_emotion(emotion):
    return render_template('emotion.html', emotion=emotion)


# Chatbot Setup
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
    return return_list

def get_response(intents_list):
    tag = intents_list[0]['intent']
    for intent_data in intents['intents']:
        if intent_data['tag'] == tag:
            response = random.choice(intent_data['responses'])
            break
    return response

@app.route('/send_message', methods=['POST'])
def send_message():
    message = request.json['message']
    predicted_intents = predict_class(message)
    response = get_response(predicted_intents)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
