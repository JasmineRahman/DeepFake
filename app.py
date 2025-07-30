from flask import Flask, render_template, request, redirect, url_for
import os
import requests
from templates.predict_image import detect_deepfake
from templates.predicting_video import predict_video

app = Flask(__name__)
UPLOAD_FOLDER = 'uploaded_videos'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/deepfake_awareness')
def deepfake_awareness():
    return render_template('deepfake_awareness.html')

@app.route('/deepfake_examples')
def deepfake_examples():
    return render_template('deepfake_examples.html')

@app.route('/tutorial_on_spotting_deepfakes')
def tutorial_on_spotting_deepfakes():
    return render_template('tutorial_on_spotting_deepfakes.html')

@app.route('/image_authenticity_checker')
def image_authenticity_checkers():
    return render_template('index.html')

@app.route('/community_reports')
def community_reports():
    return render_template('community_reports.html')

@app.route('/upload_img')
def upload_image():
    return render_template('upload_img.html')

@app.route('/upload_vid')
def upload_video():
    return render_template('upload_vid.html')

@app.route('/quiz')
def quiz():
    return render_template('quiz.html')

@app.route('/newsfeed')
def get_newsfeed():
    api_key = "b88af15a44a34695ac9df0190bee8d74"
    base_url = "https://newsapi.org/v2/everything"

    params = {
        'q': 'deepfake',
        'apiKey': api_key,
        'sortBy': 'publishedAt',
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    if data['status'] == 'ok' and data['totalResults'] > 0:
        articles = data['articles']
    else:
        articles = []

    return render_template('newsfeed.html', news_articles=articles)

@app.route('/user_generated_content_analysis')
def user_generated_content_analysis():
    return render_template('user_generated_content_analysis.html')

@app.route('/image_history')
def image_history():
    return render_template('image_history.html')

@app.route('/video_upload', methods=['GET', 'POST'])
def video_upload():
    if request.method == 'POST':
        video = request.files['file_video']
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
        video.save(video_path)
        result = predict_video(video_path)
        return render_template('upload_vid.html', video_result=result)
    return render_template('upload_vid.html')

@app.route('/image_upload', methods=['GET', 'POST'])
def image_upload():
    if request.method == 'POST':
        image = request.files['file_image']
        detect_deepfake(image)
        return render_template('upload_img.html')
    return render_template('upload_img.html')


if __name__ == '__main__':
    app.run(debug=True)
