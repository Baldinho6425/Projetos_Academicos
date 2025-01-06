from flask import Flask, render_template, request, redirect, url_for
import os
import json
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
COMMENTS_FILE = 'comments.json'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(COMMENTS_FILE):
    with open(COMMENTS_FILE, 'w') as f:
        json.dump({}, f)

def load_comments():
    with open(COMMENTS_FILE, 'r') as f:
        return json.load(f)

def save_comments(comments):
    with open(COMMENTS_FILE, 'w') as f:
        json.dump(comments, f)

@app.route('/')
def index():
    posts = [{'path': os.path.join(app.config['UPLOAD_FOLDER'], f), 'comments': load_comments().get(f, [])} for f in os.listdir(app.config['UPLOAD_FOLDER'])]
    return render_template('index.html', posts=posts)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('index'))

@app.route('/comment', methods=['POST'])
def comment():
    post_path = request.form['post_path']
    comment = request.form['comment']
    comments = load_comments()
    filename = os.path.basename(post_path)
    if filename not in comments:
        comments[filename] = []
    comments[filename].append(comment)
    save_comments(comments)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)

