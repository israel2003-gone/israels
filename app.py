import os
import re
import random
import sqlite3
from flask import Flask, render_template, request, redirect, url_for
import spacy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from fuzzywuzzy import fuzz

# Download required NLTK data
nltk.download('vader_lexicon')

# Setup
app = Flask(__name__)

# --- Intent definitions ---
intents = {
    "greeting": ["hello", "hi", "hey"],
    "goodbye": ["bye", "goodbye", "see you later"],
    "thank_you": ["thanks", "thank you"],
    "ask_knowledge": ["tell me about", "what is", "explain"]
}

# --- Train intent classifier ---
X, y = [], []
for intent, examples in intents.items():
    for example in examples:
        X.append(example)
        y.append(intent)

vectorizer = TfidfVectorizer()
X_vectors = vectorizer.fit_transform(X)

classifier = LogisticRegression(solver='liblinear', random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X_vectors, y, test_size=0.2, random_state=0)
classifier.fit(X_train, y_train)

# --- Load NLP models ---
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

analyzer = SentimentIntensityAnalyzer()

# --- Database configuration ---
LEFT_DATABASE = "knowledged_base_left.db"

def create_knowledged_base_table(db_file):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS knowledged_base (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            response TEXT NOT NULL,
            image_path TEXT,
            video_url TEXT
        )
    """)
    conn.commit()
    conn.close()

def insert_knowledged_data(db_file, data=None):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.executemany("""
        INSERT INTO knowledged_base (query, response, image_path, video_url)
        VALUES (?, ?, ?, ?)
    """, data)
    conn.commit()
    conn.close()

def populate_db_if_empty(db_file, data):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM knowledged_base")
    count = cursor.fetchone()[0]
    conn.close()
    if count == 0:
        insert_knowledged_data(db_file, data)

def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

def get_knowledged_response(query, db_file, threshold=80):
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        query = remove_punctuation(query.lower())
        cursor.execute("SELECT query, response, image_path, video_url FROM knowledged_base")
        rows = cursor.fetchall()

        best_match = None
        best_score = 0

        for db_query, db_response, db_image, db_video in rows:
            score = fuzz.ratio(query, remove_punctuation(db_query.lower()))
            if score > best_score:
                best_score = score
                best_match = (db_response, db_image, db_video)

        if best_score >= threshold:
            return best_match[0], best_match[1], best_match[2]
        else:
            return "I don't have information about that.", None, None

    except sqlite3.Error as e:
        print(f"Error retrieving knowledge: {e}")
        return "Database error.", None, None
    finally:
        if conn:
            conn.close()

def predict_intent(user_input):
    user_vector = vectorizer.transform([user_input])
    return classifier.predict(user_vector)[0]

def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def get_sentiment(text):
    score = analyzer.polarity_scores(text)['compound']
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

def generate_rag_responses(intent, entities, sentiment, user_message):
    if intent == "ask_knowledge":
        return get_knowledged_response(user_message, LEFT_DATABASE)
    else:
        return random.choice(["I see.", "That's interesting."]), None, None

# --- Flask Web Routes ---
left_messages = []

@app.route('/', methods=['GET', 'POST'])
def index():
    global left_messages
    if request.method == 'POST':
        user_input = request.form['user_input']
        intent = predict_intent(user_input)
        entities = extract_entities(user_input)
        sentiment = get_sentiment(user_input)
        response, image, video = generate_rag_responses(intent, entities, sentiment, user_input)

        left_messages.append({"type": "user-message", "text": user_input})
        left_messages.append({"type": "screen-message", "text": response, "image": image, "video": video})

        return render_template('index.html', left_messages=left_messages)

    left_messages = []
    return render_template('index.html', left_messages=[])

@app.route('/clear_messages', methods=['POST'])
def clear_messages():
    global left_messages
    left_messages = []
    return redirect(url_for('index'))

# --- App Initialization ---
if __name__ == '__main__':
    os.makedirs(os.path.join(app.static_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(app.static_folder, 'videos'), exist_ok=True)

    create_knowledged_base_table(LEFT_DATABASE)

    example_data = [
        ("what is a cat?",
         "<h4>About Cats:</h4><ul><li>Feline animal</li><li>Domesticated pet</li></ul>",
         "images/cat.jpg", None),
        ("what is python?",
         "<p>Python is a powerful language for web, AI, and scripting.</p>",
         None, "videos/python_intro.MP4"),
        ("what is a dog?",
         "<b>Dogs</b> are loyal animals. They love: <ul><li>Walks</li><li>Food</li><li>You</li></ul>",
         "images/dog.jpg", None),
        ("what is the capital of France?",
         "<p>The capital is <strong>Paris</strong>.</p>", None, None)
    ]

    populate_db_if_empty(LEFT_DATABASE, example_data)

    # Use port 3000 for Glitch compatibility
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 3000)))
