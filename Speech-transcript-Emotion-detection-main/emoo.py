import requests
import pymongo
import speech_recognition as sr
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
import os
from pydub import AudioSegment
import sounddevice as sd
import tempfile
import scipy.io.wavfile as wav
import spacy
from gensim.models import KeyedVectors

# Ensure NLTK resources are downloaded
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load the spaCy model for syntactic analysis
nlp = spacy.load("en_core_web_sm")

# Load pre-trained Word2Vec model (adjust path accordingly)
word2vec_model_path = 'C:/Users/kungu/Downloads/archive/GoogleNews-vectors-negative300.bin'  # Replace this path with the correct path
word_vectors = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)

# MongoDB Connection to fetch the knowledge base and topic mappings
def connect_to_db():
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["emotion_detection"]
    return db

# Fetch knowledge base from MongoDB
def fetch_knowledge_base():
    db = connect_to_db()
    knowledge_base_collection = db["knowledge_base"]
    knowledge_base_text = []
    knowledge_base_emotion = []

    for record in knowledge_base_collection.find():
        emotion = record["emotion"]
        text = record["text"]
        knowledge_base_text.append(text)
        knowledge_base_emotion.append(emotion)

    return knowledge_base_text, knowledge_base_emotion

# Fetch topic mapping from MongoDB
def fetch_topic_mapping():
    db = connect_to_db()
    topic_mapping_collection = db["topic_mapping"]
    topic_mapping = {}

    for record in topic_mapping_collection.find():
        keyword = record["keyword"].lower()
        topic = record["topic"]
        topic_mapping[keyword] = topic

    return topic_mapping

# Preprocess text data
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    lemmatizer = WordNetLemmatizer()
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split())
    return text

# Convert Audio to Text using SpeechRecognition
def convert_audio_to_text(audio_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            print("Transcribed Text:", text)
            return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand the audio.")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
    except ValueError as e:
        print(f"Audio file could not be read: {e}")
    return ""

# Convert MP3 to WAV using pydub
def convert_mp3_to_wav(mp3_path):
    wav_path = mp3_path.replace('.mp3', '.wav')
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")
    return wav_path

# Train the SVM model using the knowledge base
def train_svm_model(knowledge_base_text, knowledge_base_emotion):
    knowledge_base_text = [preprocess_text(text) for text in knowledge_base_text]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(knowledge_base_text).toarray()
    y = knowledge_base_emotion
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    svm_classifier = SVC(kernel='linear', random_state=42)
    svm_classifier.fit(X_train, y_train)
    y_pred = svm_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"SVM Model Accuracy: {accuracy * 100:.2f}%")
    return svm_classifier, vectorizer

# Build the knowledge base dictionary
def build_knowledge_base_dict(knowledge_base_text, knowledge_base_emotion):
    knowledge_base_dict = {}
    for text, emotion in zip(knowledge_base_text, knowledge_base_emotion):
        text = preprocess_text(text)
        words = set(text.split())
        for word in words:
            if word in knowledge_base_dict:
                knowledge_base_dict[word].append(emotion)
            else:
                knowledge_base_dict[word] = [emotion]
    return knowledge_base_dict

# Map each word to an emotion using the knowledge base and Word2Vec
def map_words_to_emotions(cleaned_text, knowledge_base_dict):
    word_emotion_map = {}
    words = cleaned_text.split()
    for word in words:
        if word in knowledge_base_dict:
            emotions = knowledge_base_dict[word]
            most_common_emotion = Counter(emotions).most_common(1)
            if most_common_emotion:
                word_emotion_map[word] = most_common_emotion[0][0]
            else:
                word_emotion_map[word] = "unknown"
        else:
            try:
                similar_words = word_vectors.most_similar(word, topn=5)
                similar_emotions = []
                for similar_word, _ in similar_words:
                    if similar_word in knowledge_base_dict:
                        similar_emotions.extend(knowledge_base_dict[similar_word])
                if similar_emotions:
                    most_common_emotion = Counter(similar_emotions).most_common(1)
                    word_emotion_map[word] = most_common_emotion[0][0]
                else:
                    word_emotion_map[word] = "unknown"
            except KeyError:
                word_emotion_map[word] = "unknown"
    return word_emotion_map

# Get overall emotion based on word-emotion mapping
def get_overall_emotion(word_emotion_map):
    emotions = [emotion for emotion in word_emotion_map.values() if emotion != "unknown"]
    if emotions:
        overall_emotion = Counter(emotions).most_common(1)[0][0]
    else:
        overall_emotion = "unknown"
    return overall_emotion

# Predict emotion based on SVM model
def predict_emotion(text, classifier, vectorizer):
    text = preprocess_text(text)
    text_vectorized = vectorizer.transform([text]).toarray()
    predicted_emotion = classifier.predict(text_vectorized)
    return predicted_emotion[0]

# Perform semantic and syntactic analysis
def semantic_and_syntactic_analysis(text):
    doc = nlp(text)
    syntactic_info = []
    for token in doc:
        syntactic_info.append({
            'text': token.text,
            'lemma': token.lemma_,
            'POS': token.pos_,
            'dependency': token.dep_,
            'is_stop': token.is_stop
        })
    return syntactic_info

# Extract keywords using TF-IDF from cleaned text
def extract_keywords(cleaned_text, top_n=5):
    vectorizer = TfidfVectorizer(max_features=top_n)
    X = vectorizer.fit_transform([cleaned_text])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = X.toarray().flatten()
    keyword_scores = dict(zip(feature_names, tfidf_scores))
    sorted_keywords = sorted(keyword_scores.items(), key=lambda item: item[1], reverse=True)
    keywords = [keyword for keyword, score in sorted_keywords]
    return keywords

# Identify specific topic based on extracted keywords using database mapping
def identify_specific_topic(cleaned_text, api_key):
    keywords = [word for word in cleaned_text.split() if word.isalpha()]
    topic_mapping = fetch_topic_mapping()
    topics = [topic_mapping.get(keyword.lower(), None) for keyword in keywords]
    topics = [topic for topic in topics if topic]  # Filter out None values

    # If no topic is found in the database, use TextRazor API
    if not topics:
        print("No topic found in the database. Using TextRazor API for topic extraction...")
        textrazor_topics = extract_topics_textrazor(cleaned_text, api_key)
        if textrazor_topics:
            return textrazor_topics[0]  # Returning the first relevant topic
        else:
            return "Miscellaneous"
    else:
        return Counter(topics).most_common(1)[0][0]  # Return the most common topic

# Function to extract topics using TextRazor API
def extract_topics_textrazor(text, api_key):
    url = 'https://api.textrazor.com'
    headers = {'x-textrazor-key': api_key}
    data = {
        'text': text,
        'extractors': 'topics'
    }
    
    response = requests.post(url, headers=headers, data=data)
    
    if response.status_code == 200:
        result = response.json()
        if 'response' in result and 'topics' in result['response']:
            topics = [topic['label'] for topic in result['response']['topics']]
            return topics
        else:
            print("No topics found in the response.")
            return []
    else:
        print(f"Error {response.status_code}: {response.text}")
        return []

# Record live audio and save it to a temporary file
def record_live_audio(duration=5):
    fs = 44100  # Sample rate
    print(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float64')
    sd.wait()  # Wait until recording is finished
    # Save to a temporary WAV file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmpfile:
        wav_path = tmpfile.name
        wav.write(wav_path, fs, (recording * 32767).astype(np.int16))  # Scale to int16
    print(f"Recording saved to {wav_path}")
    return wav_path


def process_audio_for_emotion_detection(audio_path):
    transcribed_text = convert_audio_to_text(audio_path)

    if transcribed_text:
        # Preprocess the transcribed text
        cleaned_text = preprocess_text(transcribed_text)
        print(f"Cleaned Text: {cleaned_text}")  # Add this to check the cleaned text

        knowledge_base_text, knowledge_base_emotion = fetch_knowledge_base()
        svm_classifier, vectorizer = train_svm_model(knowledge_base_text, knowledge_base_emotion)

        predicted_emotion = predict_emotion(transcribed_text, svm_classifier, vectorizer)
        knowledge_base_dict = build_knowledge_base_dict(knowledge_base_text, knowledge_base_emotion)
        word_emotion_map = map_words_to_emotions(cleaned_text, knowledge_base_dict)
        overall_emotion = get_overall_emotion(word_emotion_map)
        syntactic_info = semantic_and_syntactic_analysis(transcribed_text)
        keywords = extract_keywords(cleaned_text)

        # Extract topics using TextRazor if not found in the DB
        api_key = '1c63134ef20f239ac2c2c205089c50491125bb47dc57afc77549d5d1'  # Replace with your actual API key
        specific_topic = identify_specific_topic(cleaned_text, api_key)

        # Prepare results as a dictionary to be passed to the template
        results = {
            'transcribed_text': transcribed_text,
            'cleaned_text': cleaned_text,
            'predicted_emotion': predicted_emotion,
            'word_emotion_map': word_emotion_map,
            'overall_emotion': overall_emotion,
            'syntactic_info': syntactic_info,
            'keywords': keywords,
            'specific_topic': specific_topic
        }

        return results

    else:
        print("No transcribed text available from audio.")
        return None

# Process audio for emotion detection
def process_audio_for_emotion_detection(audio_path):
    transcribed_text = convert_audio_to_text(audio_path)
    
    if transcribed_text:
        # Preprocess the transcribed text
        cleaned_text = preprocess_text(transcribed_text)
        print(f"Cleaned Text: {cleaned_text}")  # Add this to check the cleaned text
        
        knowledge_base_text, knowledge_base_emotion = fetch_knowledge_base()
        svm_classifier, vectorizer = train_svm_model(knowledge_base_text, knowledge_base_emotion)
        
        predicted_emotion = predict_emotion(transcribed_text, svm_classifier, vectorizer)
        knowledge_base_dict = build_knowledge_base_dict(knowledge_base_text, knowledge_base_emotion)
        word_emotion_map = map_words_to_emotions(cleaned_text, knowledge_base_dict)
        overall_emotion = get_overall_emotion(word_emotion_map)
        syntactic_info = semantic_and_syntactic_analysis(transcribed_text)
        keywords = extract_keywords(cleaned_text)

        # Extract chunk relationships (NP, VP, AdjP)
        doc = nlp(transcribed_text)
        np_chunks = [(chunk.text, chunk.root.text, chunk.root.pos_) for chunk in doc.noun_chunks]
        vp_chunks = [(token.text, token.lemma_, token.pos_) for token in doc if token.dep_ in ('VERB', 'AUX')]
        adjp_chunks = [(chunk.text, chunk.root.text, chunk.root.pos_) for chunk in doc.noun_chunks]

        # Display chunk relationships
        print("Chunk Relationships:")
        print("Noun Phrases (NP):")
        for chunk in np_chunks:
            print(f"Text: {chunk[0]}, Root: {chunk[1]}, POS: {chunk[2]}")
        print("Verb Phrases (VP):")
        for chunk in vp_chunks:
            print(f"Text: {chunk[0]}, Lemma: {chunk[1]}, POS: {chunk[2]}")
        print("Adjective Phrases (AdjP):")
        for chunk in adjp_chunks:
            print(f"Text: {chunk[0]}, Root: {chunk[1]}, POS: {chunk[2]}")

        # Extract topics using TextRazor if not found in the DB
        api_key = '1c63134ef20f239ac2c2c205089c50491125bb47dc57afc77549d5d1'  # Replace with your actual API key
        specific_topic = identify_specific_topic(cleaned_text, api_key)

        # Display predicted emotion
        print(f"Predicted Emotion (SVM): {predicted_emotion}")
        
        # Display Word-Emotion Mapping line by line
        print("Word-Emotion Mapping:")
        for word, emotion in word_emotion_map.items():
            print(f"{word}: {emotion}")
        
        # Display overall emotion
        print(f"Overall Emotion: {overall_emotion}")

        # Display Syntactic Information line by line
        print("Syntactic Information:")
        for token_info in syntactic_info:
            print(f"Text: {token_info['text']}, Lemma: {token_info['lemma']}, POS: {token_info['POS']}, Dependency: {token_info['dependency']}, Is Stop Word: {token_info['is_stop']}")

        # Display extracted keywords
        print(f"Keywords: {keywords}")

        # Display the identified topic (either from DB or TextRazor)
        print(f"Identified Topic: {specific_topic}")

    else:
        print("No transcribed text available from audio.")



# Main function for user interaction
def main():
    while True:
        choice = input("Choose an option:\n1. Record Live Audio\n2. Process MP3 File\n3. Exit\n> ")
        if choice == '1':
            duration = int(input("Enter duration of recording in seconds: "))
            audio_path = record_live_audio(duration)
            process_audio_for_emotion_detection(audio_path)
        elif choice == '2':
            mp3_path = input("Enter the path of the Audio file: ")
            if os.path.exists(mp3_path):
                wav_path = convert_mp3_to_wav(mp3_path)
                process_audio_for_emotion_detection(wav_path)
            else:
                print("MP3 file not found.")
        elif choice == '3':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please select again.")

if __name__ == "__main__":
    main()
