# voice_assistant_flask_rag_lazy.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
import json
import base64
import tempfile
import speech_recognition as sr
import pyttsx3
from pydub import AudioSegment
import time
import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import threading

# OpenAI compatibility
try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY', ''))
    NEW_OPENAI = True
    print("✅ Using new OpenAI library (v1.0+)")
except (ImportError, AttributeError):
    import openai
    openai.api_key = os.getenv('OPENAI_API_KEY', '')
    NEW_OPENAI = False
    print("✅ Using old OpenAI library")

MODEL = 'gpt-4o-mini'

# Google Sheets Configuration
SERVICE_ACCOUNT_FILE = r"C:\industrialproject\voice-assistant-main\credentials.json"
SHEET_URL = 'https://docs.google.com/spreadsheets/d/1mHgBAgTNYKPtN3CQS5DTxir5mKZ5jfdgu6dPH-X8nKo/edit?usp=sharing'

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'temp_audio'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables
property_db = pd.DataFrame()
broker_db = pd.DataFrame()
property_collection = None
embedding_func = None
rag_initialized = False
initialization_lock = threading.Lock()

# Session management
sessions = {}
questions = [
    ("name", "What is your full name?"),
    ("contact", "Please say your contact number."),
    ("rent_or_buy", "Are you looking to rent or buy a property?"),
    ("location", "Which location are you looking for the property in?"),
    ("budget", "What is your budget?"),
    ("availability_date", "When would you be available to see the finalized properties?")
]

class Session:
    def __init__(self):
        self.stage = 'greeting'
        self.question_index = 0
        self.user_data = {}
        self.retry_count = 0
        self.history = []
        self.preference_questions = []
        self.preference_index = 0
        self.user_location = ""
        self.properties_shown = False

class SimpleTfidfEmbedding:
    """Simple TF-IDF based embedding function"""
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=384, stop_words='english')
        self.is_fitted = False
    
    def fit(self, documents):
        self.vectorizer.fit(documents)
        self.is_fitted = True
    
    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        if not self.is_fitted:
            self.fit(input)
        try:
            embeddings = self.vectorizer.transform(input).toarray()
            return embeddings.tolist()
        except:
            return [[0.0] * 384 for _ in input]

def initialize_rag_background():
    """Initialize RAG in background thread"""
    global property_db, broker_db, property_collection, embedding_func, rag_initialized
    
    with initialization_lock:
        if rag_initialized:
            return
        
        try:
            print("\n🔄 Initializing RAG system in background...")
            
            # Initialize embedding function
            embedding_func = SimpleTfidfEmbedding()
            
            # Initialize ChromaDB
            chroma_client = chromadb.Client()
            try:
                property_collection = chroma_client.create_collection(
                    name="properties",
                    embedding_function=embedding_func
                )
            except:
                chroma_client.delete_collection(name="properties")
                property_collection = chroma_client.create_collection(
                    name="properties",
                    embedding_function=embedding_func
                )
            
            # Load property database
            prop_path = r"C:\industrialproject\voice-assistant-main\propdata.csv"
            if os.path.exists(prop_path):
                property_db = pd.read_csv(prop_path)
                print(f"✅ Loaded {len(property_db)} properties")
                
                # Index to vector database
                documents = []
                metadatas = []
                ids = []
                
                for idx, row in property_db.iterrows():
                    doc_text = f"""Property Type: {row.get('Property Type', 'N/A')} Location: {row.get('Location', 'N/A')} Building: {row.get('Building Name', 'N/A')} Rooms: {row.get('No. of Rooms', 'N/A')} Area: {row.get('Area (sqft)', 'N/A')} sqft"""
                    documents.append(doc_text.strip())
                    
                    metadata = {
                        'location': str(row.get('Location', 'N/A')),
                        'property_type': str(row.get('Property Type', 'N/A')),
                        'building_name': str(row.get('Building Name', 'N/A')),
                        'rooms': str(row.get('No. of Rooms', 'N/A')),
                        'area_sqft': str(row.get('Area (sqft)', 'N/A')),
                        'purchase_price': float(row.get('Purchase Price (USD)', 0)),
                        'monthly_rent': float(row.get('Monthly Rent (USD)', 0)),
                        'parking_spots': int(row.get('Parking Spots', 0))
                    }
                    metadatas.append(metadata)
                    ids.append(f"property_{idx}")
                
                # Fit vectorizer
                embedding_func.fit(documents)
                print("✅ Vectorizer trained")
                
                # Add to ChromaDB in batches
                batch_size = 50
                for i in range(0, len(documents), batch_size):
                    end_idx = min(i + batch_size, len(documents))
                    property_collection.add(
                        documents=documents[i:end_idx],
                        metadatas=metadatas[i:end_idx],
                        ids=ids[i:end_idx]
                    )
                    print(f"   Indexed batch {i//batch_size + 1}")
                
                print(f"✅ Indexed {len(documents)} properties")
            
            # Load broker database
            broker_path = r"C:\industrialproject\voice-assistant-main\brokdata.csv"
            if os.path.exists(broker_path):
                broker_db = pd.read_csv(broker_path)
                print(f"✅ Loaded {len(broker_db)} brokers")
            
            rag_initialized = True
            print("✅ RAG system ready!\n")
            
        except Exception as e:
            print(f"❌ Error initializing RAG: {e}")
            import traceback
            traceback.print_exc()

# Start RAG initialization in background thread
print("\n" + "="*50)
print("🚀 Starting Flask Server")
print("="*50)
initialization_thread = threading.Thread(target=initialize_rag_background, daemon=True)
initialization_thread.start()
print("✅ Server starting (RAG loading in background)...")
print("="*50 + "\n")

# -------------------- UTILITY FUNCTIONS --------------------

def text_to_speech_bytes(text):
    try:
        engine = pyttsx3.init()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
            temp_name = tf.name
        engine.save_to_file(text, temp_name)
        engine.runAndWait()
        with open(temp_name, "rb") as f:
            audio_data = f.read()
        os.remove(temp_name)
        return base64.b64encode(audio_data).decode('utf-8')
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

def speech_to_text(wav_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(wav_path) as source:
            audio = recognizer.record(source)
            return recognizer.recognize_google(audio)
    except Exception as e:
        print(f"STT Error: {e}")
        return ""

def call_openai_api(messages, temperature=0, max_tokens=400):
    try:
        if NEW_OPENAI:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        else:
            response = openai.ChatCompletion.create(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"OpenAI error: {e}")
        return None

def search_properties_with_rag(user_question, n_results=5):
    """Use RAG to search properties"""
    if not rag_initialized or property_collection is None:
        print("⚠️ RAG not ready, using fallback search")
        return []
    
    try:
        results = property_collection.query(
            query_texts=[user_question],
            n_results=n_results
        )
        
        matching_properties = []
        if results['metadatas'] and len(results['metadatas'][0]) > 0:
            for metadata in results['metadatas'][0]:
                matching_properties.append(metadata)
        
        return matching_properties
    except Exception as e:
        print(f"RAG search error: {e}")
        return []

def answer_property_question_with_rag(user_question):
    """Answer property questions using RAG"""
    if property_db.empty:
        return "I'm still loading the property database. Please try again in a moment."
    
    matching_properties = search_properties_with_rag(user_question, n_results=3)
    
    if not matching_properties:
        prompt = f"""A user asked: "{user_question}"
Give a helpful real estate response as Agent Shreyash (under 100 words)."""
        ai_reply = call_openai_api([{"role": "user", "content": prompt}])
        return ai_reply if ai_reply else "Could you be more specific about your property needs?"
    
    # Format results
    result_text = f"I found {len(matching_properties)} matching properties:\n"
    for i, prop in enumerate(matching_properties, 1):
        result_text += f"\n{i}. {prop.get('building_name', 'Property')} in {prop.get('location')}"
        result_text += f", {prop.get('property_type')} with {prop.get('rooms')} rooms"
        result_text += f", {prop.get('area_sqft')} sqft"
        
        rent = prop.get('monthly_rent', 0)
        price = prop.get('purchase_price', 0)
        if rent > 0:
            result_text += f", rent ${rent:.0f}/month"
        elif price > 0:
            result_text += f", ${price:,.0f}"
    
    return result_text

def generate_preference_questions(location):
    prompt = f"""Generate 4 real estate preference questions for properties in {location}.
Keep brief, one per line. Return ONLY questions."""
    try:
        response = call_openai_api([{"role": "user", "content": prompt}])
        if response:
            questions = [q.strip().lstrip('- 1234567890. ') for q in response.split('\n') if q.strip()]
            return [q for q in questions if len(q) > 10][:4]
        return []
    except:
        return []

def assign_broker(user_data):
    if broker_db.empty:
        return None
    
    user_location = user_data.get('location', '').lower().strip()
    if not user_location:
        return None
    
    try:
        exact_matches = broker_db[broker_db['Location'].str.lower().str.contains(user_location, na=False)]
        if not exact_matches.empty:
            broker = exact_matches.iloc[0]
            return {
                'Assigned_Broker_Name': str(broker.get('Agent Name', '')),
                'Assigned_Broker_Phone': str(broker.get('Phone Number', '')),
                'Assigned_Broker_Email': str(broker.get('Email', '')),
                'Assigned_Broker_Location': str(broker.get('Location', ''))
            }
    except:
        pass
    
    if len(broker_db) > 0:
        broker = broker_db.iloc[0]
        return {
            'Assigned_Broker_Name': str(broker.get('Agent Name', '')),
            'Assigned_Broker_Phone': str(broker.get('Phone Number', '')),
            'Assigned_Broker_Email': str(broker.get('Email', '')),
            'Assigned_Broker_Location': str(broker.get('Location', ''))
        }
    return None

def save_to_google_sheet(data_dict):
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(SERVICE_ACCOUNT_FILE, scope)
        client = gspread.authorize(creds)
        sheet = client.open_by_url(SHEET_URL)
        worksheet = sheet.sheet1
        
        row = [
            data_dict.get("name", ""),
            data_dict.get("contact", ""),
            data_dict.get("rent_or_buy", ""),
            data_dict.get("location", ""),
            data_dict.get("budget", ""),
            data_dict.get("availability_date", ""),
            data_dict.get("preferences_summary", ""),
            data_dict.get("Assigned_Broker_Name", ""),
            data_dict.get("Assigned_Broker_Phone", ""),
            data_dict.get("Assigned_Broker_Email", ""),
            data_dict.get("Assigned_Broker_Location", ""),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ]
        
        worksheet.append_row(row)
        return True
    except Exception as e:
        print(f"Google Sheet error: {e}")
        return False

def handle_form_completion(session):
    assigned_broker = assign_broker(session.user_data)
    
    if assigned_broker:
        session.user_data.update(assigned_broker)
        broker_name = assigned_broker.get('Assigned_Broker_Name', 'our team')
        response_text = f"Perfect! I've assigned you to {broker_name}. They'll contact you within 24 hours."
    else:
        response_text = "Thank you! Our team will contact you within 24 hours."
    
    save_to_google_sheet(session.user_data)
    response_text += "\n\nDo you have questions about properties?"
    return response_text

def is_affirmative_response(text):
    affirmative = ['yes', 'yeah', 'yep', 'ok', 'okay', 'sure', 'alright']
    return any(word in text.lower() for word in affirmative)

def is_negative_response(text):
    negative = ['no', 'nope', 'not', 'never', 'nothing']
    return any(word in text.lower() for word in negative)

# -------------------- API ROUTES --------------------

@app.route('/api/voice-init', methods=['GET'])
def voice_init():
    session_id = f"session_{int(time.time())}"
    sessions[session_id] = Session()
    
    intro = "Hi! I'm Agent Shreyash, your AI-powered real estate assistant. How are you today?"
    
    return jsonify({
        'success': True,
        'session_id': session_id,
        'agent_response': intro,
        'audio_response': text_to_speech_bytes(intro),
        'status': 'Listening...'
    })

@app.route('/api/voice-chat', methods=['POST'])
def voice_chat():
    if 'audio' not in request.files or 'session_id' not in request.form:
        return jsonify({'success': False, 'error': 'Missing data'}), 400

    session_id = request.form['session_id']
    session = sessions.get(session_id)
    if not session:
        return jsonify({'success': False, 'error': 'Session expired'}), 400

    try:
        audio_file = request.files['audio']
        original_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
        wav_path = os.path.splitext(original_path)[0] + ".wav"
        audio_file.save(original_path)
        AudioSegment.from_file(original_path).export(wav_path, format="wav")

        user_text = speech_to_text(wav_path)
        os.remove(original_path)
        os.remove(wav_path)

        if not user_text.strip():
            response_text = "Sorry, I didn't catch that. Could you repeat?"
            return jsonify({
                'success': True,
                'user_text': user_text,
                'agent_response': response_text,
                'audio_response': text_to_speech_bytes(response_text),
                'status': 'Listening...'
            })

        session.history.append({'user': user_text})

        # Handle stages
        if session.stage == 'greeting':
            response_text = "Would you like to fill out a property inquiry form?"
            session.stage = 'ask_form_consent'

        elif session.stage == 'ask_form_consent':
            if is_affirmative_response(user_text):
                session.stage = 'form'
                key, q = questions[session.question_index]
                response_text = f"Great! {q}"
            elif is_negative_response(user_text):
                response_text = "Do you have questions about properties?"
                session.stage = 'qa'
            else:
                response_text = "Would you like to fill out the form? Say yes or no."

        elif session.stage == 'qa':
            if is_negative_response(user_text):
                response_text = "Would you like to fill out the inquiry form?"
                session.stage = 'ask_form_consent'
            else:
                response_text = answer_property_question_with_rag(user_text)

        elif session.stage == 'form':
            key, _ = questions[session.question_index]
            session.user_data[key] = user_text.strip()
            
            if key == 'location':
                session.user_location = user_text.strip()
                session.preference_questions = generate_preference_questions(session.user_location)
                
                if session.preference_questions:
                    session.stage = 'preferences'
                    session.preference_index = 0
                    response_text = f"Great! {session.preference_questions[0]}"
                else:
                    session.question_index += 1
                    if session.question_index < len(questions):
                        _, next_q = questions[session.question_index]
                        response_text = next_q
                    else:
                        session.stage = 'done'
                        response_text = handle_form_completion(session)
            else:
                session.question_index += 1
                if session.question_index < len(questions):
                    _, next_q = questions[session.question_index]
                    response_text = next_q
                else:
                    session.stage = 'done'
                    response_text = handle_form_completion(session)

        elif session.stage == 'preferences':
            pref_key = f"preference_{session.preference_index + 1}"
            session.user_data[pref_key] = user_text.strip()
            session.preference_index += 1
            
            if session.preference_index < len(session.preference_questions):
                response_text = session.preference_questions[session.preference_index]
            else:
                search_query = f"{session.user_location} {' '.join([session.user_data.get(f'preference_{i+1}', '') for i in range(len(session.preference_questions))])}"
                property_results = answer_property_question_with_rag(search_query)
                
                session.user_data['preferences_summary'] = "; ".join([f"{session.preference_questions[i]}: {session.user_data.get(f'preference_{i+1}', '')}" for i in range(len(session.preference_questions))])
                
                session.stage = 'form'
                session.question_index += 1
                
                if session.question_index < len(questions):
                    _, next_q = questions[session.question_index]
                    response_text = f"{property_results}\n\n{next_q}"
                else:
                    session.stage = 'done'
                    response_text = f"{property_results}\n\n{handle_form_completion(session)}"

        elif session.stage == 'done':
            if is_affirmative_response(user_text):
                response_text = "What would you like to know?"
                session.stage = 'qa'
            else:
                response_text = "Thank you! Your broker will contact you soon."

        else:
            response_text = "Would you like to ask about properties or fill out our form?"

        session.history[-1]['agent'] = response_text

        return jsonify({
            'success': True,
            'user_text': user_text,
            'agent_response': response_text,
            'audio_response': text_to_speech_bytes(response_text),
            'status': 'Listening...'
        })

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'success': True,
        'rag_initialized': rag_initialized,
        'properties_loaded': len(property_db) if not property_db.empty else 0,
        'brokers_loaded': len(broker_db) if not broker_db.empty else 0
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050, use_reloader=False)
