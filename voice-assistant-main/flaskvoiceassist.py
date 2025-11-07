# voice_assistant_flask.py
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
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# OpenAI compatibility
try:
    from openai import OpenAI
    # SECURITY WARNING: Replace with environment variable
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY', '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'))
    NEW_OPENAI = True
except ImportError:
    import openai
    # SECURITY WARNING: Replace with environment variable
    openai.api_key = os.getenv('OPENAI_API_KEY', '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')
    NEW_OPENAI = False

MODEL = 'gpt-4o-mini'

# Google Sheets Configuration
SERVICE_ACCOUNT_FILE = r"C:\industrialproject\voice-assistant-main\credentials.json" # Update this path
SHEET_URL = 'https://docs.google.com/spreadsheets/d/1mHgBAgTNYKPtN3CQS5DTxir5mKZ5jfdgu6dPH-X8nKo/edit?usp=sharing'

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'temp_audio'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load databases with better error handling
def load_databases():
    global property_db, broker_db
    
    # Load property database
    if os.path.exists(r"C:\industrialproject\voice-assistant-main\propdata.csv"):
        try:
            property_db = pd.read_csv(r"C:\industrialproject\voice-assistant-main\propdata.csv")
            print(f"✅ Loaded property database with {len(property_db)} properties")
            print(f"Property columns: {list(property_db.columns)}")
            if not property_db.empty:
                print(f"Sample locations: {property_db['Location'].unique()[:5].tolist()}")
        except Exception as e:
            print(f"❌ Error loading propdata.csv: {e}")
            property_db = pd.DataFrame()
    else:
        print("❌ propdata.csv not found!")
        property_db = pd.DataFrame()
    
    # Load broker database
    if os.path.exists(r"C:\industrialproject\voice-assistant-main\brokdata.csv"):
        try:
            broker_db = pd.read_csv(r"C:\industrialproject\voice-assistant-main\brokdata.csv")
            print(f"✅ Loaded broker database with {len(broker_db)} brokers")
            print(f"Broker columns: {list(broker_db.columns)}")
            if not broker_db.empty:
                print(f"Broker locations: {broker_db['Location'].unique()[:5].tolist()}")
                print(f"Sample broker data:\n{broker_db.head()}")
        except Exception as e:
            print(f"❌ Error loading brokdata.csv: {e}")
            broker_db = pd.DataFrame()
    else:
        print("❌ brokdata.csv not found!")
        broker_db = pd.DataFrame()

# Load databases
load_databases()

sessions = {}
questions = [
    ("name", "What is your full name?"),
    ("contact", "Please say your contact number."),
    ("rent_or_buy", "Are you looking to rent or buy a property?"),
    ("location", "Which location are you looking for the property in?"),
    ("budget", "What is your budget?"),
    ("availability_date", "When would you be available to see the finalized properties? Please provide a date.")
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

# -------------------- UTIL FUNCTIONS --------------------
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
        print("TTS Error:", e)
        return None

def speech_to_text(wav_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(wav_path) as source:
            audio = recognizer.record(source)
            return recognizer.recognize_google(audio)
    except Exception as e:
        print("Speech Recognition Error:", e)
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
        print("OpenAI API error:", e)
        return None

def generate_preference_questions(location):
    """Generate real estate preference questions using OpenAI"""
    prompt = f"""
    Generate 4-5 specific real estate preference questions for a user looking for properties in {location}. 
    Focus on practical preferences that would help narrow down property recommendations.
    Questions should be about:
    1. Property type (apartment, villa, studio, etc.)
    2. Specific amenities they prioritize
    3. Lifestyle preferences related to the location
    4. Furnishing preferences
    5. Any specific requirements like parking, floor preference, etc.
    
    Format as a simple list, one question per line. Keep questions conversational and brief.
    Return ONLY the questions, no additional text.
    Example format:
    What type of property are you looking for - apartment, villa, or studio?
    Do you prefer furnished or unfurnished properties?
    Are amenities like gym, pool, or parking important to you?
    """
    
    try:
        response = call_openai_api([{"role": "user", "content": prompt}])
        if response:
            # Parse the response into individual questions
            questions = [q.strip().lstrip('- ').lstrip('1234567890. ') for q in response.split('\n') if q.strip()]
            return [q for q in questions if q and len(q) > 10]  # Filter out empty or too short questions
        return []
    except Exception as e:
        print(f"Error generating preference questions: {e}")
        return []

def search_properties_with_preferences(location, preferences_data):
    """Search properties based on location and user preferences"""
    if property_db.empty:
        return "I don't have access to the property database right now."
    
    location_lower = location.lower()
    
    try:
        # Start with location-based filtering
        location_matches = property_db[property_db['Location'].str.contains(location_lower, case=False, na=False)]
        
        if location_matches.empty:
            return f"I couldn't find properties in {location}. Let me show you what's available in nearby areas."
        
        # Apply preference-based filtering using OpenAI to interpret preferences
        if preferences_data:
            filtered_properties = apply_preference_filtering(location_matches, preferences_data)
            return format_property_results(filtered_properties, f"{location} with your preferences")
        else:
            return format_property_results(location_matches, location)
            
    except Exception as e:
        print(f"Error in property search with preferences: {e}")
        return format_property_results(location_matches if 'location_matches' in locals() else pd.DataFrame(), location)

def apply_preference_filtering(properties_df, preferences_data):
    """Apply user preferences to filter properties"""
    filtered = properties_df.copy()
    
    try:
        # Convert preferences to a searchable string
        prefs_text = " ".join(preferences_data.values()).lower()
        
        # Simple keyword-based filtering
        if 'furnished' in prefs_text:
            furnished_filter = filtered['Furnished'].str.contains('Yes', case=False, na=False)
            if furnished_filter.any():
                filtered = filtered[furnished_filter]
        
        if 'apartment' in prefs_text:
            apt_filter = filtered['Property Type'].str.contains('apartment', case=False, na=False)
            if apt_filter.any():
                filtered = filtered[apt_filter]
        
        if 'villa' in prefs_text:
            villa_filter = filtered['Property Type'].str.contains('villa', case=False, na=False)
            if villa_filter.any():
                filtered = filtered[villa_filter]
        
        if 'studio' in prefs_text:
            studio_filter = filtered['Property Type'].str.contains('studio', case=False, na=False)
            if studio_filter.any():
                filtered = filtered[studio_filter]
        
        # If filtering resulted in no matches, return original location matches
        if filtered.empty:
            return properties_df
        
        return filtered
        
    except Exception as e:
        print(f"Error in preference filtering: {e}")
        return properties_df

def search_properties(query):
    """Search properties in the database based on user query"""
    if property_db.empty:
        return "I don't have access to the property database right now."
    
    query_lower = query.lower()
    
    try:
        # Search by location
        location_matches = property_db[property_db['Location'].str.contains(query_lower, case=False, na=False)]
        
        # Search by building name
        building_matches = property_db[property_db['Building Name'].str.contains(query_lower, case=False, na=False)]
        
        # Search by property type
        type_matches = property_db[property_db['Property Type'].str.contains(query_lower, case=False, na=False)]
        
        # Combine all matches - only include non-empty DataFrames
        dataframes_to_concat = []
        
        if not location_matches.empty:
            dataframes_to_concat.append(location_matches)
        if not building_matches.empty:
            dataframes_to_concat.append(building_matches)
        if not type_matches.empty:
            dataframes_to_concat.append(type_matches)
        
        # Only concatenate if we have dataframes to concatenate
        if dataframes_to_concat:
            all_matches = pd.concat(dataframes_to_concat, ignore_index=True).drop_duplicates()
        else:
            all_matches = pd.DataFrame()
        
    except Exception as e:
        print(f"Error in property search: {e}")
        all_matches = pd.DataFrame()
    
    # If no matches found, try partial matching on other columns
    if all_matches.empty:
        try:
            for col in ['Location', 'Building Name', 'Nearest District Name']:
                if col in property_db.columns:
                    partial_matches = property_db[property_db[col].str.contains(query_lower, case=False, na=False)]
                    if not partial_matches.empty:
                        all_matches = partial_matches
                        break
        except Exception as e:
            print(f"Error in partial matching: {e}")
    
    return format_property_results(all_matches, query)

def format_property_results(matches, query):
    """Format property search results into readable text"""
    if len(matches) == 0:
        available_locations = property_db['Location'].unique()[:5] if not property_db.empty else []
        # Fix: Check if array has elements using len() instead of boolean context
        locations_text = ", ".join(available_locations) if len(available_locations) > 0 else "various locations"
        return f"I couldn't find properties matching '{query}'. We have properties in {locations_text}. Could you be more specific about the location or property type you're looking for?"
    
    # Limit to top 3 results for voice response
    top_matches = matches.head(3)
    
    result_text = f"Based on your preferences, I found {len(matches)} matching properties"
    if len(matches) > 3:
        result_text += f". Here are the top 3 recommendations:"
    else:
        result_text += ":"
    
    for idx, (_, prop) in enumerate(top_matches.iterrows(), 1):
        prop_info = f"\n{idx}. "
        
        # Building name and location
        if pd.notna(prop.get('Building Name')):
            prop_info += f"{prop['Building Name']} in {prop['Location']}"
        else:
            prop_info += f"Property in {prop['Location']}"
        
        # Property type and rooms
        if pd.notna(prop.get('Property Type')):
            prop_info += f", {prop['Property Type']}"
        if pd.notna(prop.get('No. of Rooms')):
            prop_info += f" with {prop['No. of Rooms']} rooms"
        
        # Area
        if pd.notna(prop.get('Area (sqft)')):
            prop_info += f", {prop['Area (sqft)']} square feet"
        
        # Price info
        if pd.notna(prop.get('Monthly Rent (USD)')) and prop['Monthly Rent (USD)'] > 0:
            prop_info += f", rent ${prop['Monthly Rent (USD)']:.0f} per month"
        elif pd.notna(prop.get('Purchase Price (USD)')) and prop['Purchase Price (USD)'] > 0:
            prop_info += f", priced at ${prop['Purchase Price (USD)']:,.0f}"
        
        # Key amenities
        amenities = []
        if prop.get('Furnished') == 'Yes':
            amenities.append('furnished')
        if prop.get('Parking Spots', 0) > 0:
            amenities.append('parking')
        if prop.get('Gym') == 'Yes':
            amenities.append('gym')
        if prop.get('Pool') == 'Yes':
            amenities.append('pool')
        
        if amenities:
            prop_info += f". Amenities: {', '.join(amenities)}"
        
        result_text += prop_info
    
    if len(matches) > 3:
        result_text += f"\n\nThere are {len(matches) - 3} more properties available that match your criteria."
    
    return result_text

def is_affirmative_response(text):
    """Check if user response is affirmative (more flexible than just 'yes')"""
    affirmative_words = ['yes', 'yeah', 'yep', 'ok', 'okay', 'sure', 'alright', 'go ahead', 'proceed', 'continue', 'absolutely', 'definitely']
    text_lower = text.lower().strip()
    
    # Direct matches
    if text_lower in affirmative_words:
        return True
    
    # Partial matches for phrases
    for word in affirmative_words:
        if word in text_lower:
            return True
    
    return False

def is_negative_response(text):
    """Check if user response is negative"""
    negative_words = ['no', 'nope', 'not', 'never', 'nothing', "i'm good", "im good", 'skip', 'pass']
    text_lower = text.lower().strip()
    
    for word in negative_words:
        if word in text_lower:
            return True
    
    return False

def save_to_google_sheet(data_dict):
    """Save user data and assigned broker to Google Sheets"""
    try:
        import traceback
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(SERVICE_ACCOUNT_FILE, scope)
        client = gspread.authorize(creds)
        print("✅ Credentials loaded successfully.")
        
        # Try opening the spreadsheet
        sheet = client.open_by_url(SHEET_URL)
        print("✅ Spreadsheet opened successfully.")
        worksheet = sheet.sheet1
        print("✅ Accessed worksheet successfully.")
        
        # Prepare row data
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
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Timestamp
        ]
        
        worksheet.append_row(row)
        print("✅ Data successfully written to Google Sheet.")
        return True
        
    except Exception as e:
        print(f"❌ Error saving to Google Sheet: {e}")
        print(traceback.format_exc())
        return False

def assign_broker(user_data):
    """Assign a broker based on user's location preference with detailed debugging"""
    print(f"🔍 Starting broker assignment for user data: {user_data}")
    
    if broker_db.empty:
        print("❌ Broker database is empty!")
        return None
    
    user_location = user_data.get('location', '').lower().strip()
    print(f"🔍 User location: '{user_location}'")
    
    if not user_location:
        print("❌ No location provided by user")
        return None
    
    # Debug: Show all available broker locations
    available_locations = broker_db['Location'].unique() if 'Location' in broker_db.columns else []
    print(f"🔍 Available broker locations: {available_locations}")
    
    # Try exact location match first
    try:
        exact_matches = broker_db[broker_db['Location'].str.lower().str.contains(user_location, na=False, case=False)]
        print(f"🔍 Exact matches found: {len(exact_matches)}")
        
        if not exact_matches.empty:
            broker = exact_matches.iloc[0]
            print(f"✅ Found exact match broker: {broker.get('Agent Name', 'Unknown')}")
            return create_broker_dict(broker)
    except Exception as e:
        print(f"❌ Error in exact matching: {e}")
    
    # If no exact match, try partial matching
    try:
        for idx, broker in broker_db.iterrows():
            broker_location = str(broker.get('Location', '')).lower()
            print(f"🔍 Checking broker location: '{broker_location}' against user: '{user_location}'")
            
            if user_location in broker_location or broker_location in user_location:
                print(f"✅ Found partial match broker: {broker.get('Agent Name', 'Unknown')}")
                return create_broker_dict(broker)
    except Exception as e:
        print(f"❌ Error in partial matching: {e}")
    
    # If still no match, assign the first available broker
    try:
        if len(broker_db) > 0:
            broker = broker_db.iloc[0]
            print(f"⚠️ No location match found, assigning default broker: {broker.get('Agent Name', 'Unknown')}")
            return create_broker_dict(broker)
    except Exception as e:
        print(f"❌ Error assigning default broker: {e}")
    
    print("❌ No brokers available in database")
    return None

def create_broker_dict(broker):
    """Create standardized broker dictionary from broker row"""
    return {
        'Assigned_Broker_Name': str(broker.get('Agent Name', '')),
        'Assigned_Broker_Phone': str(broker.get('Phone Number', '')),
        'Assigned_Broker_Email': str(broker.get('Email', '')),
        'Assigned_Broker_Location': str(broker.get('Location', ''))
    }

def handle_form_completion(session):
    """Handle completion of the form and broker assignment"""
    # Assign broker based on user data
    assigned_broker = assign_broker(session.user_data)
    
    if assigned_broker:
        # Add broker info to user data
        session.user_data.update(assigned_broker)
        
        # Prepare response with broker info
        broker_name = assigned_broker.get('Assigned_Broker_Name', 'our team')
        broker_phone = assigned_broker.get('Assigned_Broker_Phone', '')
        
        response_text = f"Perfect! I've saved all your details and assigned you to {broker_name}"
        if broker_phone:
            response_text += f" (Phone: {broker_phone})"
        response_text += ". They specialize in your preferred location and will contact you within 24 hours with personalized property recommendations and arrange property viewings based on your availability date."
    else:
        response_text = "Thank you! I've saved your details. Our property team will contact you within 24 hours with suitable recommendations and will arrange property viewings based on your availability date."
    
    # Save to Google Sheet
    print(f"User data collected: {session.user_data}")
    save_success = save_to_google_sheet(session.user_data)
    if save_success:
        print("✅ Data saved to Google Sheet successfully")
    else:
        print("❌ Failed to save data to Google Sheet")
    
    # Ask if they want to know more about properties - this is moved to the end
    response_text += "\n\nDo you have any questions about our available properties? I can tell you about specific locations, property types, or amenities."
    
    return response_text

# -------------------- API ROUTES --------------------
@app.route('/api/voice-init', methods=['GET'])
def voice_init():
    session_id = f"session_{int(time.time())}"
    sessions[session_id] = Session()
    intro = [
        "Hi! I'm Agent Shreyash, your real estate assistant.",
        
        "How are you today?"
    ]
    combined = " ".join(intro)
    return jsonify({
        'success': True,
        'session_id': session_id,
        'agent_response': combined,
        'audio_response': text_to_speech_bytes(combined),
        'status': 'Listening...'
    })

@app.route('/api/voice-chat', methods=['POST'])
def voice_chat():
    if 'audio' not in request.files or 'session_id' not in request.form:
        return jsonify({'success': False, 'error': 'Missing audio or session ID'}), 400

    session_id = request.form['session_id']
    session = sessions.get(session_id)
    if not session:
        return jsonify({'success': False, 'error': 'Session expired or invalid'}), 400

    try:
        audio_file = request.files['audio']
        original_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
        wav_path = os.path.splitext(original_path)[0] + ".wav"
        audio_file.save(original_path)
        AudioSegment.from_file(original_path).export(wav_path, format="wav")

        user_text = speech_to_text(wav_path)
        os.remove(original_path)
        os.remove(wav_path)

        print(f"User said: '{user_text}'")

        if not user_text.strip():
            session.retry_count += 1
            if session.retry_count >= 2:
                response_text = "Sorry, I didn't catch that. Could you please repeat?"
            else:
                response_text = "Could you say that again?"
            return jsonify({
                'success': True,
                'user_text': user_text,
                'agent_response': response_text,
                'audio_response': text_to_speech_bytes(response_text),
                'status': 'Listening...'
            })

        session.retry_count = 0
        session.history.append({'user': user_text})

        if session.stage == 'greeting':
            response_text = "Would you like to proceed with filling out a property inquiry form so I can match you with the perfect property?"
            session.stage = 'ask_form_consent'

        elif session.stage == 'ask_form_consent':
            if is_affirmative_response(user_text):
                session.stage = 'form'
                key, q = questions[session.question_index]
                response_text = f"Great! Let's begin the form. {q}"
            elif is_negative_response(user_text):
                response_text = "Do you have any questions about our available properties? I can tell you about specific locations, property types, or amenities."
                session.stage = 'qa'
            else:
                response_text = "I didn't quite catch that. Would you like to fill out a property inquiry form? Just say yes or no."

        elif session.stage == 'qa':
            if is_negative_response(user_text):
                response_text = "Would you like to proceed with filling out a property inquiry form so I can match you with the perfect property?"
                session.stage = 'ask_form_consent'
            else:
                # First try to search our property database
                property_results = search_properties(user_text)
                
                if "couldn't find properties" in property_results or property_db.empty:
                    # Fallback to OpenAI for general real estate questions
                    prompt = f"""
A user asked: "{user_text}"
This is a real estate inquiry. Give a helpful response as Agent Shreyash, but keep it concise (under 100 words).
Focus on being helpful and encouraging them to be more specific about their property needs.
"""
                    ai_reply = call_openai_api([{"role": "user", "content": prompt}])
                    response_text = ai_reply if ai_reply else "Could you please be more specific about what type of property you're looking for?"
                else:
                    response_text = property_results

        elif session.stage == 'form':
            key, _ = questions[session.question_index]
            session.user_data[key] = user_text.strip()
            
            # Special handling for location question - generate preferences and show properties
            if key == 'location':
                session.user_location = user_text.strip()
                # Generate preference questions based on location
                session.preference_questions = generate_preference_questions(session.user_location)
                if session.preference_questions:
                    session.stage = 'preferences'
                    session.preference_index = 0
                    response_text = f"Great! Now I'd like to understand your preferences better to show you the most suitable properties in {session.user_location}. {session.preference_questions[0]}"
                else:
                    # If no preference questions generated, continue with next form question
                    session.question_index += 1
                    if session.question_index < len(questions):
                        _, next_q = questions[session.question_index]
                        response_text = next_q
                    else:
                        session.stage = 'done'
                        response_text = handle_form_completion(session)
            else:
                # Continue with next question
                session.question_index += 1
                if session.question_index < len(questions):
                    _, next_q = questions[session.question_index]
                    response_text = next_q
                else:
                    session.stage = 'done'
                    response_text = handle_form_completion(session)

        elif session.stage == 'preferences':
            # Store preference answer
            pref_key = f"preference_{session.preference_index + 1}"
            session.user_data[pref_key] = user_text.strip()
            session.preference_index += 1
            
            if session.preference_index < len(session.preference_questions):
                # Ask next preference question
                response_text = session.preference_questions[session.preference_index]
            else:
                # Preferences complete, show matching properties
                preferences_data = {k: v for k, v in session.user_data.items() if k.startswith('preference_')}
                property_results = search_properties_with_preferences(session.user_location, preferences_data)
                
                # Create preferences summary for storage
                session.user_data['preferences_summary'] = "; ".join([f"Q{i+1}: {session.preference_questions[i]} A: {session.user_data.get(f'preference_{i+1}', '')}" for i in range(len(session.preference_questions))])
                
                # Continue with remaining form questions (budget and availability)
                session.stage = 'form'
                session.question_index += 1  # Move to budget question
                session.properties_shown = True
                
                if session.question_index < len(questions):
                    _, next_q = questions[session.question_index]
                    response_text = f"{property_results}\n\nNow, let's continue with the form. {next_q}"
                else:
                    session.stage = 'done'
                    response_text = f"{property_results}\n\n{handle_form_completion(session)}"

        elif session.stage == 'done':
            if is_affirmative_response(user_text) and any(word in user_text.lower() for word in ['property', 'properties', 'help', 'question']):
                response_text = "Of course! What would you like to know about our available properties?"
                session.stage = 'qa'
            elif is_negative_response(user_text):
                response_text = "Thank you for your interest! Your assigned broker will contact you soon with detailed information about the properties that match your preferences and will coordinate the property viewings based on your availability."
            else:
                response_text = "I'm here to help! If you have any questions about properties, locations, amenities, or anything else, just let me know. Otherwise, your broker will be in touch soon!"

        else:
            response_text = "I'm not sure how to help with that. Would you like to ask about properties or fill out our inquiry form?"

        session.history[-1]['agent'] = response_text

        return jsonify({
            'success': True,
            'user_text': user_text,
            'agent_response': response_text,
            'audio_response': text_to_speech_bytes(response_text),
            'status': 'Listening...'
            })

    except Exception as e:
        print(f"Error in voice_chat: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)      