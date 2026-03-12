from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
import sqlite3
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import google.generativeai as genai

app = Flask(__name__)
app.secret_key = 'your_secret_key_here_change_this'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Configure Gemini API
genai.configure(api_key='AIzaSyALQl3IlQPXT_dD8k5kvBA9j3aXenmfDAg')

# Load both models
skin_disease_model = load_model('models/best_model_finetuned.keras')
skin_cancer_model = load_model('models/best_densenet201_skin_cancer.keras')

# Skin Disease classes
skin_disease_classes = ['Acne', 'Carcinoma', 'Eczema', 'Keratosis', 'Milia', 'Rosacea']

# Skin Cancer classes
skin_cancer_classes = ['ACK', 'BCC', 'MEL', 'NEV', 'SCC', 'SEK']

# Disease information for skin cancer
DISEASE_INFO = {
    'ACK': {
        'name': 'Actinic Keratosis',
        'full_name': 'ACK - Actinic Keratosis (Non-Cancerous)',
        'cancerous': False,
        'description': 'Pre-cancerous skin condition caused by sun damage. Regular monitoring recommended.'
    },
    'BCC': {
        'name': 'Basal Cell Carcinoma', 
        'full_name': 'BCC - Basal Cell Carcinoma (Cancerous)',
        'cancerous': True,
        'description': 'Most common form of skin cancer. Generally slow-growing and treatable when caught early.'
    },
    'MEL': {
        'name': 'Melanoma',
        'full_name': 'MEL - Melanoma (Cancerous)', 
        'cancerous': True,
        'description': 'Most serious form of skin cancer. Requires immediate medical attention and treatment.'
    },
    'NEV': {
        'name': 'Nevus',
        'full_name': 'NEV - Nevus (Non-Cancerous)',
        'cancerous': False,
        'description': 'Common benign skin lesion. Monitor for changes in size, color, or shape.'
    },
    'SCC': {
        'name': 'Squamous Cell Carcinoma',
        'full_name': 'SCC - Squamous Cell Carcinoma (Cancerous)',
        'cancerous': True, 
        'description': 'Second most common skin cancer. Can spread if left untreated.'
    },
    'SEK': {
        'name': 'Seborrheic Keratosis',
        'full_name': 'SEK - Seborrheic Keratosis (Non-Cancerous)',
        'cancerous': False,
        'description': 'Benign skin growth. Generally harmless but can be cosmetically concerning.'
    }
}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def init_db():
    """Initialize the SQLite database for user authentication"""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            age TEXT NOT NULL,
            purpose TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Redirect to detection page if logged in, otherwise to login"""
    if 'user_id' in session:
        return redirect(url_for('detection'))
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """Handle user registration"""
    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        age = request.form['age']
        purpose = request.form['purpose']
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('signup.html')
        
        hashed_password = generate_password_hash(password)
        
        try:
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            cursor.execute('''INSERT INTO users 
                            (first_name, last_name, email, username, password, age, purpose) 
                            VALUES (?, ?, ?, ?, ?, ?, ?)''',
                         (first_name, last_name, email, username, hashed_password, age, purpose))
            conn.commit()
            conn.close()
            flash('Account created successfully! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists', 'error')
            return render_template('signup.html')
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()
        
        if user and check_password_hash(user[5], password):
            session['user_id'] = user[0]
            session['username'] = user[4]
            session['first_name'] = user[1]
            flash('Login successful!', 'success')
            return redirect(url_for('detection'))
        else:
            flash('Invalid credentials', 'error')
            return render_template('login.html')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Handle user logout"""
    session.clear()
    flash('Logged out successfully', 'success')
    return redirect(url_for('login'))

@app.route('/detection')
def detection():
    """Render the detection page with options for skin disease or skin cancer"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('detection.html', username=session.get('first_name'))

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction for both skin disease and skin cancer"""
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    detection_type = request.form.get('detection_type', 'disease')  # 'disease' or 'cancer'
    
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Preprocess and predict based on detection type
            if detection_type == 'disease':
                # Skin Disease Detection - NO NORMALIZATION
                img = image.load_img(filepath, target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                
                # Make prediction
                predictions = skin_disease_model.predict(img_array)
                predicted_class_idx = np.argmax(predictions[0])
                predicted_class = skin_disease_classes[predicted_class_idx]
                confidence = float(np.max(predictions[0]) * 100)
                
                # Get all predictions
                all_predictions = []
                for idx, prob in enumerate(predictions[0]):
                    all_predictions.append({
                        'class': skin_disease_classes[idx],
                        'confidence': float(prob * 100)
                    })
                all_predictions = sorted(all_predictions, key=lambda x: x['confidence'], reverse=True)
                
                # Generate Gemini recommendations
                prompt = f"""You are a medical AI assistant specializing in dermatology. Analyze the following skin condition detection:

Detected Skin Condition: {predicted_class}
Confidence Level: {confidence:.2f}%

Please provide comprehensive, professional recommendations including:

1. **Condition Overview**: Brief explanation of {predicted_class}
2. **Common Symptoms**: What to look for
3. **Possible Causes**: What typically causes this condition
4. **Treatment Options**: 
   - Over-the-counter treatments
   - When to see a dermatologist
   - Prescription options (if needed)
5. **Lifestyle Recommendations**: 
   - Skincare routine adjustments
   - Products to avoid
   - Dietary considerations
6. **Prevention Tips**: How to prevent recurrence or worsening
7. **Warning Signs**: When immediate medical attention is needed

Keep the response professional, empathetic, clear, and actionable. Format with appropriate headers and bullet points."""

                model = genai.GenerativeModel("gemini-2.5-flash-lite")
                response = model.generate_content(prompt)
                
                return jsonify({
                    'success': True,
                    'detection_type': 'disease',
                    'condition_name': predicted_class,
                    'skin_condition': predicted_class,
                    'confidence': confidence,
                    'skin_confidence': confidence,
                    'all_predictions': all_predictions,
                    'recommendations': response.text,
                    'image_path': filename
                })
                
            else:  # detection_type == 'cancer'
                # Skin Cancer Detection - WITH NORMALIZATION
                img = image.load_img(filepath, target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0  # Normalize for skin cancer model
                
                # Make prediction
                predictions = skin_cancer_model.predict(img_array)
                predicted_class_idx = np.argmax(predictions[0])
                predicted_class = skin_cancer_classes[predicted_class_idx]
                confidence = float(np.max(predictions[0]) * 100)
                
                # Get disease info
                disease_info = DISEASE_INFO[predicted_class]
                is_cancerous = disease_info['cancerous']
                
                # Get all predictions
                all_predictions = []
                for idx, prob in enumerate(predictions[0]):
                    all_predictions.append({
                        'class': skin_cancer_classes[idx],
                        'confidence': float(prob * 100)
                    })
                all_predictions = sorted(all_predictions, key=lambda x: x['confidence'], reverse=True)
                
                # Generate Gemini recommendations
                cancer_status = "CANCEROUS - HIGH RISK" if is_cancerous else "NON-CANCEROUS - LOW RISK"
                prompt = f"""You are a medical AI assistant specializing in dermatology and oncology. Analyze the following skin cancer detection:

Detected Condition: {disease_info['full_name']}
Confidence Level: {confidence:.2f}%
Cancer Status: {cancer_status}

Description: {disease_info['description']}

Please provide comprehensive, professional recommendations including:

1. **Condition Overview**: Detailed explanation of {disease_info['name']}
2. **Cancer Risk Assessment**: Explain the significance of this finding
3. **Urgency Level**: How quickly should the patient seek medical attention
4. **Recommended Specialists**: Which medical professionals to consult (dermatologist, oncologist, etc.)
5. **Diagnostic Next Steps**: What additional tests or biopsies might be needed
6. **Treatment Options**: 
   - Potential treatment approaches
   - Expected outcomes
   - Timeline for treatment
7. **Prevention and Monitoring**: 
   - Sun protection strategies
   - Self-examination tips
   - Follow-up schedule
8. **Lifestyle Recommendations**: Specific advice for this condition
9. **Warning Signs**: What symptoms require immediate attention

{"**CRITICAL**: This is a potentially cancerous condition requiring immediate medical attention." if is_cancerous else "This appears to be a benign condition, but professional evaluation is still recommended."}

Keep the response professional, clear, empathetic, and actionable. Format with appropriate headers and bullet points."""

                model = genai.GenerativeModel('gemini-2.5-flash-lite')
                response = model.generate_content(prompt)
                
                return jsonify({
                    'success': True,
                    'detection_type': 'cancer',
                    'condition_name': disease_info['full_name'],
                    'confidence': confidence,
                    'is_cancerous': is_cancerous,
                    'description': disease_info['description'],
                    'all_predictions': all_predictions,
                    'recommendations': response.text,
                    'image_path': filename
                })
            
        except Exception as e:
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file format'}), 400

@app.route('/results')
def results():
    """Render the results page"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('results.html', username=session.get('first_name'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)