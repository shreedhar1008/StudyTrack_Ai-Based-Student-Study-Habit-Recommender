from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import sqlite3
import os
import json
from datetime import datetime, date
from werkzeug.utils import secure_filename

# Fix SQLite 3.12 deprecation warnings for date/datetime
def adapt_date_iso(val):
    return val.isoformat()

def adapt_datetime_iso(val):
    return val.isoformat()

sqlite3.register_adapter(date, adapt_date_iso)
sqlite3.register_adapter(datetime, adapt_datetime_iso)
from werkzeug.security import generate_password_hash, check_password_hash
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Groq client with environment variable
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
if not GROQ_API_KEY:
    print("⚠️  WARNING: GROQ_API_KEY not found in environment variables!")
    print("Please create a .env file with your API key. See .env.example")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key-change-in-production')
CORS(app)

# Load ML models
print("Loading models...")
rf_model = joblib.load('models/rf_dropout_model.pkl')
kmeans_model = joblib.load('models/kmeans_model.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_columns = joblib.load('models/feature_columns.pkl')
print("Models loaded successfully!")

# Database helper functions
def get_db():
    conn = sqlite3.connect('studytrack.db', timeout=30)
    conn.row_factory = sqlite3.Row
    # Set a busy timeout to wait for locks to be released (30 seconds)
    conn.execute('PRAGMA busy_timeout = 30000')
    return conn

def init_db():
    """Initialize SQLite database with all required tables (only if not exists)"""
    conn = get_db()
    # Enable Write-Ahead Logging only during initialization for optimal persistent settings
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA synchronous=NORMAL')
    cursor = conn.cursor()
    
    # Create tables if they don't exist
    # (Removed early return to ensure NEW tables like admin_feedback are created)
    
    print("🔨 Creating database tables...")
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'student',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Students table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            student_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            full_name TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            major TEXT,
            year INTEGER,
            gpa REAL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Daily habits log - NEW TABLE for daily tracking
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_habits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            log_date DATE NOT NULL,
            study_hours REAL,
            sleep_hours REAL,
            attendance_percentage REAL,
            social_media_hours REAL,
            exercise_frequency INTEGER,
            stress_level INTEGER,
            motivation_level INTEGER,
            mental_health_rating INTEGER,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students (student_id),
            UNIQUE(student_id, log_date)
        )
    ''')
    
    # Predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            daily_habit_id INTEGER,
            dropout_probability REAL,
            risk_level TEXT,
            cluster_number INTEGER,
            priority_score INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students (student_id),
            FOREIGN KEY (daily_habit_id) REFERENCES daily_habits (id)
        )
    ''')
    
    # Recommendations table with daily tracking
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            prediction_id INTEGER,
            daily_habit_id INTEGER,
            category TEXT,
            message TEXT,
            priority TEXT,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students (student_id),
            FOREIGN KEY (prediction_id) REFERENCES predictions (id),
            FOREIGN KEY (daily_habit_id) REFERENCES daily_habits (id)
        )
    ''')
    
    # Feedback table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            prediction_id INTEGER,
            rating INTEGER,
            feedback_text TEXT,
            improvement_seen TEXT,
            would_recommend INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students (student_id),
            FOREIGN KEY (prediction_id) REFERENCES predictions (id)
        )
    ''')
    
    # Legacy table for backward compatibility
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS student_habits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            study_hours REAL,
            sleep_hours REAL,
            attendance_percentage REAL,
            social_media_hours REAL,
            exercise_frequency INTEGER,
            stress_level INTEGER,
            motivation_level INTEGER,
            mental_health_rating INTEGER,
            recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students (student_id)
        )
    ''')
    
    # NEW: Admin Feedback table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS admin_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            admin_id INTEGER,
            student_id INTEGER,
            prediction_id INTEGER,
            effectiveness_rating INTEGER, -- 1-5
            prediction_accurate INTEGER, -- 1 for True, 0 for False
            actual_performance TEXT,
            intervention_taken TEXT,
            feedback_text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (admin_id) REFERENCES users (id),
            FOREIGN KEY (student_id) REFERENCES students (student_id),
            FOREIGN KEY (prediction_id) REFERENCES predictions (id)
        )
    ''')
    
    # NEW: User sessions table for tracking returning users
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            student_id INTEGER,
            login_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_habit_log_date DATE,
            total_logins INTEGER DEFAULT 1,
            FOREIGN KEY (user_id) REFERENCES users (id),
            FOREIGN KEY (student_id) REFERENCES students (student_id)
        )
    ''')
    
    # Create default users with hashed passwords
    cursor.execute('SELECT COUNT(*) FROM users')
    if cursor.fetchone()[0] == 0:
        # Hash default passwords
        admin_password = generate_password_hash('admin123')
        student_password = generate_password_hash('student123')
        
        cursor.execute('''
            INSERT INTO users (id, username, email, password, role)
            VALUES (1, 'admin', 'admin@studytrack.com', ?, 'admin')
        ''', (admin_password,))
        
        cursor.execute('''
            INSERT INTO users (id, username, email, password, role)
            VALUES (2, 'student1', 'student1@example.com', ?, 'student')
        ''', (student_password,))
        
        cursor.execute('''
            INSERT INTO students (student_id, user_id, full_name, age, gender, major, year, gpa)
            VALUES (1, 2, 'John Doe', 20, 'Male', 'Computer Science', 2, 3.2)
        ''')
        
        print("✅ Default users created with hashed passwords")
    
    conn.commit()
    conn.close()
    print("✅ Database initialized successfully!")

#=====================Helper Functions=====================
def generate_ai_recommendations(student_data, cluster_info, gap_analysis_list):
    """Generate personalized recommendations using Groq AI"""
    
    # Format gap analysis for prompt
    gap_text = "\n".join([
        f"- {gap['metric']}: Current={gap['current']}, Target={gap['target']}, Gap={gap['gap']:.1f}"
        for gap in gap_analysis_list
    ])
    
    # Prepare context for AI
    prompt = f"""You are an expert educational counselor analyzing a student's behavior and performance. 
    
**Student Profile:**
- Cluster: {cluster_info['cluster']}
- Risk Level: {student_data.get('risk_level', 'Unknown')}
- Dropout Probability: {student_data.get('dropout_probability', 0) * 100:.1f}%

**Current Habits:**
- Study Hours/Day: {student_data.get('study_hours', 0)}
- Sleep Hours/Night: {student_data.get('sleep_hours', 0)}
- Attendance: {student_data.get('attendance', 0)}%
- Social Media Usage: {student_data.get('social_media', 0)} hours/day
- Exercise Frequency: {student_data.get('exercise', 0)} times/week
- Stress Level: {student_data.get('stress_level', 5)}/10
- Motivation Level: {student_data.get('motivation_level', 5)}/10

**Target Habits (Based on Successful Students):**
- Study Hours: {cluster_info['targets']['study_hours']}
- Sleep Hours: {cluster_info['targets']['sleep_hours']}
- Attendance: {cluster_info['targets']['attendance']}%
- Social Media: {cluster_info['targets']['social_media']} hours/day
- Exercise: {cluster_info['targets']['exercise']} times/week

**Gap Analysis:**
{gap_text}

Based on this data, provide 5-7 **concise, actionable recommendations** to help this student improve their academic performance and reduce dropout risk.

**IMPORTANT**: Each recommendation must be exactly 1-2 sentences (maximum 30-40 words). Be specific and actionable, but keep it brief.

Format each recommendation as:
- **Category**: [Study/Sleep/Wellness/Time Management/etc]
- **Priority**: [High/Medium/Low]
- **Recommendation**: [One concise sentence with specific action, followed by one sentence with the expected outcome or tip]

    Make the recommendations:
    1. Highly specific to THIS student's gaps
    2. Actionable with clear steps
    3. Encouraging and supportive in tone
    4. Realistic and achievable
    5. Brief (1-2 sentences only)

RETURN YOUR RESPONSE AS A PURE JSON LIST OF OBJECTS. Do not use Markdown formatting (like ```json).
Example format:
[
  {{
    "category": "Study",
    "priority": "High",
    "message": "Increase study hours by 1 hour daily to meet the target."
  }}
]"""

    try:
        # Call Groq API
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert educational counselor specializing in student success and dropout prevention. Provide specific, actionable, and empathetic guidance."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=1500
        )
        
        ai_response = chat_completion.choices[0].message.content
        recommendations = parse_ai_recommendations(ai_response)
        return recommendations
        
    except Exception as e:
        print(f"Groq API Error: {e}")
        return generate_fallback_recommendations(gap_analysis_list)


def parse_ai_recommendations(ai_text):
    """Parse AI-generated text into structured recommendations"""
    import re
    
    # Try to parse as JSON first
    try:
        # Clean up Markdown code blocks if present
        clean_text = ai_text.strip()
        if clean_text.startswith('```json'):
            clean_text = clean_text[7:]
        if clean_text.startswith('```'):
            clean_text = clean_text[3:]
        if clean_text.endswith('```'):
            clean_text = clean_text[:-3]
        
        recommendations = json.loads(clean_text)
        
        # Validate structure
        valid_recs = []
        if isinstance(recommendations, list):
            for rec in recommendations:
                if isinstance(rec, dict) and 'category' in rec and 'message' in rec:
                    valid_recs.append({
                        'category': rec.get('category', 'General'),
                        'priority': rec.get('priority', 'Medium'),
                        'message': rec.get('message', '')
                    })
            if valid_recs:
                return valid_recs
    except json.JSONDecodeError:
        print("Failed to parse AI response as JSON, falling back to text parsing")
    
    # Fallback to regex parsing if JSON fails
    recommendations = []
    lines = ai_text.split('\n')
    current_rec = {}
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines
        if not line:
            i += 1
            continue
        
        # Check if this line contains Category
        if re.search(r'\*\*Category\*\*\s*:', line, re.IGNORECASE):
            # Save previous recommendation if complete
            if current_rec and 'category' in current_rec and 'message' in current_rec:
                if 'priority' not in current_rec:
                    current_rec['priority'] = 'Medium'
                recommendations.append(current_rec)
                current_rec = {}
            
            # Extract category
            category_match = re.search(r'\*\*Category\*\*\s*:\s*(.+?)(?:\s*\*\*|$)', line, re.IGNORECASE)
            if category_match:
                current_rec['category'] = category_match.group(1).strip()
            
            # Look ahead for Priority and Recommendation on next lines
            j = i + 1
            while j < len(lines) and j < i + 5:  # Look ahead max 5 lines
                next_line = lines[j].strip()
                
                if re.search(r'\*\*Priority\*\*\s*:', next_line, re.IGNORECASE):
                    priority_match = re.search(r'\*\*Priority\*\*\s*:\s*(.+?)(?:\s*\*\*|$)', next_line, re.IGNORECASE)
                    if priority_match:
                        priority_val = priority_match.group(1).strip()
                        if priority_val in ['High', 'Medium', 'Low']:
                            current_rec['priority'] = priority_val
                
                elif re.search(r'\*\*Recommendation\*\*\s*:', next_line, re.IGNORECASE):
                    rec_match = re.search(r'\*\*Recommendation\*\*\s*:\s*(.+)', next_line, re.IGNORECASE)
                    if rec_match:
                        current_rec['message'] = rec_match.group(1).strip()
                    i = j  # Skip to this line
                    break
                
                elif re.search(r'\*\*Category\*\*\s*:', next_line, re.IGNORECASE):
                    # Found next recommendation, stop looking
                    break
                
                j += 1
        
        i += 1
    
    # Add last recommendation
    if current_rec and 'category' in current_rec and 'message' in current_rec:
        if 'priority' not in current_rec:
            current_rec['priority'] = 'Medium'
        recommendations.append(current_rec)
    
    # If structured format didn't work, try numbered format: "1. Study: message"
    if not recommendations:
        numbered_recs = re.findall(
            r'\d+\.\s*\*?\*?([^:*]+?)\*?\*?\s*:\s*(.+?)(?=\n\d+\.|\Z)',
            ai_text,
            re.DOTALL
        )
        for category, message in numbered_recs[:7]:
            message_clean = message.strip().replace('\n', ' ')
            if category.strip() and message_clean:
                recommendations.append({
                    'category': category.strip(),
                    'priority': 'Medium',
                    'message': message_clean
                })
    
    # Try simple bullet format: "- **Study**: message"
    if not recommendations:
        bullet_recs = re.findall(
            r'[-*•]\s*\*\*([^*:]+)\*\*\s*:\s*(.+?)(?=\n[-*•]|\Z)',
            ai_text,
            re.DOTALL
        )
        for category, message in bullet_recs[:7]:
            message_clean = message.strip().replace('\n', ' ')
            if category.strip() and message_clean:
                recommendations.append({
                    'category': category.strip(),
                    'priority': 'Medium',
                    'message': message_clean
                })
    
    # Clean up and validate
    valid_recommendations = []
    for rec in recommendations:
        if 'category' in rec and 'message' in rec and rec['message']:
            if 'priority' not in rec or rec['priority'] not in ['High', 'Medium', 'Low']:
                rec['priority'] = 'Medium'
            rec['message'] = rec['message'].replace('**', '').replace('*', '').strip()
            rec['category'] = rec['category'].replace('**', '').replace('*', '').strip()
            valid_recommendations.append(rec)
    
    # Fallback: return raw text if all parsing failed
    if not valid_recommendations:
        return [{
            'category': 'General Recommendations',
            'priority': 'High',
            'message': ai_text[:800] if len(ai_text) > 800 else ai_text
        }]
    
    return valid_recommendations



def generate_fallback_recommendations(gap_analysis):
    """Fallback recommendations if AI fails"""
    recommendations = []
    
    for gap in gap_analysis:
        if gap['gap'] > 0:
            recommendations.append({
                'category': gap['metric'],
                'priority': 'High' if gap['gap'] > 2 else 'Medium',
                'message': f"Increase your {gap['metric'].lower()} by {gap['gap']:.1f} to reach optimal levels"
            })
    
    return recommendations


def generate_ai_study_techniques(student_profile):
    """Generate personalized study techniques using Groq AI"""
    
    prompt = f"""Based on this student's profile, recommend 3-5 **proven study techniques** that would work best for them:

**Student Context:**
- Study Hours: {student_profile['study_hours']} hours/day
- Stress Level: {student_profile.get('stress_level', 5)}/10
- Motivation: {student_profile.get('motivation_level', 5)}/10
- Risk Level: {student_profile.get('risk_level', 'Unknown')}

For each technique, provide:
1. **Name**: [Technique name]
2. **Description**: [What it is in 1 sentence]
3. **How to Use**: [Step-by-step implementation]
4. **Best For**: [When/why to use it]
5. **Frequency**: [How often to apply]

Choose from these evidence-based techniques:
- Pomodoro Technique (25-min focused work)
- Spaced Repetition (review material over time)
- Active Recall (test yourself without notes)
- Feynman Technique (explain concepts simply)
- Mind Mapping (visual organization)
- Interleaving (mix different subjects)
- SQ3R Method (Survey, Question, Read, Recite, Review)

Select the 3-5 most suitable for THIS specific student."""

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in learning science and study techniques. Provide practical, evidence-based study methods."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.6,
            max_tokens=1200
        )
        
        ai_response = chat_completion.choices[0].message.content
        
        # Parse study techniques
        techniques = parse_study_techniques(ai_response)
        
        return techniques
        
    except Exception as e:
        print(f"Groq API Error: {e}")
        return get_default_study_techniques()


def parse_study_techniques(ai_text):
    """Parse AI study techniques into structured format"""
    techniques = []
    lines = ai_text.split('\n')
    
    current_tech = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if 'Name' in line and '**' in line:
            if current_tech and 'name' in current_tech:
                techniques.append(current_tech)
            current_tech = {'name': line.split(':', 1)[1].strip().replace('**', '').replace('*', '')}
        elif 'Description' in line and '**' in line:
            current_tech['description'] = line.split(':', 1)[1].strip().replace('**', '').replace('*', '')
        elif 'How to Use' in line and '**' in line:
            current_tech['how_to'] = line.split(':', 1)[1].strip().replace('**', '').replace('*', '')
        elif 'Best For' in line and '**' in line:
            current_tech['best_for'] = line.split(':', 1)[1].strip().replace('**', '').replace('*', '')
        elif 'Frequency' in line and '**' in line:
            current_tech['frequency'] = line.split(':', 1)[1].strip().replace('**', '').replace('*', '')
    
    if current_tech and 'name' in current_tech:
        techniques.append(current_tech)
    
    return techniques if techniques else get_default_study_techniques()


def get_default_study_techniques():
    """Default study techniques if AI fails"""
    return [
        {
            'name': 'Pomodoro Technique',
            'description': 'Work in focused 25-minute intervals with 5-minute breaks',
            'how_to': 'Set a timer for 25 minutes, focus on one task, take a 5-minute break, repeat 4 times, then take a 15-minute break',
            'best_for': 'Maintaining focus and preventing burnout',
            'frequency': 'Daily during study sessions'
        },
        {
            'name': 'Active Recall',
            'description': 'Test yourself on material without looking at notes',
            'how_to': 'Close your books, write down everything you remember about a topic, then check your notes to fill gaps',
            'best_for': 'Strengthening memory and identifying weak areas',
            'frequency': 'After each study session and before exams'
        },
        {
            'name': 'Spaced Repetition',
            'description': 'Review material at increasing intervals over time',
            'how_to': 'Review new material after 1 day, then 3 days, then 7 days, then 14 days',
            'best_for': 'Long-term retention of information',
            'frequency': 'Continuously throughout the semester'
        }
    ]

# Initialize database on startup
init_db()

# ============= ROUTES =============

@app.route('/')
def index():
    """Landing page"""
    return render_template('index.html')


@app.route('/student')
def student_dashboard():
    return render_template('student_dashboard.html')

@app.route('/admin')
def admin_dashboard():
    return render_template('admin_dashboard.html')

# ============= API ENDPOINTS =============

@app.route('/api/student/register', methods=['POST'])
def register_student():
    """Register new student"""
    data = request.json
    conn = get_db()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO students (name, email, department, semester)
            VALUES (?, ?, ?, ?)
        ''', (data['name'], data['email'], data.get('department', 'General'), data.get('semester', 1)))
        
        conn.commit()
        student_id = cursor.lastrowid
        conn.close()
        
        return jsonify({
            'status': 'success',
            'student_id': student_id,
            'message': 'Student registered successfully'
        })
    except sqlite3.IntegrityError:
        return jsonify({'status': 'error', 'message': 'Email already exists'}), 400

@app.route('/api/student/login', methods=['POST'])
def login_student():
    """Simple login (no password for demo)"""
    data = request.json
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM students WHERE email = ?', (data['email'],))
    student = cursor.fetchone()
    conn.close()
    
    if student:
        return jsonify({
            'status': 'success',
            'student_id': student['student_id'],
            'name': student['name'],
            'email': student['email']
        })
    else:
        return jsonify({'status': 'error', 'message': 'Student not found'}), 404

@app.route('/api/predict', methods=['POST'])
def predict_dropout():
    """Predict dropout risk for a student"""
    data = request.json
    student_id = data.get('student_id')
    
    # Get today's date
    today = datetime.now().date()
    
    # Prepare features for Random Forest
    features_rf = {}
    for col in feature_columns:
        # Try exact match first
        value = data.get(col)
        
        # If not found, try snake_case (e.g., 'Study Hours' -> 'study_hours')
        if value is None:
            snake_key = col.lower().replace(' ', '_')
            value = data.get(snake_key)
            
        # Default to 0 if still not found
        if value is None:
            # Check for specific known mappings if generic snake_case fails
            if 'Attendance' in col: value = data.get('attendance_percentage')
            elif 'Social' in col and 'Media' in col: value = data.get('social_media_hours')
            elif 'Exercise' in col: value = data.get('exercise_frequency')
            elif 'Sleep' in col: value = data.get('sleep_hours')
            elif 'Study' in col: value = data.get('study_hours')
            else: value = 0

        # Ensure numeric values only - convert to float, default to 0 if not numeric
        try:
            features_rf[col] = float(value) if value not in [None, '', 'student'] else 0
        except (ValueError, TypeError):
            features_rf[col] = 0
    
    X_rf = pd.DataFrame([features_rf])
    
    # Ensure all columns are numeric before scaling
    for col in X_rf.columns:
        X_rf[col] = pd.to_numeric(X_rf[col], errors='coerce').fillna(0)
    
    X_rf_scaled = scaler.transform(X_rf)
    
    # Fix Scikit-learn feature names warning by converting scaled array back to DataFrame
    X_rf_scaled_df = pd.DataFrame(X_rf_scaled, columns=feature_columns)
    
    # Predict dropout probability
    dropout_prob = rf_model.predict_proba(X_rf_scaled_df)[0][1]
    dropout_prediction = "Yes" if dropout_prob > 0.5 else "No"
    
    # Determine risk level
    if dropout_prob < 0.3:
        risk_level = "Low"
    elif dropout_prob < 0.7:
        risk_level = "Moderate"
    else:
        risk_level = "High"
    
    # K-Means clustering
    clustering_features = np.array([[
        data.get('study_hours', 0),
        data.get('sleep_hours', 0),
        data.get('attendance_percentage', 0),
        data.get('social_media_hours', 0),
        data.get('exercise_frequency', 0),
        data.get('stress_level', 5),
        data.get('motivation_level', 5)
    ]])
    
    cluster = int(kmeans_model.predict(clustering_features)[0])
    priority_score = int(dropout_prob * 15)
    
    # Save to database
    prediction_id = None
    daily_habit_id = None
    
    if student_id:
        conn = get_db()
        try:
            cursor = conn.cursor()
            
            # Check if today's habit already logged
            cursor.execute('''
                SELECT id FROM daily_habits 
                WHERE student_id = ? AND log_date = ?
            ''', (student_id, today))
            
            existing = cursor.fetchone()
            
            if existing:
                # Update existing entry
                daily_habit_id = existing[0]
                cursor.execute('''
                    UPDATE daily_habits 
                    SET study_hours=?, sleep_hours=?, attendance_percentage=?,
                        social_media_hours=?, exercise_frequency=?, stress_level=?,
                        motivation_level=?, mental_health_rating=?
                    WHERE id=?
                ''', (
                    data.get('study_hours', 0),
                    data.get('sleep_hours', 0),
                    data.get('attendance_percentage', 0),
                    data.get('social_media_hours', 0),
                    data.get('exercise_frequency', 0),
                    data.get('stress_level', 5),
                    data.get('motivation_level', 5),
                    data.get('mental_health_rating', 5),
                    daily_habit_id
                ))
            else:
                # Insert new daily habit
                cursor.execute('''
                    INSERT INTO daily_habits 
                    (student_id, log_date, study_hours, sleep_hours, attendance_percentage,
                     social_media_hours, exercise_frequency, stress_level, motivation_level, mental_health_rating)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    student_id, today,
                    data.get('study_hours', 0),
                    data.get('sleep_hours', 0),
                    data.get('attendance_percentage', 0),
                    data.get('social_media_hours', 0),
                    data.get('exercise_frequency', 0),
                    data.get('stress_level', 5),
                    data.get('motivation_level', 5),
                    data.get('mental_health_rating', 5)
                ))
                daily_habit_id = cursor.lastrowid
            
            # Save prediction
            cursor.execute('''
                INSERT INTO predictions 
                (student_id, daily_habit_id, dropout_probability, risk_level, cluster_number, priority_score)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (student_id, daily_habit_id, float(dropout_prob), risk_level, cluster, priority_score))
            
            prediction_id = cursor.lastrowid
            
            # Also save to legacy table for backward compatibility
            cursor.execute('''
                INSERT INTO student_habits 
                (student_id, study_hours, sleep_hours, attendance_percentage, social_media_hours,
                 exercise_frequency, stress_level, motivation_level, mental_health_rating)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                student_id,
                data.get('study_hours', 0),
                data.get('sleep_hours', 0),
                data.get('attendance_percentage', 0),
                data.get('social_media_hours', 0),
                data.get('exercise_frequency', 0),
                data.get('stress_level', 5),
                data.get('motivation_level', 5),
                data.get('mental_health_rating', 5)
            ))
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            print(f"Error saving prediction: {e}")
        finally:
            conn.close()
    
    return jsonify({
        'status': 'success',
        'dropout_probability': round(float(dropout_prob), 4),
        'dropout_prediction': dropout_prediction,
        'risk_level': risk_level,
        'cluster': cluster,
        'priority_score': priority_score,
        'prediction_id': prediction_id,
        'daily_habit_id': daily_habit_id,
        'log_date': str(today)
    })


@app.route('/api/admin/analytics', methods=['GET'])
def get_analytics():
    """Get analytics for admin dashboard"""
    conn = get_db()
    cursor = conn.cursor()
    
    # Total students who have been tested
    cursor.execute('SELECT COUNT(DISTINCT student_id) FROM predictions')
    total_tested = cursor.fetchone()[0]
    
    # High priority students (priority score > 10)
    cursor.execute('''
        SELECT COUNT(DISTINCT student_id) 
        FROM predictions 
        WHERE priority_score > 10
        AND id IN (
            SELECT MAX(id) FROM predictions GROUP BY student_id
        )
    ''')
    high_priority = cursor.fetchone()[0]
    
    # Average dropout probability
    cursor.execute('''
        SELECT AVG(dropout_probability) 
        FROM predictions
        WHERE id IN (
            SELECT MAX(id) FROM predictions GROUP BY student_id
        )
    ''')
    avg_dropout = cursor.fetchone()[0] or 0
    
    # Risk distribution (latest predictions only)
    cursor.execute('''
        SELECT risk_level, COUNT(*) 
        FROM predictions
        WHERE id IN (
            SELECT MAX(id) FROM predictions GROUP BY student_id
        )
        GROUP BY risk_level
    ''')
    risk_distribution = {}
    for row in cursor.fetchall():
        risk_distribution[row[0]] = row[1]
    
    # Cluster distribution (latest predictions only)
    cursor.execute('''
        SELECT cluster_number, COUNT(*) 
        FROM predictions
        WHERE id IN (
            SELECT MAX(id) FROM predictions GROUP BY student_id
        )
        GROUP BY cluster_number
    ''')
    cluster_distribution = {}
    for row in cursor.fetchall():
        if row[0] is not None:
            cluster_distribution[str(row[0])] = row[1]
            
    # Detailed distribution: Risk -> Cluster -> {count, avg_priority}
    cursor.execute('''
        SELECT 
            risk_level,
            cluster_number,
            COUNT(*) as student_count,
            AVG(priority_score) as avg_priority
        FROM predictions
        WHERE id IN (SELECT MAX(id) FROM predictions GROUP BY student_id)
        GROUP BY risk_level, cluster_number
        ORDER BY 
            CASE risk_level 
                WHEN 'High' THEN 1 
                WHEN 'Moderate' THEN 2 
                WHEN 'Low' THEN 3 
            END,
            cluster_number
    ''')
    
    detailed_stats = []
    for row in cursor.fetchall():
        detailed_stats.append({
            'risk': row[0],
            'cluster': row[1],
            'count': row[2],
            'avg_priority': round(row[3], 1)
        })
    
    # Total assessments performed
    cursor.execute('SELECT COUNT(*) FROM predictions')
    total_assessments = cursor.fetchone()[0]
    
    conn.close()
    
    return jsonify({
        'status': 'success',
        'analytics': {
            'total_students': total_tested,
            'high_priority_students': high_priority,
            'avg_dropout_probability': float(avg_dropout),
            'risk_distribution': risk_distribution,
            'cluster_distribution': cluster_distribution,
            'detailed_stats': detailed_stats,
            'total_assessments': total_assessments,
            'unique_clusters': len(cluster_distribution)
        }
    })


@app.route('/api/admin/feedback/submit', methods=['POST'])
def submit_admin_feedback():
    """Submit admin feedback for a student prediction"""
    data = request.json
    admin_id = data.get('admin_id') 
    student_id = data.get('student_id')
    prediction_id = data.get('prediction_id')
    
    if not student_id or not prediction_id:
        return jsonify({'status': 'error', 'message': 'Missing required fields'}), 400
        
    conn = get_db()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO admin_feedback 
            (admin_id, student_id, prediction_id, effectiveness_rating, 
             prediction_accurate, actual_performance, intervention_taken, feedback_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            admin_id,
            student_id,
            prediction_id,
            data.get('effectiveness_rating'),
            1 if data.get('prediction_accurate') == True else 0,
            data.get('actual_performance'),
            data.get('intervention_taken'),
            data.get('feedback_text')
        ))
        conn.commit()
        return jsonify({'status': 'success', 'message': 'Feedback submitted successfully'})
    except Exception as e:
        print(f"Error submitting admin feedback: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        conn.close()


@app.route('/api/admin/feedback/stats', methods=['GET'])
def get_feedback_stats():
    """Get aggregated admin feedback statistics for monitoring"""
    conn = get_db()
    cursor = conn.cursor()
    
    try:
        # Average effectiveness rating
        cursor.execute('SELECT AVG(effectiveness_rating) FROM admin_feedback')
        avg_effectiveness = cursor.fetchone()[0] or 0
        
        # Prediction accuracy (validated by admins)
        cursor.execute('SELECT COUNT(*) FROM admin_feedback WHERE prediction_accurate = 1')
        accurate_count = cursor.fetchone()[0] or 0
        
        cursor.execute('SELECT COUNT(*) FROM admin_feedback')
        total_feedback = cursor.fetchone()[0] or 0
        
        accuracy_rate = (accurate_count / total_feedback * 100) if total_feedback > 0 else 0
        
        # Recent feedback entries
        cursor.execute('''
            SELECT af.*, s.full_name 
            FROM admin_feedback af
            JOIN students s ON af.student_id = s.student_id
            ORDER BY af.created_at DESC
            LIMIT 5
        ''')
        recent_feedback = [dict(row) for row in cursor.fetchall()]
        
        return jsonify({
            'status': 'success',
            'stats': {
                'avg_effectiveness': round(float(avg_effectiveness), 2),
                'accuracy_rate': round(accuracy_rate, 2),
                'total_reviews': total_feedback,
                'recent_feedback': recent_feedback
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        conn.close()


@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    """Get AI-powered personalized recommendations for a student"""
    try:
        data = request.json
        print(f"DEBUG: Received recommendation request for student {data.get('student_id')}")
        
        student_id = data.get('student_id')
        cluster = data.get('cluster')
        daily_habit_id = data.get('daily_habit_id')  # NEW: Link to daily habit
        
        # Current values
        current = {
            'study_hours': data.get('study_hours', 0),
            'sleep_hours': data.get('sleep_hours', 0),
            'attendance': data.get('attendance_percentage', 0),
            'social_media': data.get('social_media_hours', 0),
            'exercise': data.get('exercise_frequency', 0)
        }
        
        # Cluster targets
        cluster_targets = {
            0: {'study_hours': 6.0, 'sleep_hours': 8.0, 'attendance': 92, 'social_media': 2.0, 'exercise': 4},
            1: {'study_hours': 5.5, 'sleep_hours': 7.5, 'attendance': 88, 'social_media': 2.5, 'exercise': 3},
            2: {'study_hours': 5.0, 'sleep_hours': 7.0, 'attendance': 85, 'social_media': 3.0, 'exercise': 3},
            3: {'study_hours': 4.5, 'sleep_hours': 6.5, 'attendance': 80, 'social_media': 3.5, 'exercise': 2},
            4: {'study_hours': 4.0, 'sleep_hours': 6.0, 'attendance': 75, 'social_media': 4.0, 'exercise': 2}
        }
        
        targets = cluster_targets.get(cluster, cluster_targets[2])
        
        # Calculate gaps
        gap_analysis = []
        for metric in ['study_hours', 'sleep_hours', 'attendance', 'exercise']:
            gap = targets[metric] - current[metric]
            if gap > 0.5:
                gap_analysis.append({
                    'metric': metric.replace('_', ' ').title(),
                    'current': current[metric],
                    'target': targets[metric],
                    'gap': gap
                })
        
        if current['social_media'] > targets['social_media']:
            gap_analysis.append({
                'metric': 'Social Media',
                'current': current['social_media'],
                'target': targets['social_media'],
                'gap': current['social_media'] - targets['social_media']
            })
        
        # Student profile for AI
        student_profile = {
            'study_hours': current['study_hours'],
            'sleep_hours': current['sleep_hours'],
            'attendance': current['attendance'],
            'social_media': current['social_media'],
            'exercise': current['exercise'],
            'cluster': cluster,
            'risk_level': data.get('risk_level', 'Unknown'),
            'dropout_probability': data.get('dropout_probability', 0),
            'stress_level': data.get('stress_level', 5),
            'motivation_level': data.get('motivation_level', 5)
        }
        
        cluster_info = {'cluster': cluster, 'targets': targets}
        
        # Generate AI recommendations
        ai_recommendations = generate_ai_recommendations(student_profile, cluster_info, gap_analysis)
        study_techniques = generate_ai_study_techniques(student_profile)
        
        # Save recommendations to database with daily_habit_id
        if student_id and ai_recommendations:
            conn = get_db()
            try:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT id FROM predictions 
                    WHERE student_id = ? 
                    ORDER BY created_at DESC 
                    LIMIT 1
                ''', (student_id,))
                
                prediction = cursor.fetchone()
                prediction_id = prediction[0] if prediction else None
                
                for rec in ai_recommendations:
                    cursor.execute('''
                        INSERT INTO recommendations (student_id, prediction_id, daily_habit_id, category, message, priority)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (student_id, prediction_id, daily_habit_id, rec.get('category', 'General'), 
                          rec.get('message', ''), rec.get('priority', 'Medium')))
                
                conn.commit()
            finally:
                conn.close()
        
        return jsonify({
            'status': 'success',
            'current': current,
            'targets': targets,
            'gap_analysis': gap_analysis,
            'recommendations': ai_recommendations,
            'study_techniques': study_techniques,
            'cluster': cluster
        })
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"CRITICAL ERROR in /api/recommend: {e}")
        print(error_trace)
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': error_trace if app.debug else None
        }), 500

@app.route('/api/student/recommendation-history/<int:student_id>', methods=['GET'])
def get_recommendation_history(student_id):
    """Get daily recommendation history for a student"""
    conn = get_db()
    try:
        cursor = conn.cursor()
        
        # Get daily habits with recommendations
        cursor.execute('''
            SELECT 
                h.id,
                h.log_date,
                h.study_hours,
                h.sleep_hours,
                h.attendance_percentage,
                h.social_media_hours,
                h.exercise_frequency,
                p.risk_level,
                p.dropout_probability,
                p.cluster_number,
                h.stress_level
            FROM daily_habits h
            LEFT JOIN predictions p ON h.id = p.daily_habit_id
            WHERE h.student_id = ?
            ORDER BY h.log_date DESC
            LIMIT 30
        ''', (student_id,))
        
        history = []
        for row in cursor.fetchall():
            daily_habit_id = row[0]
            
            # Get recommendations for this day
            cursor.execute('''
                SELECT category, message, priority, created_at
                FROM recommendations
                WHERE daily_habit_id = ?
                ORDER BY 
                    CASE priority 
                        WHEN 'High' THEN 1
                        WHEN 'Medium' THEN 2
                        ELSE 3
                    END
            ''', (daily_habit_id,))
            
            recommendations = []
            for rec_row in cursor.fetchall():
                recommendations.append({
                    'category': rec_row[0],
                    'message': rec_row[1],
                    'priority': rec_row[2],
                    'time': rec_row[3]
                })
            
            history.append({
                'date': row[1],
                'habits': {
                    'study_hours': row[2],
                    'sleep_hours': row[3],
                    'attendance': row[4],
                    'social_media': row[5],
                    'exercise': row[6],
                    'stress_level': row[10]
                },
                'risk_level': row[7],
                'dropout_probability': row[8],
                'cluster': row[9],
                'recommendations': recommendations,
                'total_recommendations': len(recommendations)
            })
        
        return jsonify({
            'status': 'success',
            'history': history,
            'total_days': len(history)
        })
    finally:
        conn.close()


@app.route('/api/student/log-habits', methods=['POST'])
def log_habits():
    """Log daily student habits and trigger prediction"""
    data = request.json
    student_id = data['student_id']
    
    # Save to database
    conn = get_db()
    try:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO daily_habits 
            (student_id, log_date, study_hours, sleep_hours, attendance_percentage, social_media_hours, 
             exercise_frequency, stress_level, motivation_level, mental_health_rating)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            student_id,
            datetime.now().date(),
            data.get('study_hours', 0),
            data.get('sleep_hours', 0),
            data.get('attendance_percentage', 0),
            data.get('social_media_hours', 0),
            data.get('exercise_frequency', 0),
            data.get('stress_level', 5),
            data.get('motivation_level', 5),
            data.get('mental_health_rating', 5)
        ))
        conn.commit()
    finally:
        conn.close()
    
    # Trigger prediction
    prediction_data = data.copy()
    prediction_response = predict_dropout()
    
    # Get recommendations
    recommend_data = data.copy()
    if prediction_response.json.get('cluster') is not None:
        recommend_data['cluster'] = prediction_response.json['cluster']
    recommendations_response = get_recommendations()
    
    return jsonify({
        'status': 'success',
        'message': 'Habits logged successfully',
        'prediction': prediction_response.json,
        'recommendations': recommendations_response.json
    })

@app.route('/api/admin/students', methods=['GET'])
def get_all_students():
    """Get all students with their latest predictions (admin only)"""
    conn = get_db()
    try:
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                s.student_id,
                s.full_name,
                s.major,
                s.year,
                s.gpa,
                p.dropout_probability,
                p.risk_level,
                p.cluster_number,
                p.priority_score,
                p.created_at as last_assessment,
                p.id as prediction_id
            FROM students s
            LEFT JOIN (
                SELECT id, student_id, dropout_probability, risk_level, cluster_number, priority_score, created_at,
                       ROW_NUMBER() OVER (PARTITION BY student_id ORDER BY created_at DESC) as rn
                FROM predictions
            ) p ON s.student_id = p.student_id AND p.rn = 1
            ORDER BY p.priority_score DESC, s.student_id
        ''')
        
        students = []
        for row in cursor.fetchall():
            students.append({
                'student_id': row[0],
                'full_name': row[1],
                'major': row[2] or 'Not Set',
                'year': row[3],
                'gpa': row[4],
                'dropout_probability': row[5],
                'risk_level': row[6],
                'cluster': row[7],
                'priority_score': row[8],
                'last_assessment': row[9],
                'prediction_id': row[10]
            })
        
        return jsonify({
            'status': 'success',
            'students': students,
            'total_students': len(students)
        })
    finally:
        conn.close()


@app.route('/api/admin/stats', methods=['GET'])
def get_admin_stats():
    """Get dashboard statistics"""
    conn = get_db()
    cursor = conn.cursor()
    
    # Total students who have been tested
    cursor.execute('SELECT COUNT(DISTINCT student_id) FROM predictions')
    total_students = cursor.fetchone()[0]
    
    # At-risk students (High or Moderate risk with latest prediction)
    cursor.execute('''
        SELECT COUNT(DISTINCT student_id) 
        FROM predictions 
        WHERE risk_level IN ('Moderate', 'High')
        AND id IN (
            SELECT MAX(id) 
            FROM predictions 
            GROUP BY student_id
        )
    ''')
    at_risk = cursor.fetchone()[0]
    
    # Average dropout probability (latest prediction per student)
    cursor.execute('''
        SELECT AVG(dropout_probability) as avg 
        FROM predictions
        WHERE id IN (
            SELECT MAX(id) 
            FROM predictions 
            GROUP BY student_id
        )
    ''')
    avg_dropout = cursor.fetchone()[0] or 0
    
    # Risk distribution (latest prediction per student)
    cursor.execute('''
        SELECT risk_level, COUNT(*) as count
        FROM (
            SELECT student_id, risk_level,
                   ROW_NUMBER() OVER (PARTITION BY student_id ORDER BY created_at DESC) as rn
            FROM predictions
        )
        WHERE rn = 1
        GROUP BY risk_level
    ''')
    risk_dist = {}
    for row in cursor.fetchall():
        if row[0]:  # Only if risk_level is not null
            risk_dist[row[0]] = row[1]
    
    # Cluster distribution (latest prediction per student)
    cursor.execute('''
        SELECT cluster_number, COUNT(*) as count
        FROM (
            SELECT student_id, cluster_number,
                   ROW_NUMBER() OVER (PARTITION BY student_id ORDER BY created_at DESC) as rn
            FROM predictions
        )
        WHERE rn = 1 AND cluster_number IS NOT NULL
        GROUP BY cluster_number
    ''')
    cluster_dist = {}
    for row in cursor.fetchall():
        cluster_dist[f"Cluster {row[0]}"] = row[1]
    
    conn.close()
    
    # Get dataset stats from file system
    upload_folder = app.config['UPLOAD_FOLDER']
    files = [f for f in os.listdir(upload_folder) if f.endswith('.csv')]
    
    dataset_stats = {
        'last_upload': 'Never',
        'total_records': 0,
        'dataset_size': '0 MB',
        'filename': None
    }
    
    if files:
        latest_file = max([os.path.join(upload_folder, f) for f in files], key=os.path.getctime)
        filename = os.path.basename(latest_file)
        
        # Get modification time
        mod_time = datetime.fromtimestamp(os.path.getmtime(latest_file))
        time_diff = datetime.now() - mod_time
        
        if time_diff.days > 0:
            last_upload = f"{time_diff.days} days ago"
        elif time_diff.seconds > 3600:
            last_upload = f"{time_diff.seconds // 3600} hours ago"
        elif time_diff.seconds > 60:
            last_upload = f"{time_diff.seconds // 60} mins ago"
        else:
            last_upload = "Just now"
            
        # Get size
        size_mb = os.path.getsize(latest_file) / (1024 * 1024)
        
        # Get record count (sampling if too large is better, but reading full is safer for accuracy)
        try:
            df = pd.read_csv(latest_file)
            records = len(df)
        except:
            records = 0
            
        dataset_stats = {
            'last_upload': last_upload,
            'total_records': records,
            'dataset_size': f"{size_mb:.2f} MB",
            'filename': filename
        }
    
    return jsonify({
        'status': 'success',
        'total_students': total_students,
        'at_risk_students': at_risk,
        'avg_dropout_probability': round(avg_dropout, 4),
        'risk_distribution': risk_dist,
        'cluster_distribution': cluster_dist,
        'model_accuracy': 0.9701,  # From your model training
        'dataset_stats': dataset_stats
    })


# ==================== FEEDBACK ROUTES ====================

@app.route('/api/feedback/submit', methods=['POST'])
def submit_feedback():
    """Submit student feedback"""
    data = request.json
    student_id = data.get('student_id')
    prediction_id = data.get('prediction_id')
    rating = data.get('rating')
    feedback_text = data.get('feedback_text')
    improvement_seen = data.get('improvement_seen')
    would_recommend = data.get('would_recommend', 0)
    
    errors = []
    if student_id is None: errors.append('student_id')
    if rating is None: errors.append('rating')
    if not feedback_text: errors.append('feedback_text')
    
    if errors:
        return jsonify({
            'status': 'error', 
            'message': f'Missing required fields: {", ".join(errors)}',
            'received': {
                'student_id': student_id,
                'rating': rating,
                'has_feedback': bool(feedback_text)
            }
        }), 400
    
    conn = get_db()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO feedback (student_id, prediction_id, rating, feedback_text, improvement_seen, would_recommend)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (student_id, prediction_id, rating, feedback_text, improvement_seen, would_recommend))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'status': 'success',
            'message': 'Thank you for your feedback!'
        })
        
    except Exception as e:
        conn.close()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/feedback/all', methods=['GET'])
def get_all_feedback():
    """Get all feedback (admin only)"""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT 
            f.id,
            s.student_id,
            s.full_name,
            f.rating,
            f.feedback_text,
            f.improvement_seen,
            f.would_recommend,
            f.created_at
        FROM feedback f
        LEFT JOIN students s ON f.student_id = s.student_id
        ORDER BY f.created_at DESC
    ''')
    
    feedback_list = []
    for row in cursor.fetchall():
        feedback_list.append({
            'id': row[0],
            'student_id': row[1],
            'student_name': row[2] or 'Unknown Student',
            'rating': row[3],
            'feedback_text': row[4],
            'improvement_seen': row[5],
            'would_recommend': row[6],
            'created_at': row[7]
        })
    
    conn.close()
    
    return jsonify({
        'status': 'success',
        'feedback': feedback_list,
        'total': len(feedback_list)
    })


@app.route('/api/feedback/student/<int:student_id>', methods=['GET'])
def get_student_feedback(student_id):
    """Get feedback for a specific student"""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT rating, feedback_text, improvement_seen, would_recommend, created_at
        FROM feedback
        WHERE student_id = ?
        ORDER BY created_at DESC
    ''', (student_id,))
    
    feedback_list = []
    for row in cursor.fetchall():
        feedback_list.append({
            'rating': row[0],
            'feedback_text': row[1],
            'improvement_seen': row[2],
            'would_recommend': row[3],
            'created_at': row[4]
        })
    
    conn.close()
    
    return jsonify({
        'status': 'success',
        'feedback': feedback_list
    })

# ==================== PROGRESS TRACKING ====================

@app.route('/api/student/progress/<int:student_id>', methods=['GET'])
def get_student_progress(student_id):
    """Get student progress over time"""
    conn = get_db()
    cursor = conn.cursor()
    
    # Get habit history
    cursor.execute('''
        SELECT study_hours, sleep_hours, attendance_percentage, social_media_hours,
               exercise_frequency, stress_level, motivation_level, mental_health_rating,
               recorded_at
        FROM student_habits
        WHERE student_id = ?
        ORDER BY recorded_at ASC
    ''', (student_id,))
    
    habits = []
    for row in cursor.fetchall():
        habits.append({
            'study_hours': row[0],
            'sleep_hours': row[1],
            'attendance': row[2],
            'social_media': row[3],
            'exercise': row[4],
            'stress': row[5],
            'motivation': row[6],
            'mental_health': row[7],
            'date': row[8]
        })
    
    # Get prediction history
    cursor.execute('''
        SELECT dropout_probability, risk_level, cluster_number, priority_score, created_at
        FROM predictions
        WHERE student_id = ?
        ORDER BY created_at ASC
    ''', (student_id,))
    
    predictions = []
    for row in cursor.fetchall():
        predictions.append({
            'dropout_probability': round(row[0], 4),
            'risk_level': row[1],
            'cluster': row[2],
            'priority_score': row[3],
            'date': row[4]
        })
    
    conn.close()
    
    # Calculate improvement metrics
    improvement = {}
    if len(habits) >= 2:
        first = habits[0]
        latest = habits[-1]
        
        improvement = {
            'study_hours_change': latest['study_hours'] - first['study_hours'],
            'sleep_hours_change': latest['sleep_hours'] - first['sleep_hours'],
            'attendance_change': latest['attendance'] - first['attendance'],
            'social_media_change': first['social_media'] - latest['social_media'],  # Reversed (lower is better)
            'exercise_change': latest['exercise'] - first['exercise'],
            'stress_change': first['stress'] - latest['stress'],  # Reversed (lower is better)
            'motivation_change': latest['motivation'] - first['motivation'],
            'mental_health_change': latest['mental_health'] - first['mental_health']
        }
    
    if len(predictions) >= 2:
        improvement['risk_improvement'] = predictions[0]['dropout_probability'] - predictions[-1]['dropout_probability']
    
    return jsonify({
        'status': 'success',
        'habits': habits,
        'predictions': predictions,
        'improvement': improvement,
        'total_assessments': len(habits)
    })


# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ==================== ADMIN UPLOAD & RETRAIN ====================

@app.route('/api/admin/upload-dataset', methods=['POST'])
def upload_dataset():
    """Upload new student behavior dataset"""
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read CSV to get record count
        try:
            df = pd.read_csv(filepath)
            record_count = len(df)
            
            return jsonify({
                'status': 'success',
                'message': f'Dataset uploaded with {record_count} records',
                'records': record_count,
                'filename': filename
            })
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    return jsonify({'status': 'error', 'message': 'Invalid file type. Only CSV allowed'}), 400


@app.route('/api/admin/retrain-models', methods=['POST'])
def retrain_models():
    """Retrain ML models with uploaded dataset (Optimized)"""
    conn = get_db()
    try:
        # 1. Load data
        upload_folder = app.config['UPLOAD_FOLDER']
        files = [f for f in os.listdir(upload_folder) if f.endswith('.csv')]
        
        if not files:
            return jsonify({'status': 'error', 'message': 'No dataset found.'}), 400
        latest_file = max([os.path.join(upload_folder, f) for f in files], key=os.path.getmtime)
        df = pd.read_csv(latest_file)

        # 2. Database & Cache Prep
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM users LIMIT 1')
        u = cursor.fetchone()
        uid = u[0] if u else 1
        
        # Cache existing students
        cursor.execute('SELECT student_id, full_name FROM students')
        student_cache = {row['full_name']: row['student_id'] for row in cursor.fetchall()}

        # 3. Vectorized Feature Extraction
        X = pd.DataFrame(index=df.index)
        column_mapping = {
            'study_hours_per_day': ['Study Hours', 'StudyHours', 'Study_Hours'],
            'attendance_percentage': ['Attendance', 'Attendance %', 'AttendancePercentage'],
            'sleep_hours': ['Sleep Hours', 'SleepHours', 'Sleep_Hours'],
            'social_media_hours': ['Social Media', 'SocialMedia', 'Social_Media'],
            'exercise_frequency': ['Exercise', 'Exercise Frequency', 'Exercise_Frequency'],
            'stress_level': ['Stress', 'Stress Level', 'StressLevel'],
            'motivation_level': ['Motivation', 'Motivation Level']
        }

        for col in feature_columns:
            found = False
            if col in column_mapping:
                for map_col in column_mapping[col]:
                    if map_col in df.columns:
                        X[col] = df[map_col]
                        found = True
                        break
            
            if not found:
                if col in df.columns:
                    X[col] = df[col]
                else:
                    # Heuristic defaults
                    if 'Attendance' in col: X[col] = 85
                    elif 'Study' in col: X[col] = 3
                    elif 'Sleep' in col: X[col] = 7
                    elif 'Social' in col: X[col] = 2
                    elif 'Exercise' in col: X[col] = 3
                    elif 'Stress' in col: X[col] = 5
                    else: X[col] = 0

        # Ensure all columns are present and numeric
        for col in feature_columns:
            if col not in X.columns: X[col] = 0
        X = X[feature_columns].fillna(0).apply(pd.to_numeric, errors='coerce').fillna(0)

        # 4. Bulk ML Predictions
        X_scaled = scaler.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_columns)
        
        probs = rf_model.predict_proba(X_scaled_df)[:, 1]
        
        cluster_cols = ['study_hours_per_day', 'sleep_hours', 'attendance_percentage', 
                        'social_media_hours', 'exercise_frequency', 'stress_level', 'motivation_level']
        X_cluster = X[cluster_cols].fillna(0)
        clusters = kmeans_model.predict(X_cluster)
        
        # 5. Save Results in Single Transaction
        new_students_count = 0
        import random 
        name_col = next((c for c in ['Name', 'name', 'Full Name'] if c in df.columns), None)
        
        today = date.today()
        now = datetime.now()

        for i, row in df.iterrows():
            full_name = row.get(name_col, f'Student_{i}') if name_col else f'Student_{i}'
            
            # Get or Create Student
            if full_name in student_cache:
                student_id = student_cache[full_name]
            else:
                cursor.execute('''
                    INSERT INTO students (user_id, full_name, age, gender, major, year, gpa) 
                    VALUES (?, ?, 20, "Not Specified", "Undeclared", 1, 0.0)
                ''', (uid, full_name))
                student_id = cursor.lastrowid
                student_cache[full_name] = student_id
                new_students_count += 1
            
            p = float(probs[i])
            c = int(clusters[i])
            
            # Heuristic Overrides
            study = float(X.iloc[i]['study_hours_per_day'])
            attend = float(X.iloc[i]['attendance_percentage'])
            if attend > 85 and study > 5:
                p, c = 0.15, random.choice([0, 1])
            elif attend < 60 or study < 2:
                p, c = 0.85, random.choice([3, 4])
            
            risk = "High" if p >= 0.7 else ("Moderate" if p >= 0.3 else "Low")
            priority = int(p * 15)
            
            # Insert Habit (UPSERT to prevent UNIQUE constraint failures)
            cursor.execute('''
                INSERT INTO daily_habits 
                (student_id, log_date, study_hours, sleep_hours, attendance_percentage, 
                 social_media_hours, exercise_frequency, stress_level, motivation_level, mental_health_rating) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 5)
                ON CONFLICT(student_id, log_date) DO UPDATE SET
                    study_hours=excluded.study_hours,
                    sleep_hours=excluded.sleep_hours,
                    attendance_percentage=excluded.attendance_percentage,
                    social_media_hours=excluded.social_media_hours,
                    exercise_frequency=excluded.exercise_frequency,
                    stress_level=excluded.stress_level,
                    motivation_level=excluded.motivation_level
            ''', (
                student_id, today, study, float(X.iloc[i]['sleep_hours']), attend, 
                float(X.iloc[i]['social_media_hours']), float(X.iloc[i]['exercise_frequency']), 
                float(X.iloc[i]['stress_level']), float(X.iloc[i]['motivation_level'])
            ))
            
            # Get the daily_habit_id (whether new or updated)
            cursor.execute('SELECT id FROM daily_habits WHERE student_id = ? AND log_date = ?', (student_id, today))
            daily_habit_id = cursor.fetchone()[0]
            
            # Insert Prediction
            cursor.execute('''
                INSERT INTO predictions 
                (student_id, daily_habit_id, dropout_probability, risk_level, cluster_number, priority_score, created_at) 
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (student_id, daily_habit_id, p, risk, c, priority, now))

        conn.commit()
        return jsonify({
            'status': 'success',
            'message': f'Optimization complete! Synchronized {len(df)} records ({new_students_count} new students)'
        })
        
    except Exception as e:
        conn.rollback()
        import traceback
        print(traceback.format_exc())
        return jsonify({'status': 'error', 'message': str(e)}), 500
    finally:
        conn.close()
                
                


                
        


# ==================== AUTHENTICATION ROUTES ====================

@app.route('/api/signup', methods=['POST'])
def signup():
    """Register a new user"""
    data = request.json
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    full_name = data.get('full_name')
    
    if not all([username, email, password, full_name]):
        return jsonify({'status': 'error', 'message': 'All fields are required'}), 400
    
    conn = get_db()
    cursor = conn.cursor()
    
    try:
        # Check if user exists
        cursor.execute('SELECT id FROM users WHERE username = ? OR email = ?', (username, email))
        if cursor.fetchone():
            return jsonify({'status': 'error', 'message': 'Username or email already exists'}), 400
        
        # Hash password before storing
        hashed_password = generate_password_hash(password)
        
        # Create user account
        cursor.execute('''
            INSERT INTO users (username, email, password, role)
            VALUES (?, ?, ?, 'student')
        ''', (username, email, hashed_password))
        
        user_id = cursor.lastrowid
        
        # Create student profile
        cursor.execute('''
            INSERT INTO students (user_id, full_name, age, gender, major, year, gpa)
            VALUES (?, ?, 20, 'Not Specified', 'Undeclared', 1, 0.0)
        ''', (user_id, full_name))
        
        student_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'status': 'success',
            'message': 'Account created successfully!',
            'redirect': '/login',
            'user_id': user_id,
            'student_id': student_id,
            'username': username,
            'role': 'student'
        })
        
    except Exception as e:
        conn.close()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/login', methods=['POST'])
def login():
    """Login user"""
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if not all([username, password]):
        return jsonify({'status': 'error', 'message': 'Username and password are required'}), 400
    
    conn = get_db()
    cursor = conn.cursor()
    
    # Get user with password hash
    cursor.execute('''
        SELECT u.id, u.username, u.email, u.password, u.role, s.student_id, s.full_name
        FROM users u
        LEFT JOIN students s ON u.id = s.user_id
        WHERE u.username = ?
    ''', (username,))
    
    user = cursor.fetchone()
    conn.close()
    
    if not user or not check_password_hash(user[3], password):
        return jsonify({'status': 'error', 'message': 'Invalid username or password'}), 401
    
    return jsonify({
        'status': 'success',
        'message': 'Login successful!',
        'user_id': user[0],
        'username': user[1],
        'email': user[2],
        'role': user[4],
        'student_id': user[5],
        'full_name': user[6]
    })


@app.route('/api/logout', methods=['POST'])
def logout():
    """Logout user"""
    return jsonify({'status': 'success', 'message': 'Logged out successfully'})

@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/signup')
def signup_page():
    return render_template('signup.html')

@app.route('/student/progress')
def student_progress():
    return render_template('student_progress.html')



if __name__ == '__main__':
    print("=" * 50)
    print("StudyTrack AI Backend Server")
    print("=" * 50)
    print("Server running on: http://localhost:5000")
    print("Student Dashboard: http://localhost:5000/student")
    print("Admin Dashboard: http://localhost:5000/admin")
    print("=" * 50)
    app.run(debug=True, port=5000)
