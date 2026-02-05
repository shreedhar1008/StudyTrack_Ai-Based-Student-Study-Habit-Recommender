# 🚀 Quick Setup Guide

Follow these steps to get StudyTrack running:

## Step 1: Create .env File

Create a file named `.env` in the project root (d:\NOTES\StudyTrack\STSSHR\.env) with:

```ini
GROQ_API_KEY=your_actual_groq_api_key_here
FLASK_SECRET_KEY=any_random_string_you_want
FLASK_ENV=development
```

**Get your FREE Groq API key**: https://console.groq.com/

## Step 2: Install Dependencies

```powershell
cd d:\NOTES\StudyTrack\STSSHR
pip install -r requirements.txt
```

## Step 3: (Optional) Reset Database

If you have an old database, delete it to get the new schema with all features:

```powershell
if (Test-Path "studytrack.db") { Remove-Item "studytrack.db" }
```

## Step 4: Run the Application

```powershell
python app.py
```

## Step 5: Access the App

Open your browser to:
- **Login**: http://localhost:5000/login
- **Student Dashboard**: http://localhost:5000/student  
- **Admin Dashboard**: http://localhost:5000/admin

## Default Credentials

**Admin**: admin / admin123
**Student**: student1 / student123

---

## To Use New Endpoints

The new features are in `new_endpoints.py`. To activate them, add this near the end of `app.py` (before `if __name__ == '__main__':`):

```python
# Import and register new features
try:
    from new_endpoints import register_new_endpoints
    register_new_endpoints(app, get_db, jsonify, datetime, generate_password_hash)
    print("✅ New endpoints registered")
except ImportError as e:
    print(f"⚠️  New endpoints not loaded: {e}")
```

Then restart the server. The new endpoints will be available:
- GET `/api/admin/retraining-history`
- POST `/api/admin/save-retraining-result`
- GET `/api/student/check-returning-user/<id>`
- POST `/api/student/log-session`

---

## Testing the Fixes

### Test 1: Verify API Key Loading
```powershell
python app.py
# Should print either:
# - "Models loaded successfully!" (if API key is set) 
# - "WARNING: GROQ_API_KEY not found" (if not set)
```

### Test 2: Verify Password Hashing
```powershell
sqlite3 studytrack.db "SELECT username, substr(password,1,10) FROM users"
# Should show "pbkdf2:" prefix (hash), NOT plaintext passwords
```

### Test 3: Test Habit Logging
- Login as student1
- Fill out habit form
- Submit - should work without errors
- Check database: `sqlite3 studytrack.db "SELECT * FROM daily_habits"`

### Test 4: Test New Endpoints
```powershell
# Check returning user (after logging in)
curl http://localhost:5000/api/student/check-returning-user/1

# Check retraining history
curl http://localhost:5000/api/admin/retraining-history
```

---

## Troubleshooting

**"GROQ_API_KEY not found"**
→ Make sure you created `.env` file with your actual API key

**"No module named 'dotenv'"**
→ Run: `pip install python-dotenv`

**Database errors**
→ Delete studytrack.db and restart app to recreate with new schema

**Port 5000 in use**
→ Change port in app.py line 1516 to 5001

---

All set! 🎉
