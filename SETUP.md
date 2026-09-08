# 🚀 StudyTrack AI - Setup & Deployment Guide

Quick step-by-step instructions to configure and run the StudyTrack AI platform.

---

## 1. Environment Configuration

Create a `.env` file in the project root:

```ini
GROQ_API_KEY=your_actual_groq_api_key_here
FLASK_SECRET_KEY=any_random_secret_key_here
FLASK_ENV=development
# Optional: Cloud PostgreSQL (Neon / Supabase)
DATABASE_URL=postgresql://user:password@host/dbname?sslmode=require
```

> **Groq API Key**: Obtain a free API key at [console.groq.com](https://console.groq.com/).

---

## 2. Install Dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

---

## 3. Database Initialization & Seeding

The application automatically initializes database schemas on startup.

- To manually seed sample student data:
```powershell
python backend/database/seed.py
```

- To migrate data from SQLite to PostgreSQL:
```powershell
python backend/database/migrate.py
```

---

## 4. Run the Application

```powershell
python app.py
# or
python run.py
```

---

## 5. Access the Platform

Open your browser to:
- **Landing Page**: [http://localhost:5000/](http://localhost:5000/)
- **About Us**: [http://localhost:5000/about](http://localhost:5000/about)
- **Login**: [http://localhost:5000/login](http://localhost:5000/login)
- **Sign Up**: [http://localhost:5000/signup](http://localhost:5000/signup)
- **Student Dashboard**: [http://localhost:5000/student](http://localhost:5000/student)
- **Student Progress**: [http://localhost:5000/student/progress](http://localhost:5000/student/progress)
- **Admin Dashboard**: [http://localhost:5000/admin](http://localhost:5000/admin)
