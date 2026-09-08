# StudyTrack AI - Academic Intelligence & Student Retention Platform

[![CI/CD Pipeline](https://github.com/shreedhar1008/StudyTrack_Ai-Based-Student-Study-Habit-Recommender/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/shreedhar1008/StudyTrack_Ai-Based-Student-Study-Habit-Recommender/actions)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-black.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Render](https://img.shields.io/badge/Render-Deployed-46E3B7.svg?logo=render&logoColor=white)](https://render.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Cloud%20DB-336791.svg?logo=postgresql&logoColor=white)](https://neon.tech)

An enterprise-grade, machine learning-powered academic analytics platform designed to predict student dropout risk, cultivate personalized study habits, and provide comprehensive institutional intelligence for educators.

---

## 📸 Screenshots

### 1. Landing Page
![Homepage](docs/demo/homepage.png)

### 2. About Us Platform Architecture
![About Us](docs/demo/about_page.png)

### 3. Student Portal & Habit Tracking
![Student Dashboard](docs/demo/student_dashboard.png)

### 4. Student Performance Journey
![Student Progress](docs/demo/student_progress.png)

### 5. Institutional Leadership Dashboard
![Admin Dashboard](docs/demo/admin_dashboard.png)

---

## 📁 Clean & Modular Project Structure

```
STUDYTRACK AI/
├── backend/
│   ├── __init__.py
│   ├── routes/
│   │   ├── __init__.py
│   │   └── extended_routes.py    # Modular API endpoints & AI Tutor handlers
│   ├── database/
│   │   ├── __init__.py
│   │   ├── adapter.py            # PostgreSQL + SQLite connection pool adapter
│   │   ├── migrate.py            # SQLite -> PostgreSQL migration tool
│   │   └── seed.py               # Database seeder
│   └── services/                 # ML & AI inference services
│       └── __init__.py
├── frontend/
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css         # Responsive white-theme design system
│   │   └── images/               # Wordmark logos, illustrations & favicons
│   │       ├── logo.png
│   │       ├── logo_darkbg.png
│   │       ├── favicon.png
│   │       └── ...
│   └── templates/                # Responsive Jinja2 HTML templates
│       ├── index.html
│       ├── about.html
│       ├── login.html
│       ├── signup.html
│       ├── student_dashboard.html
│       ├── student_progress.html
│       └── admin_dashboard.html
├── models/                       # Pre-trained ML models & scalers
│   ├── rf_dropout_model.pkl      # Random Forest Risk Classifier
│   ├── kmeans_model.pkl          # K-Means Student Archetype Clusterer
│   ├── scaler.pkl                # Standard feature scaler
│   └── feature_columns.pkl       # 52-factor model feature definition
├── database/                     # Local Database & Preprocessed Datasets
│   ├── studytrack.db             # Local SQLite database (fallback)
│   └── datasets/
│       └── student_study_hours_preprocessed.csv
├── uploads/                      # User-uploaded dataset storage
│   ├── .gitkeep
│   └── enhanced_student_habits_performance_dataset.csv
├── docs/                         # Project media and documentation assets
│   └── demo/
│       ├── homepage.png
│       ├── student_dashboard.png
│       ├── admin_dashboard.png
│       └── demo_video.webm
├── app.py                        # Primary Flask backend server
├── run.py                        # Alternative application runner
├── requirements.txt              # Project dependencies
├── .env.example                  # Environment configuration template
├── .env                          # Local credentials
├── .gitignore                    # Version control ignore rules
├── README.md                     # Project documentation
└── SETUP.md                      # Setup instructions
```

---

## 🚀 Key Capabilities

- **Predictive Risk Modeling**: Random Forest classifier evaluating 52+ multi-dimensional behavioral factors.
- **Cluster Personalization**: K-Means clustering assigning learners to behavioral archetypes with tailored study timetables.
- **Daily Habit Tracking**: Interactive timeline journals, habit logging, and visual growth analytics.
- **AI Tutoring & Insights**: Generative study recommendations and intervention strategies powered by Groq LLM.
- **Institutional Intelligence**: Leadership dashboard featuring cohort analytics, student filters, and dynamic CSV model retraining.
- **Dual Database Architecture**: Seamless support for Cloud PostgreSQL (Neon/Supabase) and local SQLite.

---

## 🛠️ Quickstart Installation

1. **Activate Virtual Environment**
```powershell
python -m venv .venv
.\.venv\Scriptsctivate
```

2. **Install Dependencies**
```powershell
pip install -r requirements.txt
```

3. **Configure Environment Variables**
Copy `.env.example` to `.env` and fill in your keys:
```ini
GROQ_API_KEY=your_groq_api_key_here
FLASK_SECRET_KEY=your_secret_key
DATABASE_URL=postgresql://user:pass@host/dbname?sslmode=require
```

4. **Start the Application**
```powershell
python app.py
# or
python run.py
```

5. **Access Application Routes**
- **Home**: http://localhost:5000/
- **About Us**: http://localhost:5000/about
- **Student Portal**: http://localhost:5000/student
- **Student Progress**: http://localhost:5000/student/progress
- **Admin Dashboard**: http://localhost:5000/admin
- **Login**: http://localhost:5000/login
- **Sign Up**: http://localhost:5000/signup

---

## 🛡️ License
© 2026 StudyTrack AI. All rights reserved.
