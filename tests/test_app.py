import unittest
import os
import sys
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

os.environ['DATABASE_URL'] = ''
os.environ['SQLITE_DB_PATH'] = os.path.join(BASE_DIR, 'database', 'test_studytrack.db')

from backend.database.adapter import init_db, get_db
import app


class StudyTrackTestSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()
        cls.client = app.app.test_client()

    @classmethod
    def tearDownClass(cls):
        test_db = os.environ.get('SQLITE_DB_PATH')
        if test_db and os.path.exists(test_db):
            try:
                os.remove(test_db)
            except Exception:
                pass

    def test_01_models_loaded(self):
        """Verify all ML models and scaler artifacts are properly loaded"""
        self.assertIsNotNone(app.rf_model, "Random Forest model should be loaded")
        self.assertIsNotNone(app.kmeans_model, "K-Means model should be loaded")
        self.assertIsNotNone(app.scaler, "Scaler artifact should be loaded")
        self.assertIsNotNone(app.feature_columns, "Feature columns list should be loaded")

    def test_02_public_pages(self):
        """Verify public web pages return 200 OK"""
        pages = ['/', '/about', '/login', '/signup', '/student', '/student/progress', '/admin']
        for page in pages:
            res = self.client.get(page)
            self.assertEqual(res.status_code, 200, f"Page {page} should return 200 OK")

    def test_03_prediction_api(self):
        """Verify AI dropout prediction API"""
        payload = {
            'student_id': 1,
            'study_hours': 5.0,
            'sleep_hours': 7.5,
            'attendance_percentage': 88.0,
            'social_media_hours': 1.5,
            'exercise_frequency': 4,
            'stress_level': 3,
            'motivation_level': 8,
            'mental_health_rating': 8
        }
        res = self.client.post('/api/predict', json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get('status'), 'success')
        self.assertIn('dropout_probability', data)
        self.assertIn('risk_level', data)
        self.assertIn('cluster', data)
        self.assertIn(data['risk_level'], ['Low', 'Moderate', 'High'])

    def test_04_admin_analytics_api(self):
        """Verify Admin Analytics aggregated metrics API"""
        res = self.client.get('/api/admin/analytics')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get('status'), 'success')
        analytics = data.get('analytics', {})
        self.assertIn('total_students', analytics)
        self.assertIn('avg_dropout_probability', analytics)
        self.assertIn('risk_distribution', analytics)
        self.assertIn('cluster_distribution', analytics)

    def test_05_student_recommendations_api(self):
        """Verify rule-based recommendations endpoint"""
        payload = {
            'student_id': 1,
            'study_hours': 2.0,
            'sleep_hours': 5.0,
            'attendance': 65.0,
            'social_media': 5.0,
            'exercise': 1,
            'stress_level': 8,
            'motivation_level': 3,
            'mental_health_rating': 4
        }
        res = self.client.post('/api/recommend', json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get('status'), 'success')


if __name__ == '__main__':
    unittest.main(verbosity=2)
