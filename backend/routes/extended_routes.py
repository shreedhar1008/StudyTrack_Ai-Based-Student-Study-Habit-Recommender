# New API endpoints for enhanced features

def register_new_endpoints(app, get_db, jsonify, datetime, generate_password_hash):
    """Register new API endpoints to the Flask app"""
    
    # ==================== MODEL RETRAINING TRACKING ====================
    
    @app.route('/api/admin/retraining-history', methods=['GET'])
    def get_retraining_history():
        """Get all model retraining history for admin dashboard"""
        conn = get_db()
        try:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, dataset_filename, dataset_records, training_date, 
                       model_accuracy, previous_accuracy, improvement, training_duration, notes
                FROM model_retraining_history
                ORDER BY training_date DESC
            ''')
            
            history = []
            for row in cursor.fetchall():
                history.append({
                    'id': row[0],
                    'dataset_filename': row[1],
                    'records': row[2],
                    'training_date': row[3],
                    'accuracy': round(row[4], 4) if row[4] else None,
                    'previous_accuracy': round(row[5], 4) if row[5] else None,
                    'improvement': round(row[6], 4) if row[6] else None,
                    'duration': row[7],
                    'notes': row[8]
                })
            
            return jsonify({
                'status': 'success',
                'history': history,
                'total_retraining_sessions': len(history)
            })
        finally:
            conn.close()
    
    @app.route('/api/admin/save-retraining-result', methods=['POST'])
    def save_retraining_result():
        """Save model retraining results to history"""
        from flask import request
        data = request.json
        
        conn = get_db()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO model_retraining_history 
                (dataset_filename, dataset_records, model_accuracy, previous_accuracy, 
                 improvement, training_duration, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                data.get('dataset_filename'),
                data.get('dataset_records'),
                data.get('model_accuracy'),
                data.get('previous_accuracy'),
                data.get('improvement'),
                data.get('training_duration'),
                data.get('notes', '')
            ))
            
            conn.commit()
            retraining_id = cursor.lastrowid
            conn.close()
            
            return jsonify({
                'status': 'success',
                'message': 'Retraining result saved',
                'retraining_id': retraining_id
            })
            
        except Exception as e:
            conn.close()
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    # ==================== RETURNING USER EXPERIENCE ====================
    
    @app.route('/api/student/check-returning-user/<int:student_id>', methods=['GET'])
    def check_returning_user(student_id):
        """Check if user is returning and get their last session info"""
        conn = get_db()
        try:
            cursor = conn.cursor()
            
            # Get user's last session  
            cursor.execute('''
                SELECT login_time, last_habit_log_date, total_logins
                FROM user_sessions
                WHERE student_id = ?
                ORDER BY login_time DESC
                LIMIT 1
            ''', (student_id,))
            
            last_session = cursor.fetchone()
            
            # Get latest habit log
            cursor.execute('''
                SELECT log_date FROM daily_habits
                WHERE student_id = ?
                ORDER BY log_date DESC
                LIMIT 1
            ''', (student_id,))
            
            last_habit = cursor.fetchone()
            
            # Get stats: total habits logged
            cursor.execute('''
                SELECT COUNT(DISTINCT log_date) as total_days
                FROM daily_habits
                WHERE student_id = ?
            ''', (student_id,))
            
            total_days = cursor.fetchone()[0]
            
            is_returning = last_session is not None
            today = datetime.now().date().isoformat()
            needs_log = not last_habit or last_habit[0] != today
            
            result = {
                'is_returning_user': is_returning,
                'total_logins': last_session[2] if last_session else 0,
                'total_habit_days': total_days,
                'last_habit_log_date': last_habit[0] if last_habit else None,
                'needs_new_habit_log': needs_log
            }
            
            if is_returning and needs_log:
                result['message'] = f"Welcome back! You've tracked {total_days} days. Log today's habits to continue!"
            elif is_returning and not needs_log:
                result['message'] = "Welcome back! You've already logged today's habits."
            else:
                result['message'] = "Welcome! Let's start tracking your study habits."
            
            return jsonify(result)
        finally:
            conn.close()
    
    @app.route('/api/student/log-session', methods=['POST'])
    def log_session():
        """Log user session for tracking returning users"""
        from flask import request
        data = request.json
        student_id = data.get('student_id')
        user_id = data.get('user_id')
        
        conn = get_db()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT total_logins FROM user_sessions 
                WHERE student_id = ? 
                ORDER BY login_time DESC 
                LIMIT 1
            ''', (student_id,))
            
            existing = cursor.fetchone()
            new_total = (existing[0] + 1) if existing else 1
            
            cursor.execute('''
                INSERT INTO user_sessions (user_id, student_id, total_logins)
                VALUES (?, ?, ?)
            ''', (user_id, student_id, new_total))
            
            conn.commit()
            conn.close()
            
            return jsonify({
                'status': 'success',
                'total_logins': new_total
            })
            
        except Exception as e:
            conn.close()
            return jsonify({'status': 'error', 'message': str(e)}), 500
