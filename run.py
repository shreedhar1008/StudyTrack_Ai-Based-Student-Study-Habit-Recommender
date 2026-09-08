#!/usr/bin/env python
"""
StudyTrack AI - Application Entry Point
"""
import os
import sys

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app import app

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'True').lower() in ('true', '1', 't')
    print('=' * 60)
    print('  StudyTrack AI - Academic Intelligence & Retention System')
    print(f'  Server running at: http://127.0.0.1:{port}')
    print('=' * 60)
    app.run(host='0.0.0.0', port=port, debug=debug)
