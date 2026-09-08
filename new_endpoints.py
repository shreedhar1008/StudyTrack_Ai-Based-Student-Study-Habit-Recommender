"""
StudyTrack AI - Extended Routes Shim (Backward Compatibility)
Forwards all route registrations to backend.routes.extended_routes
"""

from backend.routes.extended_routes import register_new_endpoints

__all__ = ['register_new_endpoints']
