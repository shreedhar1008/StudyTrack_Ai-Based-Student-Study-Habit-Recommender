"""
StudyTrack AI - Database Package
"""

from .adapter import (
    get_db,
    init_db,
    is_postgres,
    get_pg_pool,
    get_database_url,
    PostgresConnectionWrapper,
    PostgresCursorWrapper
)

__all__ = [
    'get_db',
    'init_db',
    'is_postgres',
    'get_pg_pool',
    'get_database_url',
    'PostgresConnectionWrapper',
    'PostgresCursorWrapper'
]

