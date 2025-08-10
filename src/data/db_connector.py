"""
Database connection manager for DJ Basin production data
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import yaml
from pathlib import Path
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

class DatabaseConnector:
    """Manages database connections and queries"""
    
    def __init__(self, config_path='config/database.yaml'):
        """Initialize with database configuration"""
        self.config = self._load_config(config_path)
        self.connection_params = self.config['database']
        self.column_mapping = self.config['column_mapping']
        self.filters = self.config['filters']
        self.tables = self.config['tables']
    
    def _load_config(self, config_path):
        """Load database configuration from YAML"""
        config_file = Path(config_path)
        if not config_file.exists():
            # Try relative to project root
            config_file = Path(__file__).parent.parent.parent / config_path
        
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = psycopg2.connect(**self.connection_params)
            yield conn
        finally:
            if conn:
                conn.close()
    
    def execute_query(self, query, params=None, fetch=True):
        """Execute a query and optionally fetch results"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                if fetch:
                    return cur.fetchall()
                conn.commit()
    
    def get_dj_basin_filter(self):
        """Return SQL WHERE clause for DJ Basin wells"""
        return f"""
            {self.column_mapping['STATE']} = '{self.filters['state_code']}'
            AND {self.column_mapping['ORIENTATION']} = '{self.filters['orientation']}'
            AND {self.column_mapping['SPUD_DATE']} >= '{self.filters['min_spud_date']}'
            AND {self.column_mapping['BASIN']} = '{self.filters['basin_pattern']}'
        """
    
    def check_tables_exist(self):
        """Check if required tables exist"""
        # Extract table names without schema prefix
        wells_table = self.tables['wells'].split('.')[-1]
        prod_table = self.tables['production'].split('.')[-1]
        
        query = f"""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'data' 
        AND table_name IN ('{wells_table}', '{prod_table}')
        """
        results = self.execute_query(query)
        tables = [r['table_name'] for r in results]
        
        missing = []
        if wells_table not in tables:
            missing.append(wells_table)
        if prod_table not in tables:
            missing.append(prod_table)
        
        if missing:
            logger.warning(f"Missing tables: {missing}")
            return False
        return True
    
    def get_well_count(self):
        """Get count of DJ Basin wells matching criteria"""
        query = f"""
        SELECT COUNT(DISTINCT {self.column_mapping['WELL_ID']}) as count
        FROM {self.tables['wells']}
        WHERE {self.get_dj_basin_filter()}
        """
        result = self.execute_query(query)
        return result[0]['count'] if result else 0