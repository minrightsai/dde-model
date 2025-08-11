"""
Basin Configuration System
Centralizes all basin-specific parameters for multi-basin deployment
"""

from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class BasinConfig:
    """
    Configuration manager for different oil & gas basins.
    Each basin has specific parameters for analog selection, formations, and modeling.
    """
    
    CONFIGS = {
        'dj_basin': {
            'name': 'Denver Basin',
            'api_prefixes': ['05'],  # Colorado - NOTE: includes other CO basins too!
            'states': ['CO'],
            'bounds': {
                'lat_min': 39.51,
                'lat_max': 42.19,
                'lon_min': -105.13, 
                'lon_max': -101.99
            },
            'max_distance_miles': 15.0,  # Tight well spacing in DJ
            'lateral_tolerance': 0.2,  # ±20% lateral length
            'min_vintage_year': 2018,
            'peak_month_window': 7,  # Look for peak in first 7 months
            'months_to_analyze': 9,  # 9-month production window
            'formations': [
                'Niobrara', 'Niobrara A', 'Niobrara B', 'Niobrara C',
                'Codell', 'Greenhorn', 'Shannon', 'Sussex', 'Turner',
                'Parkman', 'Terry', 'Hygiene'
            ],
            'warping': {
                'length_coefficient': 0.6,  # κ_length for production scaling
                'proppant_coefficient': 0.2  # κ_proppant for production scaling
            },
            'completion_ranges': {
                'lateral_length': {'min': 4000, 'max': 15000},  # feet
                'proppant_per_ft': {'min': 500, 'max': 3500},  # lbs/ft
                'fluid_per_ft': {'min': 20, 'max': 100}  # gal/ft
            },
            'analog_selection': {
                'min_candidates': 10,
                'max_candidates': 500,
                'top_k_analogs': 20,  # Use top 20 for predictions
                'distance_weight_power': 2.0  # For inverse distance weighting
            },
            'database': {
                'schema': 'data',
                'tables': {
                    'wells': 'model_wells',
                    'production': 'model_prod',
                    'early_rates': 'early_rates',
                    'candidates': 'analog_candidates',
                    'embeddings': 'curve_embeddings'
                }
            }
        },
        
        'bakken': {
            'name': 'Bakken',
            'api_prefixes': ['33', '25'],  # North Dakota, Montana (corrected)
            'states': ['ND', 'MT'],
            'bounds': {
                'lat_min': 46.66,
                'lat_max': 48.99,
                'lon_min': -105.26,
                'lon_max': -102.00
            },
            'max_distance_miles': 20.0,
            'lateral_tolerance': 0.2,
            'min_vintage_year': 2016,
            'peak_month_window': 6,
            'months_to_analyze': 9,
            'formations': [
                'Bakken', 'Middle Bakken', 'Three Forks',
                'Three Forks 1st', 'Three Forks 2nd', 'Three Forks 3rd'
            ],
            'warping': {
                'length_coefficient': 0.55,
                'proppant_coefficient': 0.25
            },
            'completion_ranges': {
                'lateral_length': {'min': 5000, 'max': 12000},
                'proppant_per_ft': {'min': 600, 'max': 2500},
                'fluid_per_ft': {'min': 25, 'max': 70}
            },
            'analog_selection': {
                'min_candidates': 10,
                'max_candidates': 400,
                'top_k_analogs': 20,
                'distance_weight_power': 2.0
            },
            'database': {
                'schema': 'data',
                'tables': {
                    'wells': 'model_wells',
                    'production': 'model_prod',
                    'early_rates': 'early_rates',
                    'candidates': 'analog_candidates',
                    'embeddings': 'curve_embeddings'
                }
            }
        }
    }
    
    def __init__(self, basin_name: str = 'dj_basin'):
        """
        Initialize configuration for a specific basin
        
        Args:
            basin_name: Name of the basin (e.g., 'dj_basin', 'bakken')
        """
        if basin_name not in self.CONFIGS:
            logger.warning(f"Unknown basin '{basin_name}', defaulting to DJ Basin")
            basin_name = 'dj_basin'
        
        self.basin_name = basin_name
        self.config = self.CONFIGS[basin_name]
        logger.info(f"Initialized configuration for {self.config['name']}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key with dot notation support"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_api_filter(self) -> str:
        """Get SQL WHERE clause for API prefix filtering"""
        prefixes = self.config['api_prefixes']
        if not prefixes:
            return "1=1"  # No filter
        
        # Use %% to escape the % for psycopg2
        conditions = [f"well_id LIKE '{prefix}%%'" for prefix in prefixes]
        return f"({' OR '.join(conditions)})"
    
    def get_state_filter(self) -> str:
        """Get SQL WHERE clause for state filtering"""
        states = self.config['states']
        if not states:
            return "1=1"  # No filter
        
        state_list = "', '".join(states)
        return f"state IN ('{state_list}')"
    
    def get_formation_match_query(self, target_formation: str) -> str:
        """Get SQL for formation matching with fuzzy logic"""
        # Exact match first, then handle variations
        return f"""
        CASE 
            WHEN UPPER(formation) = UPPER('{target_formation}') THEN 1.0
            WHEN UPPER(formation) LIKE UPPER('%{target_formation.split()[0]}%') THEN 0.8
            ELSE 0.0
        END
        """
    
    def validate_completion_params(self, lateral_length: float, 
                                  proppant_per_ft: float,
                                  fluid_per_ft: float) -> Dict[str, bool]:
        """Validate if completion parameters are within expected ranges"""
        ranges = self.config['completion_ranges']
        
        return {
            'lateral_length_valid': (
                ranges['lateral_length']['min'] <= lateral_length <= 
                ranges['lateral_length']['max']
            ),
            'proppant_valid': (
                ranges['proppant_per_ft']['min'] <= proppant_per_ft <= 
                ranges['proppant_per_ft']['max']
            ),
            'fluid_valid': (
                ranges['fluid_per_ft']['min'] <= fluid_per_ft <= 
                ranges['fluid_per_ft']['max']
            )
        }
    
    def get_warping_coefficients(self) -> Dict[str, float]:
        """Get production warping coefficients for the basin"""
        return self.config['warping']
    
    def get_analog_params(self) -> Dict[str, Any]:
        """Get analog selection parameters"""
        return self.config['analog_selection']
    
    def get_table_name(self, table_key: str) -> str:
        """Get full table name with schema"""
        schema = self.config['database']['schema']
        table = self.config['database']['tables'].get(table_key)
        
        if not table:
            raise ValueError(f"Unknown table key: {table_key}")
        
        return f"{schema}.{table}"
    
    @classmethod
    def list_basins(cls) -> List[str]:
        """List all available basin configurations"""
        return list(cls.CONFIGS.keys())
    
    @classmethod
    def get_basin_info(cls, basin_name: str) -> Optional[Dict]:
        """Get basic info about a basin"""
        if basin_name not in cls.CONFIGS:
            return None
        
        config = cls.CONFIGS[basin_name]
        return {
            'name': config['name'],
            'states': config['states'],
            'formations': len(config['formations']),
            'max_distance_miles': config['max_distance_miles']
        }
    
    def __repr__(self) -> str:
        return f"BasinConfig('{self.basin_name}': {self.config['name']})"