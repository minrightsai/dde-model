"""
Step 3: Build Curve Embeddings
Generate compact vector representations of production curves using PCA
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from datetime import datetime
from typing import Tuple, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.db_connector import DatabaseConnector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CurveEmbeddings:
    """Generate and store production curve embeddings"""
    
    def __init__(self, n_components: int = 4):
        """
        Initialize embeddings generator
        
        Args:
            n_components: Number of PCA components (default 4)
        """
        self.n_components = n_components
        self.db = DatabaseConnector()
        self.pca_oil = None
        self.pca_gas = None
        self.scaler_oil = None
        self.scaler_gas = None
        self.oil_mean = None
        self.oil_std = None
        self.gas_mean = None
        self.gas_std = None
        
    def load_production_data(self) -> pd.DataFrame:
        """Load early production data from database"""
        query = """
        SELECT 
            well_id,
            oil_m1_9,
            gas_m1_9,
            cum_oil_m1_9,
            cum_gas_m1_9
        FROM data.early_rates
        WHERE oil_m1_9 IS NOT NULL
        ORDER BY well_id
        """
        
        logger.info("Loading production data...")
        with self.db.get_connection() as conn:
            df = pd.read_sql(query, conn)
        logger.info(f"Loaded {len(df)} wells with production data")
        
        return df
    
    def prepare_curves(self, production_arrays: np.ndarray, 
                      min_nonzero_months: int = 7) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare production curves for embedding
        
        Args:
            production_arrays: Array of shape (n_wells, 9) with monthly production
            min_nonzero_months: Minimum non-zero months required (default 7)
            
        Returns:
            Tuple of (transformed_curves, valid_indices)
        """
        valid_indices = []
        transformed_curves = []
        
        for i, curve in enumerate(production_arrays):
            nonzero_count = np.sum(curve > 0)
            
            if nonzero_count >= min_nonzero_months:
                log_curve = np.log1p(curve)
                transformed_curves.append(log_curve)
                valid_indices.append(i)
        
        transformed_curves = np.array(transformed_curves)
        valid_indices = np.array(valid_indices)
        
        logger.info(f"Valid curves: {len(valid_indices)} / {len(production_arrays)}")
        
        return transformed_curves, valid_indices
    
    def fit_pca(self, curves: np.ndarray) -> Tuple[PCA, StandardScaler, np.ndarray]:
        """
        Fit PCA on production curves
        
        Args:
            curves: Array of log-transformed curves
            
        Returns:
            Tuple of (pca_model, scaler, embeddings)
        """
        scaler = StandardScaler()
        scaled_curves = scaler.fit_transform(curves)
        
        pca = PCA(n_components=self.n_components)
        embeddings = pca.fit_transform(scaled_curves)
        
        explained_var = pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)
        
        logger.info(f"PCA explained variance by component: {explained_var}")
        logger.info(f"Cumulative variance: {cumulative_var}")
        
        return pca, scaler, embeddings
    
    def create_embeddings_table(self):
        """Create curve_embeddings table if it doesn't exist"""
        create_table_query = """
        CREATE TABLE IF NOT EXISTS data.curve_embeddings (
            well_id TEXT PRIMARY KEY,
            oil_embedding FLOAT[],
            gas_embedding FLOAT[],
            oil_explained_var FLOAT,
            gas_explained_var FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_curve_embeddings_well 
        ON data.curve_embeddings(well_id);
        """
        
        with self.db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(create_table_query)
                conn.commit()
        
        logger.info("Created curve_embeddings table")
    
    def store_embeddings(self, df: pd.DataFrame, oil_embeddings: np.ndarray, 
                        gas_embeddings: np.ndarray, oil_var: float, gas_var: float):
        """Store embeddings in database"""
        from psycopg2.extras import execute_batch
        
        insert_query = """
        INSERT INTO data.curve_embeddings (
            well_id, oil_embedding, gas_embedding, 
            oil_explained_var, gas_explained_var
        ) VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (well_id) DO UPDATE SET
            oil_embedding = EXCLUDED.oil_embedding,
            gas_embedding = EXCLUDED.gas_embedding,
            oil_explained_var = EXCLUDED.oil_explained_var,
            gas_explained_var = EXCLUDED.gas_explained_var,
            created_at = CURRENT_TIMESTAMP
        """
        
        records = []
        for i, (_, row) in enumerate(df.iterrows()):
            records.append((
                row['well_id'],
                oil_embeddings[i].tolist(),
                gas_embeddings[i].tolist(),
                float(oil_var),
                float(gas_var)
            ))
        
        with self.db.get_connection() as conn:
            with conn.cursor() as cursor:
                execute_batch(cursor, insert_query, records, page_size=100)
                conn.commit()
        
        logger.info(f"Stored {len(records)} embeddings in database")
    
    def save_models(self, output_dir: str = 'models/embeddings'):
        """Save PCA models and scalers to disk"""
        os.makedirs(output_dir, exist_ok=True)
        
        joblib.dump(self.pca_oil, f'{output_dir}/pca_oil.pkl')
        joblib.dump(self.pca_gas, f'{output_dir}/pca_gas.pkl')
        joblib.dump(self.scaler_oil, f'{output_dir}/scaler_oil.pkl')
        joblib.dump(self.scaler_gas, f'{output_dir}/scaler_gas.pkl')
        
        np.save(f'{output_dir}/oil_mean.npy', self.oil_mean)
        np.save(f'{output_dir}/oil_std.npy', self.oil_std)
        np.save(f'{output_dir}/gas_mean.npy', self.gas_mean)
        np.save(f'{output_dir}/gas_std.npy', self.gas_std)
        
        logger.info(f"Saved models to {output_dir}")
    
    def run(self):
        """Main execution method"""
        logger.info("Starting curve embedding generation...")
        
        df = self.load_production_data()
        
        # Convert PostgreSQL arrays to numpy arrays (handle Decimal type)
        oil_arrays = np.array([[float(val) for val in arr] for arr in df['oil_m1_9']])
        gas_arrays = np.array([[float(val) for val in arr] for arr in df['gas_m1_9']])
        
        logger.info("Processing oil curves...")
        oil_curves, oil_valid = self.prepare_curves(oil_arrays)
        
        if len(oil_curves) > 0:
            self.oil_mean = np.mean(oil_curves, axis=0)
            self.oil_std = np.std(oil_curves, axis=0)
            
            self.pca_oil, self.scaler_oil, oil_embeddings = self.fit_pca(oil_curves)
            oil_var = np.sum(self.pca_oil.explained_variance_ratio_)
        else:
            logger.warning("No valid oil curves found")
            return
        
        logger.info("Processing gas curves...")
        gas_curves, gas_valid = self.prepare_curves(gas_arrays)
        
        if len(gas_curves) > 0:
            self.gas_mean = np.mean(gas_curves, axis=0)
            self.gas_std = np.std(gas_curves, axis=0)
            
            self.pca_gas, self.scaler_gas, gas_embeddings = self.fit_pca(gas_curves)
            gas_var = np.sum(self.pca_gas.explained_variance_ratio_)
        else:
            logger.warning("No valid gas curves found")
            gas_embeddings = np.zeros((len(oil_embeddings), self.n_components))
            gas_var = 0.0
        
        valid_df = df.iloc[oil_valid].reset_index(drop=True)
        
        self.create_embeddings_table()
        self.store_embeddings(valid_df, oil_embeddings, gas_embeddings, oil_var, gas_var)
        self.save_models()
        
        logger.info("✅ Curve embedding generation complete!")
        logger.info(f"Generated embeddings for {len(valid_df)} wells")
        logger.info(f"Oil variance explained: {oil_var:.2%}")
        logger.info(f"Gas variance explained: {gas_var:.2%}")
        
        self.print_summary_stats(oil_embeddings, gas_embeddings)
    
    def print_summary_stats(self, oil_embeddings: np.ndarray, gas_embeddings: np.ndarray):
        """Print summary statistics of embeddings"""
        print("\n" + "="*60)
        print("EMBEDDING SUMMARY STATISTICS")
        print("="*60)
        
        print(f"\nOil Embeddings (n={len(oil_embeddings)}):")
        for i in range(self.n_components):
            comp = oil_embeddings[:, i]
            print(f"  Component {i+1}: mean={comp.mean():.3f}, std={comp.std():.3f}, "
                  f"min={comp.min():.3f}, max={comp.max():.3f}")
        
        print(f"\nGas Embeddings (n={len(gas_embeddings)}):")
        for i in range(self.n_components):
            comp = gas_embeddings[:, i]
            print(f"  Component {i+1}: mean={comp.mean():.3f}, std={comp.std():.3f}, "
                  f"min={comp.min():.3f}, max={comp.max():.3f}")
        
        print("\nTop contributing months to PC1 (Oil):")
        if self.pca_oil is not None:
            pc1_weights = np.abs(self.pca_oil.components_[0])
            top_months = np.argsort(pc1_weights)[::-1][:3]
            for month in top_months:
                print(f"  Month {month+1}: weight={pc1_weights[month]:.3f}")
        
        print("="*60)


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate production curve embeddings')
    parser.add_argument('--components', type=int, default=4,
                       help='Number of PCA components (default: 4)')
    args = parser.parse_args()
    
    embeddings = CurveEmbeddings(n_components=args.components)
    embeddings.run()


if __name__ == "__main__":
    main()