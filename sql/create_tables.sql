-- Create data schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS data;

-- Create wells table
DROP TABLE IF EXISTS data.ed_wells;
CREATE TABLE data.ed_wells (
    well_id VARCHAR(50) PRIMARY KEY,
    universal_doc_no VARCHAR(100),
    lat DOUBLE PRECISION,
    lon DOUBLE PRECISION,
    operator VARCHAR(255),
    formation VARCHAR(100),
    well_name VARCHAR(255),
    state VARCHAR(10),
    county VARCHAR(100),
    basin VARCHAR(100),
    first_prod_date TIMESTAMP,
    lateral_length INTEGER,
    proppant_lbs BIGINT,
    fluid_gals DOUBLE PRECISION,
    measured_depth INTEGER,
    tvd INTEGER,
    drill_type VARCHAR(50),
    well_type VARCHAR(50),
    well_status VARCHAR(50),
    spud_date TIMESTAMP,
    permit_date TIMESTAMP,
    proppant_per_ft DOUBLE PRECISION,
    fluid_per_ft DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for common queries
CREATE INDEX idx_ed_wells_location ON data.ed_wells USING GIST (ST_Point(lon, lat));
CREATE INDEX idx_ed_wells_operator ON data.ed_wells (operator);
CREATE INDEX idx_ed_wells_formation ON data.ed_wells (formation);
CREATE INDEX idx_ed_wells_first_prod_date ON data.ed_wells (first_prod_date);
CREATE INDEX idx_ed_wells_state ON data.ed_wells (state);
CREATE INDEX idx_ed_wells_basin ON data.ed_wells (basin);

-- Create production table (for future use)
DROP TABLE IF EXISTS data.ed_prod;
CREATE TABLE data.ed_prod (
    id SERIAL PRIMARY KEY,
    well_id VARCHAR(50) REFERENCES data.ed_wells(well_id),
    prod_date DATE,
    oil_bbls DOUBLE PRECISION,
    gas_mcf DOUBLE PRECISION,
    water_bbls DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_ed_prod_well_id ON data.ed_prod (well_id);
CREATE INDEX idx_ed_prod_date ON data.ed_prod (prod_date);
CREATE INDEX idx_ed_prod_well_date ON data.ed_prod (well_id, prod_date);

-- Create rigs table (for future use)  
DROP TABLE IF EXISTS data.ed_rigs;
CREATE TABLE data.ed_rigs (
    rig_id VARCHAR(50) PRIMARY KEY,
    rig_name VARCHAR(255),
    operator VARCHAR(255),
    lat DOUBLE PRECISION,
    lon DOUBLE PRECISION,
    status VARCHAR(50),
    well_id VARCHAR(50),
    spud_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_ed_rigs_location ON data.ed_rigs USING GIST (ST_Point(lon, lat));
CREATE INDEX idx_ed_rigs_operator ON data.ed_rigs (operator);
CREATE INDEX idx_ed_rigs_well_id ON data.ed_rigs (well_id);