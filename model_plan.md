Got it — I’ll make the column names generic (no assumptions about your exact schema) and add an explicit instruction in the plan telling the implementer to map them to the actual database fields before coding.

Here’s the updated Markdown spec with that adjustment.

⸻


# Analog-Based Well Production Forecaster — Build Plan

## Overview
We are building a **learned analog selector** to forecast months 1–9 oil/gas production for a proposed well.  
Instead of directly predicting production from features, the model **selects a set of historical wells (analogs)** whose production curves, when averaged, closely match the target well’s expected performance.

This approach captures **hidden operator practices** and local reservoir behavior that are **not visible in public well headers**.

---

## Step 1 — Build Early-Production Table

**Goal:** One row per well with metadata and month-1..9 production arrays.

**Required inputs** (generic names — map these to actual DB column names before implementation):
- `WELL_ID` — unique well identifier (API number or internal key)
- `LAT` / `LON` — surface location
- `FIRST_PROD_DATE` — date of first production
- `FORMATION` — producing formation/zone
- `OPERATOR` — operator name or ID
- `LATERAL_LENGTH` — completed lateral length (ft)
- `PROPPANT_PER_FT` — proppant loading (lb/ft)
- `FLUID_PER_FT` — fluid loading (gal/ft)
- `OIL_MO_PROD` — monthly oil production
- `GAS_MO_PROD` — monthly gas production

**Actions:**
1. Align production to **first_prod_date** → month index m = 1..9.
2. Build arrays:
   - `oil_m1_9`: float[9], oil rates for months 1..9
   - `gas_m1_9`: float[9], gas rates for months 1..9
3. Optional:
   - Normalized arrays (per 1,000 ft lateral)
   - `log1p` transform versions for model stability
4. Add spatial `geom` column (Point) and index for fast distance queries.

**Output:** `early_rates` table:
```text
WELL_ID | FIRST_PROD_DATE | FORMATION | OPERATOR | LATERAL_LENGTH | PROPPANT_PER_FT | FLUID_PER_FT | LAT | LON | OIL_M1_9 | GAS_M1_9 | ...


⸻

Step 2 — Precompute Candidate Pools

Goal: For each target well, store eligible analog wells with coarse filters.

Filters (adjust as needed):
	•	Distance ≤ 10–15 mi
	•	Same (or mapped equivalent) formation
	•	Lateral length within ±20%
	•	First-prod date ≤ target’s first-prod date  (prevents future leakage)
	•	Drop wells with >2 zero months in 1–9

Actions:
	•	Self-join early_rates to itself with filters
	•	Store many rows per target well, each representing a candidate analog

Fields to store:
	•	TARGET_WELL_ID
	•	CANDIDATE_WELL_ID
	•	DISTANCE_MI
	•	LENGTH_RATIO and DELTA_LENGTH
	•	FORMATION_MATCH (bool)
	•	SAME_OPERATOR (bool)
	•	(optional) VINTAGE_GAP_YEARS

Output: analog_candidates table

⸻

Step 3 — Build Curve Embeddings

Goal: Represent each well’s month-1..9 curve as a compact vector for shape similarity.

Method:
	1.	Take log1p(OIL_M1_9) (and GAS later)
	2.	Z-score by month across wells
	3.	Fit PCA(3–4 components) or a tiny autoencoder (9→3→9) on all wells
	4.	Store:
	•	Z_OIL = float[3 or 4]
	•	(later) Z_GAS = float[3 or 4]

Output:
	•	curve_latent table keyed by WELL_ID
	•	Saved PCA/autoencoder object for inference

⸻

Step 4 — Baseline Picker (No ML)

Purpose: Quick, explainable analog-based forecast to serve as a benchmark.

Algorithm:
	1.	From candidate pool, compute score per candidate:

score = cosine_similarity(Z_TARGET, Z_CAND)
      - λ_d * DISTANCE_MI
      - λ_L * |log(LENGTH_RATIO)|

Defaults: λ_d=0.02, λ_L=0.5

	2.	Optional warp for design differences:

s_len = exp(κ_L * log(LENGTH_RATIO))
s_ppf = exp(κ_P * log(PPF_RATIO))
clamp ratios to [0.7, 1.3]
warped_curve = s_len * s_ppf * candidate_curve

Defaults: κ_L=0.6, κ_P=0.2

	3.	Sort candidates by score
	4.	Take Top-K (e.g., K=20), max 3 per operator
	5.	Uniform average warped curves monthwise
	6.	Optional bootstrap of Top-M for P10/P90

Output: P50 (and optional P10/P90) month-1..9 forecast + analog list

⸻

Step 5 — Assemble Training Features

For each (TARGET_WELL_ID, CANDIDATE_WELL_ID) pair:

Feature vector:
	•	Target metadata: FORMATION, OPERATOR, LATERAL_LENGTH, PROPPANT_PER_FT, vintage bucket
	•	Candidate metadata: same fields
	•	Deltas:
	•	DISTANCE_MI
	•	log(LENGTH_RATIO)
	•	log(PPF_RATIO)
	•	VINTAGE_GAP_YEARS
	•	SAME_OPERATOR (bool)
	•	SAME_FORMATION (bool)
	•	Candidate latent vector: Z_OIL (and later Z_GAS)

Output: Training dataset grouped by TARGET_WELL_ID (variable-length sets).

⸻

Step 6 — Train Learned Analog Ranker (LightGBM LambdaRank)

Model: LightGBM with LambdaRank objective (learning-to-rank)

Training Data Structure:
	•	Each instance: (target_well, candidate_well) pair
	•	Features: target_features ⊕ candidate_features ⊕ deltas
	•	Labels: relevance score (higher = better analog for this target)
	•	Groups: one group per target well (rank candidates within groups)

Label Creation (per candidate):
	1.	Warp candidate curve to target design (bounded scalers for length/ppf)
	2.	Compute revenue-weighted error vs target:
		E_ij = Σ(t=1..9) w_t * |target_t - warped_candidate_t|
		where w_t = PRICE / (1 + r)^t
	3.	Convert to graded labels (within each target's pool):
		•	Top 10% lowest error → label 3
		•	Next 20% → label 2
		•	Next 30% → label 1
		•	Rest → label 0

Key Parameters:
	•	objective: lambdarank
	•	metric: ndcg, map
	•	lambdarank_truncation_level: 20 (focus on top ranks)
	•	label_gain: [0, 1, 3, 7] (spacing for graded labels)
	•	Standard GBM params: num_leaves=63, learning_rate=0.05, etc.

⸻

Step 7 — Inference Pipeline

Inputs: New well:
	•	Location (LAT/LON)
	•	FORMATION
	•	LATERAL_LENGTH, PROPPANT_PER_FT
	•	Planned completion date

Process:
	1.	Build candidate pool (Step 2)
	2.	Compute features for (target, candidate)
	3.	Score with sθ
	4.	Select analogs:
	•	Fixed K (e.g., 12) OR
	•	Adaptive K until cumulative weight ≥ 0.8 (with caps)
	5.	Warp curves, average → P50 m1..m9
	6.	Bootstrap for P10/P90

Output: P10/P50/P90 month-1..9 + analog list

⸻

Step 8 — (Optional) Meta-Corrector

Train a small GBM to predict residuals and nudge analog average toward target.

⸻

Step 9 — Test Harness

Simple CLI/notebook to compare:
	•	Baseline picker
	•	Learned model
Plots curves and lists analog wells used.

⸻

Implementation Notes
	1.	Column Mapping
	•	Before coding, map all generic names (WELL_ID, LAT, etc.) to actual database fields in your schema.
	•	Maintain a config file or dictionary for this mapping so code remains database-agnostic.
	2.	No Leakage Guarantee
	•	Always enforce FIRST_PROD_DATE(candidate) ≤ FIRST_PROD_DATE(target) in candidate pool generation.
	3.	Scalability
	•	Precompute and cache candidate pools and latent vectors; don’t rebuild on every inference.
	4.	Interpretability
	•	Always return analog well IDs, weights, and distances along with forecasts.

⸻


---

This way, the **AI agent** will know that the names in the plan are placeholders and that step zero is to **map them to your actual well header and production schema** before writing queries.  

If you want, I can also give you a **one-page data flow diagram** so the agent understands exactly how data moves from DB → feature tables → model → forecast. Would you like me to do that next?