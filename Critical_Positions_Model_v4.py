import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import pickle
import os
from openai import OpenAI
import numpy as np

# --- Configuration Block --- ### REFACTORED ###
# All model parameters are defined here for easy tuning.

CONFIG = {
    'input_file': 'Successors_Sample.xlsx',
    'output_file': 'Successors_Sample_v4.xlsx',
    'embedding_model': 'text-embedding-3-large',
    'embedding_size': 3072,

    # Define text columns and their PCA components
    'text_features': {
        'Cargo': {'pca_components': 2, 'pickle': 'embeddings/embeddings_cargo_unab.pkl'},
        'Departamento': {'pca_components': 2, 'pickle': 'embeddings/embeddings_departamento_unab.pkl'},
        'Mission': {'pca_components': 5, 'pickle': 'embeddings/embeddings_mision_unab.pkl'},
        'Tasks': {'pca_components': 5, 'pickle': 'embeddings/embeddings_accion_unab.pkl'},
        'Results': {'pca_components': 5, 'pickle': 'embeddings/embeddings_resultado_unab.pkl'},
    },

    ### NEW ###
    # Define categorical columns and their ordinal mapping
    'categorical_features': {
        'mapping': {'Alto': 3, 'Medio': 2, 'Bajo': 1},
        'columns': [
            'Relacionamiento político',
            'Nível técnico',
            'Impacto en la estrategia',
            'Nivel de Rotación',
            'Dificultad de Reemplazo'
        ]
    },

    # Define numeric columns (some will be created from categoricals)
    'numeric_features': {
        'columns': [
            'HayGroup',
            'Nivel de Rotación',
            'Dificultad de Reemplazo',
            'Relacionamiento político', # New mapped column
            'Nível técnico',             # New mapped column
            'Impacto en la estrategia'   # New mapped column
        ],
        'column_to_invert': 'Nivel de Rotación',
        # Weights for the initial 'Weighted Composite Score' (must sum to 1.0)
        'weights': {
            'HayGroup': 0.30,
            'Nivel de Rotación': 0.10,
            'Dificultad de Reemplazo': 0.15,
            'Relacionamiento político': 0.15,
            'Nível técnico': 0.15,
            'Impacto en la estrategia': 0.15
        }
    },

    # Weights for the final score (combining numeric score + all text features)
    # This also must sum to 1.0
    'final_score_weights': {
        'numeric_composite': 0.50, # The score from all numeric/categorical features
        'text_features': {
            'Cargo': 0.05,
            'Departamento': 0.05,
            'Mission': 0.15,
            'Tasks': 0.125,
            'Results': 0.125
        }
    },

    # Thresholds for categorization (e.g., top 20% are Critical)
'categorization_quantiles': {
    'Critical': 0.80,
    'Important': 0.50
}
}
# --- End of Configuration Block ---

WEIGHTS_FILE = 'optimized_weights.json'


def apply_optimized_weights() -> None:
    """Override configuration with weights from ``WEIGHTS_FILE`` if present."""
    if not os.path.exists(WEIGHTS_FILE):
        return
    with open(WEIGHTS_FILE, 'r') as f:
        data = json.load(f)

    num_weights = data.get('numeric_weights')
    if num_weights:
        CONFIG['numeric_features']['weights'] = num_weights

    fin_weights = data.get('final_score_weights')
    if fin_weights:
        CONFIG['final_score_weights']['numeric_composite'] = fin_weights.get(
            'numeric_composite',
            CONFIG['final_score_weights']['numeric_composite'],
        )
        for key in CONFIG['final_score_weights']['text_features'].keys():
            if key in fin_weights:
                CONFIG['final_score_weights']['text_features'][key] = fin_weights[key]

    crit_q = data.get('critical_quantile')
    imp_q = data.get('important_quantile')
    if crit_q is not None and imp_q is not None:
        CONFIG['categorization_quantiles']['Critical'] = crit_q
        CONFIG['categorization_quantiles']['Important'] = imp_q


apply_optimized_weights()

client = OpenAI()

def get_embedding(text, model, size):
    """Helper function to get the embedding from OpenAI."""
    if pd.isna(text) or text.strip() == '':
        return [0] * size
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding

def load_or_generate_embeddings(pickle_file, text_series, model, size):
    """Loads or generates embeddings for a series of texts."""
    text_series = text_series.fillna('').astype(str)
    if os.path.exists(pickle_file):
        print(f'Loading embeddings from {pickle_file}')
        with open(pickle_file, 'rb') as f:
            embeddings_dict = pickle.load(f)
    else:
        embeddings_dict = {}

    new_texts = [text for text in text_series.unique() if text not in embeddings_dict]

    if new_texts:
        print(f'Generating embeddings for {len(new_texts)} new texts...')
        for text in new_texts:
            embeddings_dict[text] = get_embedding(text, model, size)
        with open(pickle_file, 'wb') as f:
            pickle.dump(embeddings_dict, f)
    else:
        print(f'All embeddings from {pickle_file} are cached.')

    return text_series.map(embeddings_dict)


# --- Main Workflow ---

# 1. Read Data
print("1. Reading input data...")
sheet_data = pd.read_excel(CONFIG['input_file'])

# 2. Process Categorical Features ### NEW ###
print("\n2. Mapping categorical features to numeric values...")
cat_map = CONFIG['categorical_features']['mapping']
for col in CONFIG['categorical_features']['columns']:
    sheet_data[col] = sheet_data[col].map(cat_map).fillna(0) # Fill missing with 0
    print(f"  - Mapped column: '{col}'")

# 3. Process Numeric Features
print("\n3. Processing numeric features...")
numeric_cols = CONFIG['numeric_features']['columns']
sheet_data[numeric_cols] = sheet_data[numeric_cols].fillna(0)

# Normalize all numeric columns
scaler = MinMaxScaler()
normalized_numeric_data = scaler.fit_transform(sheet_data[numeric_cols])
normalized_numeric_df = pd.DataFrame(normalized_numeric_data, columns=numeric_cols)

# Invert 'Nivel de Rotación' (high turnover is bad, so it gets a low score)
col_to_invert = CONFIG['numeric_features']['column_to_invert']
if col_to_invert in normalized_numeric_df.columns:
    print(f"  - Inverting '{col_to_invert}'")
    normalized_numeric_df[col_to_invert] = 1 - normalized_numeric_df[col_to_invert]

# Calculate weighted composite score for numeric features
print("  - Calculating weighted composite numeric score...")
numeric_weights = CONFIG['numeric_features']['weights']
# Ensure weights are in the correct order
ordered_weights = np.array([numeric_weights[col] for col in numeric_cols])
sheet_data['Weighted Composite Score'] = normalized_numeric_df.dot(ordered_weights)


# 4. Process Text Features
print("\n4. Processing text features (Embeddings + PCA)...")
all_pca_features = []
final_score_weights_list = [CONFIG['final_score_weights']['numeric_composite']]

for col_name, params in CONFIG['text_features'].items():
    print(f"  - Processing '{col_name}'...")
    # Generate/load embeddings
    embeddings = load_or_generate_embeddings(
        params['pickle'],
        sheet_data[col_name],
        CONFIG['embedding_model'],
        CONFIG['embedding_size']
    )
    embeddings_df = pd.DataFrame(embeddings.tolist())

    # Normalize and apply PCA
    embeddings_normalized = MinMaxScaler().fit_transform(embeddings_df)
    n_components = params['pca_components']
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(embeddings_normalized)
    all_pca_features.append(pca_features)

    # Add distributed weights for this feature to the final list
    feature_weight = CONFIG['final_score_weights']['text_features'][col_name]
    final_score_weights_list.extend([feature_weight / n_components] * n_components)

# 5. Combine All Features for Final Score
print("\n5. Combining all features for final score calculation...")
# Stack the numeric score with all the PCA components from text fields
combined_features = np.hstack(
    [sheet_data['Weighted Composite Score'].values.reshape(-1, 1)] + all_pca_features
)

# Normalize the final combined feature set
combined_features_normalized = MinMaxScaler().fit_transform(combined_features)

# Calculate final score using the pre-defined weights
final_score_weights = np.array(final_score_weights_list)
sheet_data['Final Composite Score'] = combined_features_normalized.dot(final_score_weights)

# 6. Categorize Positions
print("6. Categorizing positions...")
critical_q = CONFIG['categorization_quantiles']['Critical']
important_q = CONFIG['categorization_quantiles']['Important']
critical_threshold = sheet_data['Final Composite Score'].quantile(critical_q)
important_threshold = sheet_data['Final Composite Score'].quantile(important_q)

def categorize_position(score):
    if score >= critical_threshold:
        return 'Critical'
    elif score >= important_threshold:
        return 'Important'
    else:
        return 'Moderate'

sheet_data['Category'] = sheet_data['Final Composite Score'].apply(categorize_position)
print(f"  - Critical threshold (top {100-critical_q*100}%): {critical_threshold:.4f}")
print(f"  - Important threshold (top {100-important_q*100}%): {important_threshold:.4f}")

# 7. Export Results
print("\n7. Exporting results...")
final_df = sheet_data.sort_values(by='Final Composite Score', ascending=False)
final_df.to_excel(CONFIG['output_file'], index=False)

print(f"\n✅ Success! Model run complete. Results saved to '{CONFIG['output_file']}'")