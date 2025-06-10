import json
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import recall_score

"""Random search optimizer to maximize recall for the Critical Positions model."""

CONFIG = {
    'input_file': 'Successors_Sample.xlsx',
    'embedding_size': 3072,
    'text_features': {
        'Cargo': {'pca_components': 2, 'pickle': 'embeddings/embeddings_cargo_unab.pkl'},
        'Departamento': {'pca_components': 2, 'pickle': 'embeddings/embeddings_departamento_unab.pkl'},
        'Mission': {'pca_components': 5, 'pickle': 'embeddings/embeddings_mision_unab.pkl'},
        'Tasks': {'pca_components': 5, 'pickle': 'embeddings/embeddings_accion_unab.pkl'},
        'Results': {'pca_components': 5, 'pickle': 'embeddings/embeddings_resultado_unab.pkl'},
    },
    'categorical_features': {
        'mapping': {'Alto': 3, 'Medio': 2, 'Bajo': 1},
        'columns': [
            'Relacionamiento político',
            'Nível técnico',
            'Impacto en la estrategia',
            'Nivel de Rotación',
            'Dificultad de Reemplazo',
        ],
    },
    'numeric_cols': [
        'HayGroup',
        'Nivel de Rotación',
        'Dificultad de Reemplazo',
        'Relacionamiento político',
        'Nível técnico',
        'Impacto en la estrategia',
    ],
    'numeric_col_to_invert': 'Nivel de Rotación',
    'categorization_quantiles': {'Critical': 0.80, 'Important': 0.50},
}

ITERATIONS = 300
MIN_W = 0.05
OUTPUT_FILE = 'optimized_weights.json'
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def load_embeddings(pickle_file: str, texts: pd.Series) -> list:
    """Return list of embedding vectors for the provided texts."""
    if not os.path.exists(pickle_file):
        raise FileNotFoundError(f"Missing cached embeddings: {pickle_file}")
    with open(pickle_file, 'rb') as f:
        emb_dict = pickle.load(f)
    return [emb_dict.get(t, np.zeros(CONFIG['embedding_size'])) for t in texts.fillna('').astype(str)]


def preprocess_data():
    """Prepare numeric matrix, text PCA matrices, label mask and labels."""
    df = pd.read_excel(CONFIG['input_file'])

    for col in CONFIG['categorical_features']['columns']:
        df[col] = df[col].map(CONFIG['categorical_features']['mapping']).fillna(0)

    numeric_mat = df[CONFIG['numeric_cols']].fillna(0).to_numpy()
    numeric_mat = MinMaxScaler().fit_transform(numeric_mat)
    inv_idx = CONFIG['numeric_cols'].index(CONFIG['numeric_col_to_invert'])
    numeric_mat[:, inv_idx] = 1 - numeric_mat[:, inv_idx]

    text_mats = []
    for col, params in CONFIG['text_features'].items():
        embeddings = load_embeddings(params['pickle'], df[col])
        emb_mat = MinMaxScaler().fit_transform(np.vstack(embeddings))
        pca = PCA(n_components=params['pca_components'])
        pca_features = pca.fit_transform(emb_mat)
        text_mats.append(MinMaxScaler().fit_transform(pca_features))

    mask = df['Desired_Outcome'].notna().values
    labels = df.loc[mask, 'Desired_Outcome'].values
    return numeric_mat, text_mats, mask, labels


def compute_scores(num_mat: np.ndarray, text_mats: list, w_num: np.ndarray, w_fin: dict) -> np.ndarray:
    """Return final composite score for each row."""
    numeric_comp = num_mat.dot(w_num)
    blocks = [numeric_comp.reshape(-1, 1)] + text_mats
    combined = np.hstack(blocks)
    weights_per_col = [w_fin['numeric_composite']]
    for name, mat in zip(CONFIG['text_features'].keys(), text_mats):
        weights_per_col.extend([w_fin[name] / mat.shape[1]] * mat.shape[1])
    combined_norm = MinMaxScaler().fit_transform(combined)
    return combined_norm.dot(np.array(weights_per_col))


def assign_categories(scores: np.ndarray) -> np.ndarray:
    """Map score vector to category labels."""
    crit_q = CONFIG['categorization_quantiles']['Critical']
    imp_q = CONFIG['categorization_quantiles']['Important']
    crit_thr = np.quantile(scores, crit_q)
    imp_thr = np.quantile(scores, imp_q)
    return np.where(scores >= crit_thr, 'Critical', np.where(scores >= imp_thr, 'Important', 'Moderate'))


def sample_weights(size: int) -> np.ndarray:
    """Generate a random weight vector of length ``size``."""
    vec = np.random.uniform(MIN_W, 1.0, size)
    vec /= vec.sum()
    if (vec < MIN_W).any():
        vec = np.maximum(vec, MIN_W)
        vec /= vec.sum()
    return vec


def optimise() -> None:
    """Run the random search and save the best weights."""
    num_mat, text_mats, mask, labels = preprocess_data()
    best_recall = -1.0
    best = None
    final_keys = ['numeric_composite'] + list(CONFIG['text_features'].keys())

    for _ in range(ITERATIONS):
        w_num = sample_weights(len(CONFIG['numeric_cols']))
        fin_vec = sample_weights(len(final_keys))
        w_fin = dict(zip(final_keys, fin_vec))
        scores = compute_scores(num_mat, text_mats, w_num, w_fin)
        preds = assign_categories(scores)[mask]
        rec = recall_score(labels, preds, labels=['Critical'], average='micro', zero_division=0)
        if rec > best_recall:
            best_recall = rec
            best = {
                'numeric_weights': dict(zip(CONFIG['numeric_cols'], w_num)),
                'final_score_weights': w_fin,
                'critical_quantile': CONFIG['categorization_quantiles']['Critical'],
                'important_quantile': CONFIG['categorization_quantiles']['Important'],
            }

    if best is not None:
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(best, f, indent=4)
        print(f'Best recall: {best_recall:.3f}')
        print(f'Weights saved to {OUTPUT_FILE}')
    else:
        print('No weights generated.')


if __name__ == '__main__':
    optimise()
