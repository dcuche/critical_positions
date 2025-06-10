"""Differential evolution optimisation focused on recall for the Critical class."""

import argparse
import json
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import recall_score
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import differential_evolution, Bounds, LinearConstraint

CONFIG = {
    'input_file': 'Successors_Sample.xlsx',
    'embedding_size': 3072,
    'text_features': {
        'Cargo': {'pca_components': 2, 'pickle': 'embeddings/embeddings_cargo_unab.pkl'},
        'Departamento': {
            'pca_components': 2,
            'pickle': 'embeddings/embeddings_departamento_unab.pkl'
        },
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
            'Dificultad de Reemplazo'
        ]
    },
    'numeric_cols': [
        'HayGroup',
        'Nivel de Rotación',
        'Dificultad de Reemplazo',
        'Relacionamiento político',
        'Nível técnico',
        'Impacto en la estrategia'
    ],
    'numeric_col_to_invert': 'Nivel de Rotación',
    'categorization_quantiles': {'Critical': 0.80, 'Important': 0.50}
}

WEIGHTS_FILE = 'optimized_weights.json'
MIN_W = 0.05
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def load_embeddings(pickle_file: str, texts: pd.Series) -> list:
    """Load cached embeddings and map them to texts."""
    with open(pickle_file, 'rb') as f:
        emb_dict = pickle.load(f)
    return [emb_dict.get(t, np.zeros(CONFIG['embedding_size'])) for t in texts.fillna('').astype(str)]


def preprocess():
    """Return numeric matrix, text matrices, mask of labels and labels."""
    df = pd.read_excel(CONFIG['input_file'])
    for col in CONFIG['categorical_features']['columns']:
        df[col] = df[col].map(CONFIG['categorical_features']['mapping']).fillna(0)
    num_mat = df[CONFIG['numeric_cols']].fillna(0).to_numpy()
    num_mat = MinMaxScaler().fit_transform(num_mat)
    inv_idx = CONFIG['numeric_cols'].index(CONFIG['numeric_col_to_invert'])
    num_mat[:, inv_idx] = 1 - num_mat[:, inv_idx]

    text_mats = []
    for col, params in CONFIG['text_features'].items():
        embs = load_embeddings(params['pickle'], df[col])
        emb_mat = MinMaxScaler().fit_transform(np.vstack(embs))
        pca = PCA(n_components=params['pca_components'])
        pca_mat = pca.fit_transform(emb_mat)
        text_mats.append(MinMaxScaler().fit_transform(pca_mat))

    mask = df['Desired_Outcome'].notna().values
    labels = df.loc[mask, 'Desired_Outcome'].values
    return num_mat, text_mats, mask, labels


def compute_scores(num_mat, text_mats, w_num, w_fin):
    """Compute final composite scores."""
    numeric_comp = num_mat.dot(w_num)
    blocks = [numeric_comp.reshape(-1, 1)] + text_mats
    combined = np.hstack(blocks)
    weights_per_col = [w_fin['numeric_composite']]
    for name, mat in zip(CONFIG['text_features'].keys(), text_mats):
        weights_per_col.extend([w_fin[name] / mat.shape[1]] * mat.shape[1])
    combined_norm = MinMaxScaler().fit_transform(combined)
    return combined_norm.dot(np.array(weights_per_col))


def assign_categories(scores, crit_q, imp_q):
    """Return class labels based on quantiles."""
    crit_thr = np.quantile(scores, crit_q)
    imp_thr = np.quantile(scores, imp_q)
    return np.where(scores >= crit_thr, 'Critical',
                    np.where(scores >= imp_thr, 'Important', 'Moderate'))


def objective(x, num_mat, text_mats, mask, labels):
    """Objective function for differential evolution: 1 - recall."""
    n_num = len(CONFIG['numeric_cols']) - 1
    n_fin = len(CONFIG['text_features']) - 1
    w_num_part = x[:n_num]
    w_fin_part = x[n_num:n_num + n_fin]
    crit_q, imp_q = x[-2:]

    w_num = np.append(w_num_part, 1 - w_num_part.sum())
    w_fin_vals = np.append(w_fin_part, 1 - w_fin_part.sum())
    fin_keys = ['numeric_composite'] + list(CONFIG['text_features'].keys())
    w_fin = dict(zip(fin_keys, w_fin_vals))

    if w_num.min() < MIN_W or w_fin_vals.min() < MIN_W or crit_q <= imp_q:
        return 1.0

    scores = compute_scores(num_mat, text_mats, w_num, w_fin)
    preds = assign_categories(scores, crit_q, imp_q)[mask]
    rec = recall_score(labels, preds, labels=['Critical'], average='micro', zero_division=0)
    return 1 - rec


def optimise(iterations=300):
    """Run optimisation and save best weights to WEIGHTS_FILE."""
    num_mat, text_mats, mask, labels = preprocess()
    bounds = [(MIN_W, 1.0)] * (len(CONFIG['numeric_cols']) - 1 + len(CONFIG['text_features']) - 1) + [
        (0.6, 0.95),
        (0.3, 0.7),
    ]
    cons_num = [1.0] * (len(CONFIG['numeric_cols']) - 1) + [0.0] * (len(CONFIG['text_features']) - 1 + 2)
    cons_fin = [0.0] * (len(CONFIG['numeric_cols']) - 1) + [1.0] * (len(CONFIG['text_features']) - 1) + [0.0, 0.0]
    constraints = [
        LinearConstraint(cons_num, 1.0 - MIN_W, 1.0 - MIN_W),
        LinearConstraint(cons_fin, 1.0 - MIN_W, 1.0 - MIN_W),
    ]
    result = differential_evolution(
        objective,
        bounds=bounds,
        args=(num_mat, text_mats, mask, labels),
        constraints=constraints,
        seed=RANDOM_SEED,
        maxiter=iterations,
        polish=True,
        updating='deferred',
    )

    x = result.x
    n_num = len(CONFIG['numeric_cols']) - 1
    n_fin = len(CONFIG['text_features']) - 1
    w_num = np.append(x[:n_num], 1 - x[:n_num].sum())
    w_fin_vals = np.append(x[n_num:n_num + n_fin], 1 - x[n_num:n_num + n_fin].sum())
    fin_keys = ['numeric_composite'] + list(CONFIG['text_features'].keys())
    w_fin = dict(zip(fin_keys, w_fin_vals))
    crit_q, imp_q = x[-2:]

    best = {
        'numeric_weights': dict(zip(CONFIG['numeric_cols'], w_num)),
        'final_score_weights': w_fin,
        'critical_quantile': float(crit_q),
        'important_quantile': float(imp_q)
    }
    with open(WEIGHTS_FILE, 'w') as f:
        json.dump(best, f, indent=4)
    print('Optimisation complete. Weights saved to', WEIGHTS_FILE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimise recall for critical class.')
    parser.add_argument('--iterations', type=int, default=300, help='DE max iterations')
    args = parser.parse_args()
    optimise(args.iterations)
