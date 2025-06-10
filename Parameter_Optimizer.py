#!/usr/bin/env python
# optimise_cjp.py  – searches for weight sets that maximise agreement with Desired_Outcome
# ------------------------------------------------------------------------------

import argparse, os, pickle, random, sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy.optimize import differential_evolution

# ---------- CONFIGURATION (identical to your original, but separated) ----------
CONFIG = {
    'input_file': 'Successors_Sample.xlsx',
    'output_file': 'Successors_Sample_result_v4.xlsx',

    'embedding_model': 'text-embedding-3-large',
    'embedding_size': 3072,

    'text_features': {
        'Cargo':        {'pca_components': 2, 'pickle': 'embeddings/embeddings_cargo_unab.pkl'},
        'Departamento': {'pca_components': 2, 'pickle': 'embeddings/embeddings_departamento_unab.pkl'},
        'Mission':      {'pca_components': 5, 'pickle': 'embeddings/embeddings_mision_unab.pkl'},
        'Tasks':        {'pca_components': 5, 'pickle': 'embeddings/embeddings_accion_unab.pkl'},
        'Results':      {'pca_components': 5, 'pickle': 'embeddings/embeddings_resultado_unab.pkl'},
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

    # Initial (seed) weights – optimisation will adjust them.
    'seed_numeric_weights': {
        'HayGroup': 0.30,
        'Nivel de Rotación': 0.10,
        'Dificultad de Reemplazo': 0.15,
        'Relacionamiento político': 0.15,
        'Nível técnico': 0.15,
        'Impacto en la estrategia': 0.15
    },

    'seed_final_score_weights': {
        'numeric_composite': 0.50,
        'Cargo':        0.05,
        'Departamento': 0.05,
        'Mission':      0.15,
        'Tasks':        0.125,
        'Results':      0.125
    },

    # Quantiles used to assign Critical / Important / Moderate
    'categorization_quantiles': {'Critical': 0.80, 'Important': 0.50}
}

MIN_W = 0.05            # every individual weight must stay >= this value
RANDOM_SEED = 42        # reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---------- HELPERS -----------------------------------------------------------

def load_or_generate_embeddings(pickle_file: str, texts: pd.Series, emb_size: int):
    """
    Loads existing embeddings dictionary if present; otherwise expects them to be
    already cached (fails early if not). We avoid calling the OpenAI API during
    optimisation to keep it lightning fast.
    """
    if not os.path.exists(pickle_file):
        raise FileNotFoundError(
            f"Need cached embeddings file '{pickle_file}'. "
            f"Generate it once with your original script, then run optimisation."
        )
    with open(pickle_file, 'rb') as f:
        emb_dict = pickle.load(f)
    # Map each text (NaNs replaced by '') to embedding vector
    return texts.fillna('').astype(str).map(lambda t: emb_dict.get(t, np.zeros(emb_size))).tolist()

def normalise_minmax(arr2d):
    return MinMaxScaler().fit_transform(arr2d)

def build_feature_matrices(sheet_df: pd.DataFrame):
    """
    Pre-computes every feature once (numeric + all PCA components) and returns
    - numeric_matrix      shape (N, 6)
    - text_feature_mats   ordered list, one np.array per text feature
    - label_mask          boolean array: rows with Desired_Outcome not-na
    - desired_labels      ndarray of strings ('Critical'/'Important'/'Moderate')
    """
    df = sheet_df.copy()

    # 1. Map categoricals to numbers
    for col in CONFIG['categorical_features']['columns']:
        df[col] = df[col].map(CONFIG['categorical_features']['mapping']).fillna(0)

    # 2. Numeric features
    num_cols = CONFIG['numeric_cols']
    df[num_cols] = df[num_cols].fillna(0)
    num_norm = normalise_minmax(df[num_cols])
    # Invert turnover
    inv_col = CONFIG['numeric_col_to_invert']
    if inv_col in num_cols:
        idx = num_cols.index(inv_col)
        num_norm[:, idx] = 1 - num_norm[:, idx]

    # 3. Text features → (embeddings → minmax → PCA → minmax)
    text_mats = []
    for col_name, params in CONFIG['text_features'].items():
        emb_list = load_or_generate_embeddings(
            params['pickle'],
            df[col_name],
            CONFIG['embedding_size']
        )
        emb_mat = normalise_minmax(np.vstack(emb_list))
        pca = PCA(n_components=params['pca_components'], random_state=RANDOM_SEED)
        pca_mat = pca.fit_transform(emb_mat)
        text_mats.append(normalise_minmax(pca_mat))

    # 4. Desired outcome mask & labels
    label_mask = df['Desired_Outcome'].notna().values
    desired = df.loc[label_mask, 'Desired_Outcome'].values

    return num_norm, text_mats, label_mask, desired

# ---------- SCORING -----------------------------------------------------------

def compute_final_scores(num_mat, text_mats, w_numeric, w_final):
    """
    Parameters
    ----------
    num_mat   : (N,6) numeric features normalised & turnover inverted
    text_mats : list of np.array[(N, k_i)]  text PCA components (already min-max)
    w_numeric : length-6 ndarray – must sum to 1
    w_final   : dict with keys
                ['numeric_composite'] + each text feature in CONFIG['text_features']
                and values summing to 1
    Returns
    -------
    final_scores : (N,) ndarray
    """
    # weighted numeric composite
    numeric_comp = num_mat.dot(w_numeric)

    # concatenate all feature blocks
    blocks = [numeric_comp.reshape(-1, 1)]
    for tm in text_mats:
        blocks.append(tm)
    combined = np.hstack(blocks)

    # build per-column weights
    percol_weights = [w_final['numeric_composite']]  # 1 column for numeric comp
    for tf_key, tm in zip(CONFIG['text_features'].keys(), text_mats):
        percol_weights.extend([w_final[tf_key] / tm.shape[1]] * tm.shape[1])

    percol_weights = np.array(percol_weights)
    assert combined.shape[1] == percol_weights.size

    # normalise combined matrix before applying final weights
    combined_norm = normalise_minmax(combined)
    return combined_norm.dot(percol_weights)

def assign_categories(scores):
    q = CONFIG['categorization_quantiles']
    crit_thr = np.quantile(scores, q['Critical'])
    imp_thr  = np.quantile(scores, q['Important'])
    labels = np.where(scores >= crit_thr, 'Critical',
              np.where(scores >= imp_thr,  'Important', 'Moderate'))
    return labels

def accuracy(pred, truth):
    return (pred == truth).mean()

# ---------- OPTIMISATION ------------------------------------------------------

def random_weight_vector(size):
    """Generate random weights ≥ MIN_W and normalised to 1."""
    vec = np.random.uniform(MIN_W, 1.0, size)
    vec /= vec.sum()
    # guarantee min-weight after normalisation
    if (vec < MIN_W).any():
        deficit = np.where(vec < MIN_W)[0]
        surplus = np.where(vec > MIN_W)[0]
        for i in deficit:
            take = MIN_W - vec[i]
            j = random.choice(surplus)
            vec[j] -= take
            vec[i] += take
    return vec

def optimise_random(num_mat, text_mats, label_mask, desired, iterations):
    best_acc, best_w_num, best_w_fin = -1, None, None
    fin_keys = list(CONFIG['seed_final_score_weights'].keys())

    for _ in range(iterations):
        w_num = random_weight_vector(len(CONFIG['numeric_cols']))

        rand_fin = random_weight_vector(len(fin_keys))
        w_fin = dict(zip(fin_keys, rand_fin))

        scores  = compute_final_scores(num_mat, text_mats, w_num, w_fin)
        preds   = assign_categories(scores)[label_mask]
        acc     = accuracy(preds, desired)
        if acc > best_acc:
            best_acc, best_w_num, best_w_fin = acc, w_num.copy(), w_fin.copy()

    return best_acc, best_w_num, best_w_fin

# ----- Differential Evolution (SciPy) -----
def optimise_diffev(num_mat, text_mats, label_mask, desired, iterations):
    """
    Optimises first 5 numeric weights (6th is residual) + 5 final weights
    (numeric_composite and first 4 text features – Last text weight is residual).
    Total variables = 10.
    """
    num_cols = CONFIG['numeric_cols']
    fin_keys = list(CONFIG['seed_final_score_weights'].keys())

    # We'll optimise first (len-1) weights & make the residual = 1 - sum
    def repair_and_split(x):
        # enforce bounds ≥ MIN_W
        x = np.clip(x, MIN_W, 1.0)
        # split
        w_num_part = x[:len(num_cols)-1]
        w_fin_part = x[len(num_cols)-1:]

        w_num = np.append(w_num_part, 1 - w_num_part.sum())
        w_fin = np.append(w_fin_part, 1 - w_fin_part.sum())

        # If residual fell below MIN_W, re-distribute uniformly
        if w_num[-1] < MIN_W or w_fin[-1] < MIN_W:
            w_num = np.maximum(w_num, MIN_W)
            w_fin = np.maximum(w_fin, MIN_W)
            w_num /= w_num.sum()
            w_fin /= w_fin.sum()

        w_fin_dict = dict(zip(fin_keys, w_fin))
        return w_num, w_fin_dict

    def objective(x):
        w_num, w_fin = repair_and_split(x)
        scores = compute_final_scores(num_mat, text_mats, w_num, w_fin)
        preds  = assign_categories(scores)[label_mask]
        return 1 - accuracy(preds, desired)   # minimise (1 - acc)

    bounds = [(MIN_W, 1.0)] * (len(num_cols) - 1 + len(fin_keys) - 1)  # 10 vars

    result = differential_evolution(
        objective,
        bounds,
        maxiter=iterations,
        polish=True,
        seed=RANDOM_SEED,
        updating='deferred'
    )

    best_w_num, best_w_fin = repair_and_split(result.x)
    best_acc = 1 - result.fun
    return best_acc, best_w_num, best_w_fin

# ----- Tiny grid (illustrative, not exhaustive) -----
def optimise_grid(num_mat, text_mats, label_mask, desired, step=0.1):
    num_cols = CONFIG['numeric_cols']
    fin_keys = list(CONFIG['seed_final_score_weights'].keys())
    best_acc, best_w_num, best_w_fin = -1, None, None

    # numeric: vary HayGroup vs rest only, coarse
    for hg in np.arange(MIN_W, 1, step):
        rest = 1 - hg
        w_num = np.array([hg] + [rest/(len(num_cols)-1)]*(len(num_cols)-1))

        # final score: coarse sweep of numeric_composite weight
        for nc_w in np.arange(0.3, 0.7, step):
            rest_f = 1 - nc_w
            # equally split text features
            txt_w = np.full(len(fin_keys)-1, rest_f / (len(fin_keys)-1))
            w_fin = dict(zip(fin_keys, np.concatenate([[nc_w], txt_w])))

            scores = compute_final_scores(num_mat, text_mats, w_num, w_fin)
            acc    = accuracy(assign_categories(scores)[label_mask], desired)
            if acc > best_acc:
                best_acc, best_w_num, best_w_fin = acc, w_num.copy(), w_fin.copy()

    return best_acc, best_w_num, best_w_fin

# ---------- MAIN --------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Optimise weights for Critical Job Positions model")
    parser.add_argument('--method', choices=['random', 'diffev', 'grid'], default='random',
                        help='optimisation strategy')
    parser.add_argument('--iterations', type=int, default=200,
                        help='iterations / maxiter (ignored for grid)')
    parser.add_argument('--minw', type=float, default=MIN_W, help='minimum weight')
    args = parser.parse_args()

    global MIN_W
    MIN_W = args.minw

    # -------------------------------------------------
    print("Reading data & pre-computing features …")
    sheet_df = pd.read_excel(CONFIG['input_file'])
    num_mat, text_mats, label_mask, desired = build_feature_matrices(sheet_df)
    print(f"   Rows: {sheet_df.shape[0]}  –  labelled for optimisation: {label_mask.sum()}")

    # -------------------------------------------------
    if args.method == 'random':
        acc, w_num, w_fin = optimise_random(num_mat, text_mats, label_mask, desired, args.iterations)
    elif args.method == 'diffev':
        acc, w_num, w_fin = optimise_diffev(num_mat, text_mats, label_mask, desired, args.iterations)
    else:
        acc, w_num, w_fin = optimise_grid(num_mat, text_mats, label_mask, desired)

    print("\n=== BEST RESULT ===")
    print(f"• Accuracy on labelled rows: {acc:.4%}")
    print("\n• Numeric feature weights")
    for col, w in zip(CONFIG['numeric_cols'], w_num):
        print(f"   {col:<25} {w:6.3f}")
    print("\n• Final-score weights")
    for k in CONFIG['seed_final_score_weights'].keys():
        print(f"   {k:<25} {w_fin[k]:6.3f}")

    # -------------------------------------------------
    # Re-run the full pipeline with best weights & write updated output file
    final_scores = compute_final_scores(num_mat, text_mats, w_num, w_fin)
    sheet_df['Final Composite Score'] = final_scores
    sheet_df['Category'] = assign_categories(final_scores)
    out_path = (Path(CONFIG['input_file']).with_suffix('')
                .as_posix() + '_optimised.xlsx')
    sheet_df.to_excel(out_path, index=False)
    print(f"\nOptimised results saved to →  {out_path}")

if __name__ == "__main__":
    main()
