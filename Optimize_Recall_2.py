# weight_optimizer_recall.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, classification_report, confusion_matrix
from scipy.optimize import differential_evolution, Bounds, LinearConstraint
import pickle
import os
from openai import OpenAI
import numpy as np
import time

# --- Configuration Block ---
CONFIG = {
    'input_file': 'Successors_Sample.xlsx',
    ### MODIFIED: Output file name reflects the new optimization goal ###
    'output_file': 'Successors_Sample_optimized_RECALL.xlsx',
    'embedding_model': 'text-embedding-3-large',
    'embedding_size': 3072,

    # --- Optimization Parameters ---
    'optimization_target_class': 'Critical',
    ### MODIFIED: The main goal is now maximizing Recall. Precision is secondary. ###
    'recall_priority_multiplier': 10, # Makes improving Recall 10x more important than improving Precision
    'class_labels': ['Critical', 'Important', 'Moderate'],
    'ground_truth_column': 'Desired_Outcome',
    'optimization_method': 'differential_evolution',

    # ### MODIFIED: Drastically increased search parameters for a much more thorough optimization ###
    'n_iterations': 1500,     # More generations to explore
    'popsize': 50,            # A larger population to search more widely
    'tol': 0.001,             # Tolerance for convergence
    'mutation': (0.5, 1.5),   # Widen mutation range for more exploration
    'recombination': 0.8,     # Standard recombination factor
    'min_weight_value': 0.01, # Minimum allowed value for any single weight

    'quantile_bounds': {
        'critical_q': (0.70, 0.98), # Allow Critical threshold to be between top 2% and top 30%
        'important_q': (0.40, 0.69) # Allow Important threshold to be between top 31% and top 60%
    },

    'text_features': { 'Cargo': {'pca_components': 2, 'pickle': 'embeddings/embeddings_cargo_unab.pkl'}, 'Departamento': {'pca_components': 2, 'pickle': 'embeddings/embeddings_departamento_unab.pkl'}, 'Mission': {'pca_components': 5, 'pickle': 'embeddings/embeddings_mision_unab.pkl'}, 'Tasks': {'pca_components': 5, 'pickle': 'embeddings/embeddings_accion_unab.pkl'}, 'Results': {'pca_components': 5, 'pickle': 'embeddings/embeddings_resultado_unab.pkl'}},
    'categorical_features': {'mapping': {'Alto': 3, 'Medio': 2, 'Bajo': 1}, 'columns': ['Relacionamiento político', 'Nível técnico', 'Impacto en la estrategia', 'Nivel de Rotación', 'Dificultad de Reemplazo']},
    'numeric_features': {'columns': ['HayGroup', 'Nivel de Rotación', 'Dificultad de Reemplazo', 'Relacionamiento político', 'Nível técnico', 'Impacto en la estrategia'], 'column_to_invert': 'Nivel de Rotación'},
    'final_score_composition': {'numeric_composite': True, 'text_features': ['Cargo', 'Departamento', 'Mission', 'Tasks', 'Results']},
}
# --- End of Configuration Block ---

# This section can remain as is...
client = OpenAI()

def get_embedding(text, model, size):
    if pd.isna(text) or text.strip() == '': return [0] * size
    try:
        response = client.embeddings.create(input=text, model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding for text: '{text[:50]}...'. Error: {e}")
        return [0] * size

def load_or_generate_embeddings(pickle_file, text_series, model, size):
    os.makedirs(os.path.dirname(pickle_file), exist_ok=True)
    text_series = text_series.fillna('').astype(str)
    try:
        with open(pickle_file, 'rb') as f: embeddings_dict = pickle.load(f)
        print(f'Loading embeddings from {pickle_file}')
    except (FileNotFoundError, EOFError): embeddings_dict = {}
    unique_texts = text_series.unique()
    new_texts = [text for text in unique_texts if text not in embeddings_dict]
    if new_texts:
        print(f'Generating embeddings for {len(new_texts)} new texts for {os.path.basename(pickle_file)}...')
        for i, text in enumerate(new_texts):
            if (i + 1) % 10 == 0: print(f"  - Generated {i+1}/{len(new_texts)}")
            embeddings_dict[text] = get_embedding(text, model, size)
        with open(pickle_file, 'wb') as f: pickle.dump(embeddings_dict, f)
    else: print(f'All embeddings for {os.path.basename(pickle_file)} are cached.')
    return text_series.map(embeddings_dict)

def calculate_criticality(dataf, params_config, full_config):
    df = dataf.copy()
    weights_config = params_config['weights']
    quantiles_config = params_config['quantiles']
    numeric_cols = full_config['numeric_features']['columns']
    normalized_numeric_df = df[numeric_cols].copy()
    col_to_invert = full_config['numeric_features']['column_to_invert']
    if col_to_invert in normalized_numeric_df.columns:
        normalized_numeric_df[col_to_invert] = 1 - normalized_numeric_df[col_to_invert]
    ordered_numeric_weights = np.array([weights_config['numeric_weights'][col] for col in numeric_cols])
    df['Weighted Composite Score'] = normalized_numeric_df.dot(ordered_numeric_weights)
    all_pca_feature_names = [f'pca_{col_name}_{i}' for col_name, params in full_config['text_features'].items() for i in range(params['pca_components'])]
    combined_features_df = pd.concat([df['Weighted Composite Score'].rename('numeric_composite')] + [df[col] for col in all_pca_feature_names], axis=1)
    combined_features_normalized = MinMaxScaler().fit_transform(combined_features_df)
    final_score_weights = np.array(weights_config['final_score_weights_list'])
    df['Final Composite Score'] = combined_features_normalized.dot(final_score_weights)
    critical_q = quantiles_config['critical_q']
    important_q = quantiles_config['important_q']
    if critical_q <= important_q: return df.assign(Category='Moderate')
    critical_threshold = df['Final Composite Score'].quantile(critical_q)
    important_threshold = df['Final Composite Score'].quantile(important_q)
    def categorize_position(score):
        if score >= critical_threshold: return 'Critical'
        elif score >= important_threshold: return 'Important'
        else: return 'Moderate'
    df['Category'] = df['Final Composite Score'].apply(categorize_position)
    return df

class Optimizer:
    def __init__(self, config, processed_data):
        self.config = config
        self.data = processed_data
        self.truth_col = config['ground_truth_column']
        self.target_class = config['optimization_target_class']
        self.recall_multiplier = config['recall_priority_multiplier']
        self.class_labels = config['class_labels']
        self.optimization_data = self.data[self.data[self.truth_col].notna()].copy()
        self.y_true = self.optimization_data[self.truth_col]

        print(f"\nFound {len(self.optimization_data)} rows with '{self.truth_col}' for optimization.")
        print(f"Optimizer will primarily target MAXIMUM RECALL for '{self.target_class}'.")
        print(f"Secondary objective: Maximize Precision for '{self.target_class}'.")

        self.numeric_cols = self.config['numeric_features']['columns']
        self.final_comp_cols = ['numeric_composite'] + self.config['final_score_composition']['text_features']
        self.n_numeric_weights = len(self.numeric_cols)
        self.n_final_weights = len(self.final_comp_cols)
        self.n_quantile_params = len(self.config['quantile_bounds'])
        self.best_score = float('inf')
        self.best_params = None
        self.iteration_count = 0

    def _unpack_parameters(self, flat_params_vector):
        p = flat_params_vector
        num_w = p[:self.n_numeric_weights]
        fin_w = p[self.n_numeric_weights : self.n_numeric_weights + self.n_final_weights]
        quantiles = p[-self.n_quantile_params:]
        final_score_weights_list = [fin_w[0]]
        w_idx = 1
        for col_name in self.config['final_score_composition']['text_features']:
            n_components = self.config['text_features'][col_name]['pca_components']
            feature_weight = fin_w[w_idx]
            final_score_weights_list.extend([feature_weight / n_components] * n_components)
            w_idx += 1
        return {
            'weights': { 'numeric_weights': dict(zip(self.numeric_cols, num_w)), 'final_score_weights_list': final_score_weights_list },
            'quantiles': { 'critical_q': quantiles[0], 'important_q': quantiles[1] }
        }

    # ### MODIFIED: The new objective function to maximize RECALL ###
    def objective_function(self, flat_params_vector):
        params_config = self._unpack_parameters(flat_params_vector)

        if params_config['quantiles']['critical_q'] <= params_config['quantiles']['important_q']:
            return 100.0 # A very large penalty

        result_df = calculate_criticality(self.optimization_data, params_config, self.config)
        y_pred = result_df['Category']

        p_score = precision_score(self.y_true, y_pred, labels=[self.target_class], average='micro', zero_division=0.0)
        r_score = recall_score(self.y_true, y_pred, labels=[self.target_class], average='micro', zero_division=0.0)

        # The goal is to MINIMIZE the loss.
        # Maximizing recall is the same as minimizing (1 - recall).
        # We give the recall component a much higher weight to prioritize it.
        loss = (1 - r_score) * self.recall_multiplier + (1 - p_score)

        return loss

    def callback(self, xk, convergence):
        self.iteration_count += 1
        loss = self.objective_function(xk)
        if loss < self.best_score:
            self.best_score = loss
            self.best_params = xk
            params_config = self._unpack_parameters(xk)
            result_df = calculate_criticality(self.optimization_data, params_config, self.config)
            y_pred = result_df['Category']
            p_score = precision_score(self.y_true, y_pred, labels=[self.target_class], average='micro', zero_division=0.0)
            r_score = recall_score(self.y_true, y_pred, labels=[self.target_class], average='micro', zero_division=0.0)
            # ### MODIFIED: Reporting now focuses on Recall ###
            print(f"Iter: {self.iteration_count:04d} | New Best Loss: {loss:.4f} -> [Recall: {r_score:.3f}, Precision: {p_score:.3f}]")

    def run(self):
        total_weights = self.n_numeric_weights + self.n_final_weights
        total_params = total_weights + self.n_quantile_params
        min_w = self.config['min_weight_value']
        q_bounds_config = self.config['quantile_bounds']
        quantile_lower_bounds = [q_bounds_config['critical_q'][0], q_bounds_config['important_q'][0]]
        quantile_upper_bounds = [q_bounds_config['critical_q'][1], q_bounds_config['important_q'][1]]
        lower_bounds = list(np.repeat(min_w, total_weights)) + quantile_lower_bounds
        upper_bounds = list(np.repeat(1.0, total_weights)) + quantile_upper_bounds
        bounds = Bounds(lower_bounds, upper_bounds)
        numeric_constraint = [1.0] * self.n_numeric_weights + [0.0] * (self.n_final_weights + self.n_quantile_params)
        final_constraint = [0.0] * self.n_numeric_weights + [1.0] * self.n_final_weights + [0.0] * self.n_quantile_params
        constraints = [
            LinearConstraint(numeric_constraint, 1.0, 1.0),
            LinearConstraint(final_constraint, 1.0, 1.0)
        ]
        start_time = time.time()
        result = differential_evolution(
            func=self.objective_function, bounds=bounds, constraints=constraints,
            maxiter=self.config['n_iterations'], popsize=self.config['popsize'], tol=self.config['tol'],
            mutation=self.config['mutation'], recombination=self.config['recombination'], # ### NEW: Using expanded search params
            callback=self.callback, disp=False, polish=True
        )
        end_time = time.time()
        print(f"\n--- Differential Evolution Finished in {end_time - start_time:.2f} seconds ---")
        print(f"Message: {result.message}")
        if self.best_params is None:
            print("Warning: The optimizer finished without finding a single valid parameter set. Using the result from the optimizer directly.")
            return result.x
        return self.best_params

# The main function remains largely the same, just executing the steps
def main():
    print("--- Step 1: Reading and Pre-processing Data ---")
    df = pd.read_excel(CONFIG['input_file'])
    cat_map = CONFIG['categorical_features']['mapping']
    for col in CONFIG['categorical_features']['columns']: df[col] = df[col].map(cat_map).fillna(0)
    numeric_cols = CONFIG['numeric_features']['columns']
    df[numeric_cols] = df[numeric_cols].fillna(0)
    df[numeric_cols] = MinMaxScaler().fit_transform(df[numeric_cols])
    for col_name, params in CONFIG['text_features'].items():
        print(f"  - Processing '{col_name}'...")
        embeddings = load_or_generate_embeddings(params['pickle'], df[col_name], CONFIG['embedding_model'], CONFIG['embedding_size'])
        embeddings_df = pd.DataFrame(embeddings.tolist(), index=df.index)
        embeddings_normalized = MinMaxScaler().fit_transform(embeddings_df)
        pca = PCA(n_components=params['pca_components'])
        pca_features = pca.fit_transform(embeddings_normalized)
        for i in range(params['pca_components']): df[f'pca_{col_name}_{i}'] = pca_features[:, i]
    print("\n--- Step 2: Optimizing Feature Weights and Thresholds ---")
    optimizer = Optimizer(CONFIG, df)
    best_params_vector = optimizer.run()
    if best_params_vector is None: print("\nOptimization failed to find a best parameter set. Exiting."); return
    best_params_config = optimizer._unpack_parameters(best_params_vector)
    print("\n--- Optimization Complete ---")
    print(f"Best Loss Found: {optimizer.best_score:.4f}")
    best_quantiles = best_params_config['quantiles']
    print("\nBest Quantile Thresholds Found:")
    print(f"  - Critical Threshold (quantile): {best_quantiles['critical_q']:.4f} (Top {100-best_quantiles['critical_q']*100:.1f}%)")
    print(f"  - Important Threshold (quantile): {best_quantiles['important_q']:.4f}")
    best_weights = best_params_config['weights']
    print("\nBest Numeric Weights:")
    for k, v in sorted(best_weights['numeric_weights'].items(), key=lambda item: item[1], reverse=True): print(f"  - {k}: {v:.4f}")
    print("\nBest Final Score Weights (Component-level):")
    fin_w = best_params_vector[optimizer.n_numeric_weights : optimizer.n_numeric_weights + optimizer.n_final_weights]
    final_weights_display = {'numeric_composite': fin_w[0]}
    w_idx = 1
    for col_name in CONFIG['final_score_composition']['text_features']: final_weights_display[col_name] = fin_w[w_idx]; w_idx += 1
    for k,v in sorted(final_weights_display.items(), key=lambda item: item[1], reverse=True): print(f"  - {k}: {v:.4f}")
    print("\n--- Step 3: Applying Optimal Parameters to Entire Dataset ---")
    final_df = calculate_criticality(df, best_params_config, CONFIG)
    print("\n--- Step 4: Evaluating Final Model Performance on Labeled Data ---")
    truth_col = CONFIG['ground_truth_column']
    evaluation_df = final_df[final_df[truth_col].notna()].copy()
    if not evaluation_df.empty:
        y_true_final = evaluation_df[truth_col]
        y_pred_final = evaluation_df['Category']
        print("\nFinal Classification Report:")
        print(classification_report(y_true_final, y_pred_final, labels=CONFIG['class_labels'], zero_division=0))
        print("Final Confusion Matrix:")
        cm = confusion_matrix(y_true_final, y_pred_final, labels=CONFIG['class_labels'])
        print(pd.DataFrame(cm, index=[f'True_{l}' for l in CONFIG['class_labels']], columns=[f'Pred_{l}' for l in CONFIG['class_labels']]))
    print("\n--- Step 5: Exporting Final Results ---")
    final_df_sorted = final_df.sort_values(by='Final Composite Score', ascending=False)
    pca_cols_to_drop = [col for col in final_df_sorted.columns if 'pca_' in col]
    final_df_sorted.drop(columns=pca_cols_to_drop, inplace=True)
    final_df_sorted.to_excel(CONFIG['output_file'], index=False)
    print(f"\n✅ Success! Optimized model run complete. Results saved to '{CONFIG['output_file']}'")

if __name__ == '__main__':
    main()