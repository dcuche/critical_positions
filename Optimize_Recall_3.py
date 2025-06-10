# Recall_Version_3.py
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
    'output_file': 'Successors_Sample_optimized_RECALL_V3.xlsx',
    'embedding_model': 'text-embedding-3-large',
    'embedding_size': 3072,

    # --- Optimization Parameters ---
    'optimization_target_class': 'Critical',
    'recall_priority_multiplier': 20, # Very high priority for recall
    'class_labels': ['Critical', 'Important', 'Moderate'],
    'ground_truth_column': 'Desired_Outcome',

    # ### MODIFIED: Extremely deep search parameters for maximum exploration ###
    'n_iterations': 2500,
    'popsize': 50,
    'tol': 0.0001,
    'recombination': 0.7,
    'mutation': (0.5, 1.9), # Wider mutation to jump out of local optima

    # ### MODIFIED: Allow weights to be zero to completely discard a feature ###
    'min_weight_value': 0.0,

    # ### MODIFIED: Increased PCA components to capture even more text variance ###
    'text_features': {
        'Cargo': {'pca_components': 5, 'pickle': 'embeddings/embeddings_cargo_unab.pkl'},
        'Departamento': {'pca_components': 5, 'pickle': 'embeddings/embeddings_departamento_unab.pkl'},
        'Mission': {'pca_components': 10, 'pickle': 'embeddings/embeddings_mision_unab.pkl'},
        'Tasks': {'pca_components': 10, 'pickle': 'embeddings/embeddings_accion_unab.pkl'},
        'Results': {'pca_components': 10, 'pickle': 'embeddings/embeddings_resultado_unab.pkl'},
    },

    # ### MODIFIED: Widen the quantile bounds for more flexibility ###
    'quantile_bounds': {
        'critical_q': (0.55, 0.99),
        'important_q': (0.25, 0.54)
    },

    'categorical_features': {'mapping': {'Alto': 3, 'Medio': 2, 'Bajo': 1}, 'columns': ['Relacionamiento político', 'Nível técnico', 'Impacto en la estrategia', 'Nivel de Rotación', 'Dificultad de Reemplazo']},
    'numeric_features': {'columns': ['HayGroup', 'Nivel de Rotación', 'Dificultad de Reemplazo', 'Relacionamiento político', 'Nível técnico', 'Impacto en la estrategia'], 'column_to_invert': 'Nivel de Rotación'},
}
# --- End of Configuration Block ---

# Helper functions (unchanged)
client = OpenAI()
def get_embedding(text, model, size):
    if pd.isna(text) or text.strip() == '': return [0] * size
    try:
        response = client.embeddings.create(input=text, model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}"); return [0] * size
def load_or_generate_embeddings(pickle_file, text_series, model, size):
    os.makedirs(os.path.dirname(pickle_file), exist_ok=True); text_series = text_series.fillna('').astype(str)
    try:
        with open(pickle_file, 'rb') as f: embeddings_dict = pickle.load(f)
    except (FileNotFoundError, EOFError): embeddings_dict = {}
    new_texts = [text for text in text_series.unique() if text not in embeddings_dict]
    if new_texts:
        print(f'Generating {len(new_texts)} new embeddings for {os.path.basename(pickle_file)}...')
        for text in new_texts: embeddings_dict[text] = get_embedding(text, model, size)
        with open(pickle_file, 'wb') as f: pickle.dump(embeddings_dict, f)
    return text_series.map(embeddings_dict)


# ### ARCHITECTURE CHANGE: Core model logic is now "flattened" ###
def calculate_criticality(dataf, params_config, full_config):
    df = dataf.copy()
    
    # Unpack parameters
    all_feature_weights = params_config['weights']
    quantiles_config = params_config['quantiles']
    
    # Get the list of all feature columns in the correct order
    numeric_cols = full_config['numeric_features']['columns']
    all_pca_feature_names = [f'pca_{c}_{i}' for c, p in full_config['text_features'].items() for i in range(p['pca_components'])]
    all_feature_columns = numeric_cols + all_pca_feature_names

    # Combine all features into one DataFrame for the dot product
    combined_features_df = df[all_feature_columns].copy()

    # Apply the inversion directly to the feature column before weighting
    col_to_invert = full_config['numeric_features']['column_to_invert']
    if col_to_invert in combined_features_df.columns:
        # We assume data is already normalized 0-1 from the pre-processing step
        combined_features_df[col_to_invert] = 1 - combined_features_df[col_to_invert]

    # Calculate the final score with a single dot product
    ordered_weights = np.array([all_feature_weights[col] for col in all_feature_columns])
    df['Final Composite Score'] = combined_features_df.dot(ordered_weights)

    # Categorization logic remains the same
    critical_q = quantiles_config['critical_q']
    important_q = quantiles_config['important_q']
    if critical_q <= important_q:
        df['Category'] = 'Moderate'
        return df
        
    critical_threshold = df['Final Composite Score'].quantile(critical_q)
    important_threshold = df['Final Composite Score'].quantile(important_q)
    def categorize_position(score):
        if score >= critical_threshold: return 'Critical'
        elif score >= important_threshold: return 'Important'
        else: return 'Moderate'
    df['Category'] = df['Final Composite Score'].apply(categorize_position)
    return df

# --- Optimization Framework ---
class Optimizer:
    def __init__(self, config, processed_data):
        self.config = config
        self.data = processed_data
        self.truth_col = config['ground_truth_column']
        self.target_class = config['optimization_target_class']
        self.recall_multiplier = config['recall_priority_multiplier']
        
        self.optimization_data = self.data[self.data[self.truth_col].notna()].copy()
        self.y_true = self.optimization_data[self.truth_col]

        print(f"\nFound {len(self.optimization_data)} rows for optimization.")
        print(f"Optimizer targeting MAX RECALL for '{self.target_class}' using a flattened architecture.")
        
        # ### ARCHITECTURE CHANGE: Define a single, unified list of features to be weighted ###
        self.numeric_cols = self.config['numeric_features']['columns']
        self.pca_cols = [f'pca_{c}_{i}' for c, p in self.config['text_features'].items() for i in range(p['pca_components'])]
        self.all_feature_cols = self.numeric_cols + self.pca_cols
        
        self.n_weights = len(self.all_feature_cols)
        self.n_quantile_params = len(self.config['quantile_bounds'])

        self.best_score = float('inf')
        self.best_params = None
        self.iteration_count = 0

    def _unpack_parameters(self, flat_params_vector):
        p = flat_params_vector
        weights = p[:self.n_weights]
        quantiles = p[-self.n_quantile_params:]
        return {
            'weights': dict(zip(self.all_feature_cols, weights)),
            'quantiles': { 'critical_q': quantiles[0], 'important_q': quantiles[1] }
        }

    # ### MODIFIED: New aggressive, squared-error loss function ###
    def objective_function(self, flat_params_vector):
        params_config = self._unpack_parameters(flat_params_vector)
        if params_config['quantiles']['critical_q'] <= params_config['quantiles']['important_q']:
            return 100.0 # High penalty

        result_df = calculate_criticality(self.optimization_data, params_config, self.config)
        y_pred = result_df['Category']

        p_score = precision_score(self.y_true, y_pred, labels=[self.target_class], average='micro', zero_division=0.0)
        r_score = recall_score(self.y_true, y_pred, labels=[self.target_class], average='micro', zero_division=0.0)

        # The goal is to MINIMIZE the loss.
        # Squaring the recall error (1-r_score) heavily punishes scores far from 1.0.
        recall_loss = (1 - r_score) ** 2
        precision_loss = (1 - p_score)
        
        loss = (recall_loss * self.recall_multiplier) + precision_loss
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
            p_score = precision_score(self.y_true, y_pred, labels=[self.target_class], average='micro', zero_division=0)
            r_score = recall_score(self.y_true, y_pred, labels=[self.target_class], average='micro', zero_division=0)
            print(f"Iter: {self.iteration_count:04d} | New Best Loss: {loss:.4f} -> [Recall: {r_score:.3f}, Precision: {p_score:.3f}]")

    def run(self):
        total_params = self.n_weights + self.n_quantile_params
        q_bounds_config = self.config['quantile_bounds']
        quantile_lower = [q_bounds_config['critical_q'][0], q_bounds_config['important_q'][0]]
        quantile_upper = [q_bounds_config['critical_q'][1], q_bounds_config['important_q'][1]]

        lower_bounds = list(np.repeat(self.config['min_weight_value'], self.n_weights)) + quantile_lower
        upper_bounds = list(np.repeat(1.0, self.n_weights)) + quantile_upper
        bounds = Bounds(lower_bounds, upper_bounds)

        # ### ARCHITECTURE CHANGE: Only one constraint for the single weight vector ###
        weights_constraint_vector = [1.0] * self.n_weights + [0.0] * self.n_quantile_params
        constraints = [LinearConstraint(weights_constraint_vector, 1.0, 1.0)]

        start_time = time.time()
        result = differential_evolution(
            func=self.objective_function, bounds=bounds, constraints=constraints,
            maxiter=self.config['n_iterations'], popsize=self.config['popsize'],
            tol=self.config['tol'], recombination=self.config['recombination'],
            mutation=self.config['mutation'], callback=self.callback, disp=False, polish=True,
            updating='deferred', workers=-1
        )
        end_time = time.time()
        print(f"\n--- Differential Evolution Finished in {end_time - start_time:.2f} seconds ---")
        print(f"Message: {result.message}")
        return self.best_params if self.best_params is not None else result.x


def main():
    print("--- Step 1: Reading and Pre-processing Data ---")
    df = pd.read_excel(CONFIG['input_file'])
    cat_map = CONFIG['categorical_features']['mapping']
    for col in CONFIG['categorical_features']['columns']: df[col] = df[col].map(cat_map).fillna(0)
    
    # Normalize all numeric features 0-1 for the flattened model
    numeric_cols = CONFIG['numeric_features']['columns']
    df[numeric_cols] = df[numeric_cols].fillna(0)
    df[numeric_cols] = MinMaxScaler().fit_transform(df[numeric_cols])

    print("\n--- Step 1.5: Processing Text Features (Embeddings + PCA) ---")
    for col_name, params in CONFIG['text_features'].items():
        print(f"  - Processing '{col_name}' with {params['pca_components']} components...")
        embeddings = load_or_generate_embeddings(params['pickle'], df[col_name], CONFIG['embedding_model'], CONFIG['embedding_size'])
        embeddings_df = pd.DataFrame(embeddings.tolist(), index=df.index).fillna(0)
        # It's crucial to normalize embeddings before PCA
        embeddings_normalized = MinMaxScaler().fit_transform(embeddings_df)
        pca = PCA(n_components=params['pca_components'])
        pca_features = pca.fit_transform(embeddings_normalized)
        for i in range(params['pca_components']): df[f'pca_{col_name}_{i}'] = pca_features[:, i]

    print("\n--- Step 2: Optimizing Feature Weights and Thresholds ---")
    optimizer = Optimizer(CONFIG, df)
    best_params_vector = optimizer.run()
    if best_params_vector is None: print("\nOptimization failed. Exiting."); return

    best_params_config = optimizer._unpack_parameters(best_params_vector)
    
    print("\n--- Optimization Complete ---")
    print(f"Best Loss Found: {optimizer.best_score:.4f}")
    
    # ### MODIFIED: Print the new unified weights ###
    print("\nBest Unified Feature Weights Found (Top 15):")
    sorted_weights = sorted(best_params_config['weights'].items(), key=lambda item: item[1], reverse=True)
    for k, v in sorted_weights[:15]: print(f"  - {k}: {v:.4f}")

    print("\nBest Quantile Thresholds Found:")
    print(f"  - Critical Quantile: {best_params_config['quantiles']['critical_q']:.4f}")
    print(f"  - Important Quantile: {best_params_config['quantiles']['important_q']:.4f}")

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
    # Clean up the many helper columns before saving
    cols_to_drop = [col for col in final_df_sorted.columns if 'pca_' in col]
    final_df_sorted.drop(columns=cols_to_drop, inplace=True)
    final_df_sorted.to_excel(CONFIG['output_file'], index=False)
    print(f"\n✅ Success! Optimized model run complete. Results saved to '{CONFIG['output_file']}'")

if __name__ == '__main__':
    main()