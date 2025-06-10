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
    'input_file': 'Successors_Output_v7.xlsx',
    'output_file': 'Successors_Output_v7_optimized_RECALL.xlsx',
    'embedding_model': 'text-embedding-3-large',
    'embedding_size': 3072,

    # --- Optimization Parameters ---
    'optimization_target_class': 'Critical',
    'recall_target': 0.90, # ### NEW: Explicitly define the RECALL goal ###
    'class_labels': ['Critical', 'Important', 'Moderate'],
    'ground_truth_column': 'Desired_Outcome',
    
    # ### MODIFIED: Maintain deep search parameters & introduce mutation ###
    'n_iterations': 2000,        # Keep high iterations
    'popsize': 40,               # Keep larger population
    'tol': 0.0001,               # Keep tight tolerance
    'recombination': 0.7,        # Standard setting
    'mutation': (0.5, 1.5),      # ### NEW: Widen mutation to escape local minima ###
    'min_weight_value': 0.0,

    # Keep the more granular PCA components
    'text_features': {
        'Cargo': {'pca_components': 4, 'pickle': 'embeddings/embeddings_cargo_unab.pkl'},
        'Departamento': {'pca_components': 4, 'pickle': 'embeddings/embeddings_departamento_unab.pkl'},
        'Mission': {'pca_components': 8, 'pickle': 'embeddings/embeddings_mision_unab.pkl'},
        'Tasks': {'pca_components': 8, 'pickle': 'embeddings/embeddings_accion_unab.pkl'},
        'Results': {'pca_components': 8, 'pickle': 'embeddings/embeddings_resultado_unab.pkl'},
    },

    'quantile_bounds': {
        'critical_q': (0.60, 0.99),
        'important_q': (0.30, 0.59)
    },

    'categorical_features': {'mapping': {'Alto': 3, 'Medio': 2, 'Bajo': 1}, 'columns': ['Relacionamiento político', 'Nível técnico', 'Impacto en la estrategia', 'Nivel de Rotación', 'Dificultad de Reemplazo']},
    'numeric_features': {'columns': ['HayGroup', 'Nivel de Rotación', 'Dificultad de Reemplazo', 'Relacionamiento político', 'Nível técnico', 'Impacto en la estrategia'], 'column_to_invert': 'Nivel de Rotación'},
    'final_score_composition': {'numeric_composite': True, 'text_features': ['Cargo', 'Departamento', 'Mission', 'Tasks', 'Results']},
}
# --- End of Configuration Block ---

client = OpenAI()

# Helper and Core Model Logic functions (unchanged from previous version)
def get_embedding(text, model, size):
    if pd.isna(text) or text.strip() == '': return [0] * size
    try:
        response = client.embeddings.create(input=text, model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}"); return [0] * size

def load_or_generate_embeddings(pickle_file, text_series, model, size):
    os.makedirs(os.path.dirname(pickle_file), exist_ok=True)
    text_series = text_series.fillna('').astype(str)
    try:
        with open(pickle_file, 'rb') as f: embeddings_dict = pickle.load(f)
    except (FileNotFoundError, EOFError): embeddings_dict = {}
    unique_texts = text_series.unique()
    new_texts = [text for text in unique_texts if text not in embeddings_dict]
    if new_texts:
        print(f'Generating {len(new_texts)} new embeddings for {os.path.basename(pickle_file)}...')
        for text in new_texts: embeddings_dict[text] = get_embedding(text, model, size)
        with open(pickle_file, 'wb') as f: pickle.dump(embeddings_dict, f)
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
        self.recall_target = config['recall_target'] # New target
        self.class_labels = config['class_labels']
        
        self.optimization_data = self.data[self.data[self.truth_col].notna()].copy()
        self.y_true = self.optimization_data[self.truth_col]
        
        print(f"\nFound {len(self.optimization_data)} rows with '{self.truth_col}' for optimization.")
        print(f"Optimizer will primarily target a Recall of >{self.recall_target:.2f} for '{self.target_class}'.")
        print(f"Secondary objective: Maximize Precision for '{self.target_class}' without dropping recall.")

        self.numeric_cols = self.config['numeric_features']['columns']
        self.final_comp_cols = ['numeric_composite'] + [name for name in self.config['text_features'].keys()]
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

    # ### HEAVILY MODIFIED: The new tiered objective function for RECALL ###
    def objective_function(self, flat_params_vector):
        params_config = self._unpack_parameters(flat_params_vector)
        if params_config['quantiles']['critical_q'] <= params_config['quantiles']['important_q']:
            return 10.0 # A large penalty

        result_df = calculate_criticality(self.optimization_data, params_config, self.config)
        y_pred = result_df['Category']

        p_score = precision_score(self.y_true, y_pred, labels=[self.target_class], average='micro', zero_division=0.0)
        r_score = recall_score(self.y_true, y_pred, labels=[self.target_class], average='micro', zero_division=0.0)
        
        # New tiered loss function, focused on RECALL. We want to MINIMIZE this value.
        recall_gap = self.recall_target - r_score
        
        if recall_gap > 0:
            # PENALTY MODE: We are below the recall target.
            # The main penalty is how far we are from the target recall.
            # A smaller penalty for low precision guides the search slightly.
            loss = (recall_gap * 10) + (1 - p_score) 
        else:
            # TARGET MET MODE: We have reached our recall goal.
            # Now, the only goal is to maximize precision.
            # Minimizing (1 - precision) is the same as maximizing precision.
            loss = (1 - p_score)

        return loss
    
    def callback(self, xk, convergence):
        self.iteration_count += 1
        # To avoid re-calculating, we can't easily get both scores and the loss
        # without running the function again. So we just show the loss.
        # This is a small price for a cleaner objective function.
        loss = self.objective_function(xk)
        if loss < self.best_score:
            self.best_score = loss
            self.best_params = xk
            # Let's get the detailed scores for our printout
            params_config = self._unpack_parameters(xk)
            result_df = calculate_criticality(self.optimization_data, params_config, self.config)
            y_pred = result_df['Category']
            p_score = precision_score(self.y_true, y_pred, labels=[self.target_class], average='micro', zero_division=0.0)
            r_score = recall_score(self.y_true, y_pred, labels=[self.target_class], average='micro', zero_division=0.0)
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
        constraints = [ LinearConstraint(numeric_constraint, 1.0, 1.0), LinearConstraint(final_constraint, 1.0, 1.0) ]
        
        start_time = time.time()
        result = differential_evolution(
            func=self.objective_function, bounds=bounds, constraints=constraints,
            maxiter=self.config['n_iterations'], popsize=self.config['popsize'],
            tol=self.config['tol'], recombination=self.config['recombination'],
            mutation=self.config['mutation'], # Add mutation parameter
            callback=self.callback, disp=False, polish=True,
            updating='deferred', # Can help with parallelization under the hood
            workers=-1 # Use all available CPU cores
        )
        end_time = time.time()
        print(f"\n--- Differential Evolution Finished in {end_time - start_time:.2f} seconds ---")
        print(f"Message: {result.message}")
        return self.best_params

def main():
    print("--- Step 1: Reading and Pre-processing Data ---")
    df = pd.read_excel(CONFIG['input_file'])
    cat_map = CONFIG['categorical_features']['mapping']
    for col in CONFIG['categorical_features']['columns']: df[col] = df[col].map(cat_map).fillna(0)
    numeric_cols = CONFIG['numeric_features']['columns']
    df[numeric_cols] = df[numeric_cols].fillna(0)
    df[numeric_cols] = MinMaxScaler().fit_transform(df[numeric_cols])
    
    print("\n--- Step 1.5: Processing Text Features (Embeddings + PCA) ---")
    for col_name, params in CONFIG['text_features'].items():
        print(f"  - Processing '{col_name}' with {params['pca_components']} components...")
        embeddings = load_or_generate_embeddings(params['pickle'], df[col_name], CONFIG['embedding_model'], CONFIG['embedding_size'])
        embeddings_df = pd.DataFrame(embeddings.tolist(), index=df.index).fillna(0)
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
    pca_cols_to_drop = [col for col in final_df_sorted.columns if 'pca_' in col]
    final_df_sorted.drop(columns=pca_cols_to_drop, inplace=True)
    final_df_sorted.to_excel(CONFIG['output_file'], index=False)
    print(f"\n✅ Success! Optimized model run complete. Results saved to '{CONFIG['output_file']}'")

if __name__ == '__main__':
    main()