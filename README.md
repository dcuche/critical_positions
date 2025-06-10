# Critical Positions Analysis

This repository contains a collection of Python scripts for evaluating job positions and classifying them into **Critical**, **Important**, or **Moderate** categories. The main workflow combines numeric attributes with text features processed through OpenAI embeddings and Principal Component Analysis (PCA). Additional optimization scripts use differential evolution to fine‑tune the weighting of these features to match manually labelled data.

## Contents

- `Critical_Positions_Model_v4.py` – baseline model that calculates a final composite score and category for each position.
- `Parameter_Optimizer.py` – differential evolution optimiser for feature weights and categorisation thresholds.
- `Optimize_*` scripts – variants of the optimiser that focus on different objectives such as precision, recall, or balanced optimisation.
- `embeddings/` – cached embedding pickle files used by the scripts.
- `Successors_Sample.xlsx` – example input spreadsheet containing the data to score.
- `requirements.txt` – Python dependencies.

## Setup

1. Install Python 3.10 or newer.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your OpenAI API key in the environment variable `OPENAI_API_KEY` before running any script that requests embeddings.

## Running the baseline model

The simplest way to generate scores is:

```bash
python Critical_Positions_Model_v4.py
```

The script reads `Successors_Sample.xlsx`, calculates embeddings (cached in `embeddings/`), performs PCA and weighting, and writes the ranked results to `Successors_Sample_v4.xlsx`.

## Optimising parameters

Several scripts search for the best feature weights and quantile thresholds to match labelled data in the input spreadsheet. For example:

```bash
python Parameter_Optimizer.py
```

The optimisers may take a long time to run. Review the configuration block at the top of each script for tunable parameters such as the number of PCA components or population size for the differential evolution algorithm.

## Data

`Successors_Sample.xlsx` is a sample dataset containing both numeric and text columns. Text columns include fields such as `Cargo`, `Departamento`, `Mission`, `Tasks`, and `Results`. Numeric columns include `HayGroup`, `Nivel de Rotación`, `Dificultad de Reemplazo`, `Relacionamiento político`, `Nível técnico`, and `Impacto en la estrategia`.

## License

This repository does not include an explicit license. Use at your own discretion.
