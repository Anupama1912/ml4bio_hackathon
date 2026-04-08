# ML4Bio: ML-Guided Protein Engineering with Active Learning

This repository contains our solution for the ML4Bio hackathon on protein fitness prediction under an active learning setting. Our approach combines ESM-2 protein language model embeddings with a large ensemble of Random Forest and Extra Trees regressors, along with an uncertainty-aware active learning strategy based on Upper Confidence Bound (UCB).

## Repository Structure

- `final_mlb.py` — main script for feature extraction, model training, test prediction, and generation of submission files
- `hackathon_with_query_strategy.py` — script used to generate active learning query files (queries 2 & 3)
- `report.pdf` — final project report

## Requirements

- Python 3.10+ recommended
- CUDA-enabled GPU recommended for faster ESM embedding extraction, though CPU execution is possible

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Data Files

Datasets are not included due to submission guidelines. Please place the required input files in the same directory before running the code.

## Running the Main Model

To train the final model and generate submission files:

```bash
python final_mlb.py
```

This will generate:

- predictions.csv
- top10.txt

## Running Query Generation

To reproduce the active learning query generation procedure (for queries 2 & 3):

```bash
python hackathon_with_query_strategy.py
```

This will generate query request files such as:
- `query2_req.txt`
- `query3_req.txt`

Output Format

- `predictions.csv` contains test-set predictions in submission format
- `top10.txt` contains the top 10 recommended mutations

## Notes

- We use ESM-2 (`facebook/esm2_t30_150M_UR50D`) for feature extraction.
- The final feature representation includes mean sequence embedding, wild-type residue embedding, mutated residue embedding, delta embedding, and positional embedding.
- The final ensemble consists of three Random Forest and two Extra Trees regressors with different feature subsampling strategies.
- Large pretrained model weights are downloaded automatically from Hugging Face during execution.

## Contributors

- Kartheek Bellamkonda
- Shreya Chivilkar
- Anupama Nair
- Jay Sunil Sawant
- Vanshika Shah