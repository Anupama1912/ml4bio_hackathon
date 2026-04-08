import pandas as pd
import numpy as np
import torch
from transformers import EsmModel, AutoTokenizer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from tqdm import tqdm
import os
from scipy.stats import rankdata

# ==========================================
# 1. CONFIGURATION
# ==========================================
ESM_MODEL_NAME = "facebook/esm2_t30_150M_UR50D" 
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"--- Final Titan Ensemble starting on {device} ---")

# ==========================================
# 2. DATA PREPARATION (Base + Queries)
# ==========================================
def apply_mutation(wt_seq, mutant_code):
    wt_aa, mut_aa = mutant_code[0], mutant_code[-1]
    pos = int(mutant_code[1:-1])
    mutated_seq = list(wt_seq)
    mutated_seq[pos] = mut_aa
    return "".join(mutated_seq)

print("Loading datasets...")
with open("sequence.fasta", 'r') as f:
    wt_sequence = "".join([line.strip() for line in f.readlines() if not line.startswith(">")])

seq_len = len(wt_sequence)
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Load all collected queries
query_files = ["query1_result.csv", "query2_result.csv", "query3_result.csv"]
query_dfs = []

for q_file in query_files:
    if os.path.exists(q_file):
        df = pd.read_csv(q_file)
        query_dfs.append(df)
        print(f"Loaded {len(df)} mutants from {q_file}")
    else:
        print(f"Warning: {q_file} not found. Skipping.")

# Combine base training data with all queried data
if query_dfs:
    all_queries_df = pd.concat(query_dfs, ignore_index=True)
    final_train_df = pd.concat([train_df, all_queries_df], ignore_index=True)
else:
    final_train_df = train_df

print(f"\nTotal training samples available: {len(final_train_df)}")

final_train_df["sequence"] = final_train_df["mutant"].apply(lambda x: apply_mutation(wt_sequence, x))
test_df["sequence"] = test_df["mutant"].apply(lambda x: apply_mutation(wt_sequence, x))


print(f"\nLoading ESM-2 model: {ESM_MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
esm_model = EsmModel.from_pretrained(ESM_MODEL_NAME).to(device)
esm_model.eval()

print("Calculating Wild-Type Baseline...")
with torch.no_grad():
    wt_inputs = tokenizer(wt_sequence, return_tensors="pt").to(device)
    wt_outputs = esm_model(**wt_inputs)
    wt_hidden_states = wt_outputs.last_hidden_state[0].cpu().numpy()

def extract_full_picture_embeddings(sequences, mutant_codes, batch_size=16):
    embeddings = []
    positions = [int(m[1:-1]) for m in mutant_codes]
    
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting Titan Embeddings"):
            batch_seqs = sequences[i:i+batch_size].tolist()
            batch_pos = positions[i:i+batch_size]
            
            inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = esm_model(**inputs)
            
            for b_idx, p in enumerate(batch_pos):
                mut_emb = outputs.last_hidden_state[b_idx, p + 1, :].cpu().numpy()
                wt_emb = wt_hidden_states[p + 1, :]
                delta_emb = mut_emb - wt_emb
                mean_emb = outputs.last_hidden_state[b_idx, 1:-1, :].mean(dim=0).cpu().numpy()
                pos_norm = np.array([p / (seq_len - 1)], dtype=np.float32)

                embeddings.append(np.concatenate([mean_emb, wt_emb, mut_emb, delta_emb, pos_norm]))
                
    return np.vstack(embeddings)

print("Pre-computing final training embeddings...")
X_train = extract_full_picture_embeddings(final_train_df["sequence"], final_train_df["mutant"])
y_train = final_train_df["DMS_score"].values

print("Pre-computing final testing embeddings...")
X_test = extract_full_picture_embeddings(test_df["sequence"], test_df["mutant"])

# Free up GPU memory
del esm_model
torch.cuda.empty_cache()

# ==========================================
# 4. THE 12,500 TREE TITAN ENSEMBLE
# ==========================================
print("\n" + "="*50)
print("FINAL TRAINING: 12,500 TREE TITAN ENSEMBLE")
print("="*50)

print("Training Random Forest 1/5 (max_features='sqrt')...")
rf1 = RandomForestRegressor(n_estimators=2500, max_features='sqrt', n_jobs=-1, random_state=42)
rf1.fit(X_train, y_train)

print("Training Random Forest 2/5 (max_features=0.33)...")
rf2 = RandomForestRegressor(n_estimators=2500, max_features=0.33, n_jobs=-1, random_state=1337)
rf2.fit(X_train, y_train)

print("Training Random Forest 3/5 (max_features=0.1)...")
rf3 = RandomForestRegressor(n_estimators=2500, max_features=0.1, n_jobs=-1, random_state=2025)
rf3.fit(X_train, y_train)

# -------------------------
# Extra Trees models
# -------------------------

print("Training Extra Trees 4/5 (max_features='log2')...")
et1 = ExtraTreesRegressor(n_estimators=2500, max_features='log2', n_jobs=-1, random_state=2024)
et1.fit(X_train, y_train)

print("Training Extra Trees 5/5 (max_features=0.25)...")
et2 = ExtraTreesRegressor(n_estimators=2500, max_features=0.25, n_jobs=-1, random_state=999)
et2.fit(X_train, y_train)

print("\nPredicting on test set with all 5 models...")
preds_rf1 = rf1.predict(X_test)
preds_rf2 = rf2.predict(X_test)
preds_rf3 = rf3.predict(X_test)
preds_et1 = et1.predict(X_test)
preds_et2 = et2.predict(X_test)

print("Performing rank-based blending of predictions...")
r1 = rankdata(preds_rf1)
r2 = rankdata(preds_rf2)
r3 = rankdata(preds_rf3)
r4 = rankdata(preds_et1)
r5 = rankdata(preds_et2)

rank_blend = (r1 + r2 + r3 + r4 + r5) / 5.0
rank_blend = (rank_blend - rank_blend.min()) / (rank_blend.max() - rank_blend.min())

# Blend
test_df["DMS_score_predicted"] = rank_blend

# ==========================================
# 5. GENERATE SUBMISSION FILES
# ==========================================
print("Formatting and saving predictions.csv...")

test_df = test_df.rename(columns={"DMS_score_predicted": "DMS_score"})

if "id" not in test_df.columns:
    test_df["id"] = test_df.index

test_df[["id", "DMS_score"]].to_csv("predictions.csv", index=False)

print("Saving top10.txt...")
# Sort by the newly renamed 'DMS_score' to get the best mutants
top10_mutants = test_df.sort_values(by="DMS_score", ascending=False).head(10)["mutant"].tolist()

with open("top10.txt", "w") as f:
    f.write("\n".join(top10_mutants))
