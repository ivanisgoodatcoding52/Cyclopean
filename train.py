# Requirements: pip install ollama scikit-learn numpy pandas datasets kaggle tqdm

import datetime
import os
import sys
import json
import pickle
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import ollama
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from datasets import load_dataset

EMBED = "nomic-embed-text" # ollama pull nomic-embed-text
OUTPUT = "detect.pkl"
MAXSOURCE = 1500
MAXCHAR = 2000
MINCHAR = 150
DATA = Path("./datasets")
BALANCE = True
SEED = 42

def check():
    try:
        models = ollama.list()
        names = [m["name"].split(":")[0] for m in models.get("models", [])]
        if EMBED not in names:
            print(f"[setup] Pulling {EMBED} from ollama")
            ollama.pull(EMBED)
        else:
            print(f"[setup] {EMBED} is already available in ollama")
    except Exception as e:
        print(f"[ERROR] Ollama is not available: {e}")
        print("Please install Ollama and ensure it's running by 'ollama serve' before running this script.")
        sys.exit(1)

def kaggle():
    cred = Path.home() / ".kaggle" / "kaggle.json"
    if not cred.exists():
        print(f"[ERROR] Kaggle credentials not found at {cred}")
        print("Please create a kaggle.json file with your Kaggle API credentials and place it in ~/.kaggle/")
        print("The kaggle.json file should have the following format:")
        print('{"username": "your_username", "key": "your_api_key"}')
        print("For more information, see https://www.kaggle.com/general/74235")
        print("Kaggle datasets will be skipped.")
        return False
    else:
        print(f"[setup] Kaggle credentials found at {cred}")
        return True

def downloadkaggle(slug: str, dest: Path) -> bool:
    dest.mkdir(parents=True, exist_ok=True)
    try:
        print(f"[kaggle] Downloading dataset {slug} to {dest}")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"],
            check=True, capture_output=True
        )
        print(f"[kaggle] Successfully downloaded and extracted {slug} to {dest}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to download dataset {slug}: {e.stderr.decode()}")
        return False

def loaddaigtv2(data_dir: Path) -> pd.DataFrame:
    """
    DAIGT V2 - found at github.com/thedrcat/daigt-v2-train-dataset
    Columns: id, text, label (0=human, 1=ai), prompt_name, source, RDizzl3_seven
    ~300k rows covering GPT3.5, Llama2, Mistral, Falcon, etc.
    """
    slug = "thedrcat/daigt-v2-train-dataset"
    dest = data_dir / "daigt-v2"

    if not any(dest.glob("*.csv")):
        if not downloadkaggle(slug, dest):
            return pd.DataFrame()

    csvs = list(dest.glob("*.csv"))
    if not csvs:
        print(f"[ERROR] No CSV files found in {dest} after downloading.")
        return pd.DataFrame()

    frames = []
    for csv in csvs:
        try:
            df = pd.read_csv(csv, usecols=lambda c: c in ["text", "label", "generated"])
            if "label" in df.columns:
                df = df.rename(columns={"label": "generated"})
            if "text" in df.columns and "generated" in df.columns:
                frames.append(df[["text", "generated"]])
        except Exception:
            pass  # Skip files that can't be read or don't have the right columns

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result = result.dropna(subset=["text", "generated"])
    result["generated"] = result["generated"].astype(int)
    result["source"] = "daigt-v2"
    print(f"[daigt-v2] Loaded {len(result)} rows from {len(csvs)} CSV files.")
    return result

def loaddaigtv4(data_dir: Path) -> pd.DataFrame:
    """
    DAIGT V4 - found at github.com/thedrcat/daigt-v4-train-dataset
    Columns: id, text, label (0=human, 1=ai), prompt_name, source, RDizzl3_seven
    ~300k rows covering GPT3.5, Llama2, Mistral, Falcon, etc.
    """
    slug = "thedrcat/daigt-v4-train-dataset"
    dest = data_dir / "daigt-v4"

    if not any(dest.glob("*.csv")):
        if not downloadkaggle(slug, dest):
            return pd.DataFrame()

    csvs = list(dest.glob("*.csv"))
    if not csvs:
        print(f"[ERROR] No CSV files found in {dest} after downloading.")
        return pd.DataFrame()

    frames = []
    for csv in csvs:
        try:
            df = pd.read_csv(csv, usecols=lambda c: c in ["text", "label", "generated"])
            if "label" in df.columns:
                df = df.rename(columns={"label": "generated"})
            if "text" in df.columns and "generated" in df.columns:
                frames.append(df[["text", "generated"]])
        except Exception:
            pass  # Skip files that can't be read or don't have the right columns

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result = result.dropna(subset=["text", "generated"])
    result["generated"] = result["generated"].astype(int)
    result["source"] = "daigt-v4"
    print(f"[daigt-v4] Loaded {len(result)} rows from {len(csvs)} CSV files.")
    return result

def sunilthite(data_dir: Path) -> pd.DataFrame:
    """
    found on kaggle: sunilthite/llm-detect-ai-generated-text-dataset
    columns: text, generated (0 or 1)
    """
    slug = "sunilthite/llm-detect-ai-generated-text-dataset"
    dest = data_dir / "sunilthite"

    if not any(dest.glob("*.csv")):
        if not downloadkaggle(slug, dest):
            return pd.DataFrame()

    csvs = list(dest.glob("*.csv"))
    if not csvs:
        print(f"[ERROR] No CSV files found in {dest} after downloading.")
        return pd.DataFrame()

    frames = []
    for csv in csvs:
        try:
            df = pd.read_csv(csv, usecols=lambda c: c in ["text", "generated"])
            if "text" in df.columns and "generated" in df.columns:
                frames.append(df[["text", "generated"]])
        except Exception:
            pass  # Skip files that can't be read or don't have the right columns

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result = result.dropna(subset=["text", "generated"])
    result["generated"] = result["generated"].astype(int)
    result["source"] = "sunilthite"
    print(f"[sunilthite] Loaded {len(result)} rows from {len(csvs)} CSV files.")
    return result

def loadhfdetectionpile(max_per_class: int = MAXSOURCE) -> pd.DataFrame:  # FIX: was MAX_PER_SOURCE (undefined), use MAXSOURCE
    """
    found at huggingface.co/datasets/artem9k/ai-text-detection-pile
    Columns: text, source ("human" or "ai")
    Contains GPT2, GPT3, ChatGPT, GPTJ vs Human text from The Pile
    """
    print("[hf-detection-pile] Loading dataset from Huggingface")
    try:
        ds = load_dataset("artem9k/ai-text-detection-pile", split="train", streaming=True)
        rows = []
        ai_n, human_n = 0, 0

        for row in ds:
            text = (row.get("text") or "").strip()
            src  = (row.get("source") or "").lower()

            if len(text) < MINCHAR:  # FIX: was MIN_TEXT_CHARS (undefined), use MINCHAR
                continue

            if src == "ai" and ai_n < max_per_class:
                rows.append({"text": text, "generated": 1, "source": "hf-detection-pile"})  # FIX: was row.append() (wrong var)
                ai_n += 1
            elif src == "human" and human_n < max_per_class:
                rows.append({"text": text, "generated": 0, "source": "hf-detection-pile"})  # FIX: same
                human_n += 1

            if ai_n >= max_per_class and human_n >= max_per_class:
                break

        df = pd.DataFrame(rows)
        print(f"[hf-detection-pile] Loaded {len(df)} rows ({ai_n} AI, {human_n} human).")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load Huggingface dataset: {e}")
        return pd.DataFrame()

def merge_and_balance(*frames: pd.DataFrame, max_per_source: int = MAXSOURCE) -> pd.DataFrame:
    combined = []
    for df in frames:
        if df.empty:  # FIX: was "if not df.empty: continue" — logic was inverted, skipped valid data
            continue
        sampled = df.sample(min(len(df), max_per_source), random_state=SEED)
        combined.append(sampled)

    if not combined:
        print("[ERROR] No valid data frames to merge.")
        sys.exit(1)

    full = pd.concat(combined, ignore_index=True)
    full = full.drop_duplicates(subset=["text"])
    full = full[full["text"].str.len() >= MINCHAR]

    print(f"[merge] Merged dataset has {len(full)} rows after filtering.")
    if "source" in full.columns:
        print(full["source"].value_counts().to_string())
    print(f"        AI=1: {(full['generated']==1).sum():,}  "
          f"Human=0: {(full['generated']==0).sum():,}")

    if BALANCE:
        ai_df    = full[full["generated"] == 1]
        human_df = full[full["generated"] == 0]
        n        = min(len(ai_df), len(human_df))
        full = pd.concat([
            ai_df.sample(n, random_state=SEED),    # FIX: was RANDOM_SEED (undefined), use SEED
            human_df.sample(n, random_state=SEED),  # FIX: same
        ]).sample(frac=1, random_state=SEED).reset_index(drop=True)
        print(f"[merge] Balanced to {n:,} per class → {len(full):,} total")

    return full

def getembed(text: str) -> np.ndarray:
    resp = ollama.embeddings(model=EMBED, prompt=text[:MAXCHAR])
    return np.array(resp.get("embedding"), dtype=np.float32)  # FIX: dtype was inside .get() as a kwarg instead of on np.array()

def buildembed(texts: list[str]) -> np.ndarray:
    embeddings = []
    failed = 0
    for text in tqdm(texts, desc="Building embeddings"):
        try:
            emb = getembed(text)
            embeddings.append(emb)
        except Exception:
            failed += 1
            embeddings.append(np.zeros(768, dtype=np.float32))  # Assuming 768-dim embeddings
    if failed > 0:
        print(f"[WARNING] Failed to get embeddings for {failed} texts. Filled with zeros.")
    return np.vstack(embeddings)

def train():
    check()
    has_kaggle = kaggle()
    DATA.mkdir(exist_ok=True)

    print("[load] Loading datasets...")
    hf_df = loadhfdetectionpile()                      # FIX: HF needs no kaggle creds, always load it
    v2_df = loaddaigtv2(DATA) if has_kaggle else pd.DataFrame()
    v4_df = loaddaigtv4(DATA) if has_kaggle else pd.DataFrame()
    st_df = sunilthite(DATA)  if has_kaggle else pd.DataFrame()

    print("[merge] Merging and balancing datasets...")
    full_df = merge_and_balance(hf_df, v2_df, v4_df, st_df)

    if "source" in full_df.columns:
        print("[sources] Dataset source distribution:")
        print(full_df["source"].value_counts().to_string())

    print("[embed] Building embeddings for training...")
    print("this is the slow part go do something else :3")
    texts  = full_df["text"].tolist()
    labels = full_df["generated"].values
    x = buildembed(texts)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=SEED)
    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)

    print(classification_report(y_test, y_pred, target_names=["Human", "AI"]))
    print(f"AUC: {auc:.4f}")

    with open(OUTPUT, "wb") as f:
        pickle.dump(clf, f)

    meta = {                                                           # FIX: closing } was inside the with block (syntax error)
        "embed":      EMBED,
        "train_size": int(X_train.shape[0]),
        "test_size":  int(X_test.shape[0]),
        "threshold":  0.65,
        "auc":        round(float(auc), 4),
        "date":       datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # FIX: was datetime.now() — needs datetime.datetime.now()
        "sources": {
            "hf-detection-pile": len(hf_df),
            "daigt-v2":          len(v2_df),
            "daigt-v4":          len(v4_df),
            "sunilthite":        len(st_df),
            "total":             len(full_df),
        }
    }
    with open("ai-detect-meta.json", "w") as f:
        json.dump(meta, f, indent=4)

    print(f"\n✅ Model saved  → {OUTPUT}")
    print(f"✅ Metadata     → ai-detect-meta.json")

if __name__ == "__main__":
    train()