# Multimedia Retrieval – 3D Shape Retrieval System

This project implements a complete **Content-Based 3D Shape Retrieval (CBSR)** pipeline, following the Multimedia Retrieval course.  
Given a 3D mesh, the system extracts geometric descriptors, compares them to a feature database, and returns the most similar shapes.  
A modern **Streamlit application** is included for interactive querying.

**Authors:**  
- Amjad Hwidy  
- Binxu Jiang  

---

## 📁 Project Structure (Important Folders)

```
Multimedia_Retrival/
│
├── app.py                     # Streamlit app (interactive viewer)
├── mesh_normalization.py      # Step 2–3.1: normalization pipeline
├── obj_viewer.py              # Step 1: simple mesh viewer
├── scalability.py             # Step 5–6: scalability + evaluation
├── show_one_class.py          # View all meshes in a class
├── validate_norm.py           # Check normalization results
│
├── step2_results/             # Preprocessing, stats, resampled meshes
├── step3_1_results/           # Fully PCA-normalized meshes
├── norm_analysis/             # Normalization analysis for step 2 & 3.1
│
└── MR System/
    ├── data/                  # Normalized meshes per class
    ├── features/              # Extracted features (step 3 outputs)
    ├── database/              # Final merged feature database
    │   ├── all_features.csv
    │   ├── all_features_raw.csv
    │   └── normalization_params.csv
    └── step4.py               # Core mesh_querying(...) logic
```

---

## 🛠️ Installation

### 1. Create a virtual environment
```bash
python -m venv .venv
```

### 2. Activate it  
**Windows:**
```bash
.\.venv\Scriptsctivate
```
**macOS / Linux:**
```bash
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Streamlit App (Recommended)

The app provides an easy interface for uploading a mesh and retrieving similar shapes.

```bash
streamlit run app.py
```

Inside the sidebar:

- **Features CSV:** `MR System/database/all_features.csv`  
- **Normalization params:** `MR System/database/normalization_params.csv`  
- **Meshes folder:** `MR System/data`  
- **Top-K:** choose how many similar meshes to display  

Upload a `.obj`, `.ply`, or `.stl` mesh and the results appear immediately.

---

## ▶️ Running the Pipeline from Terminal
### 1. Check an object (Step 1)
Example of how to run it 
``` bash 
python obj_viewer.py --mesh data/Tool/D01165.obj
```

### 2. Resampling and Normalize shapes (Step 2)
```bash
python data_analysis.py
```

### 3. Extract features (Step 3)
```bash
python "MR System/f_e.py"
```

### 4. Build the feature database (Step 4)
```bash
python "MR System/merge_all.py"
```

### 5. Run a direct query (Step 4)
```bash
python "MR System/query.py"

```

### 6. Scalability & evaluation (Steps 5–6)
```bash
 python scalability.py \
    --features_csv ".\MR System\database\all_features.csv" \
    --bench \
    --tsne \
    --eval \
    --eval_k 10 \
    --k_sweep "1,3,5,10,15,20,25,30" \
    --perplexity 40 \
    --iters 2000 \
    --out "all_classes_steps5&6_all_results"
```

---

## 📝 What the Project Implements

- **Step 1:** Mesh loading & visualization  
- **Step 2:** Preprocessing, resampling, initial normalization  
- **Step 3:**  
  - Full PCA pose normalization  
  - Surface/volume descriptors  
  - A3 / D1–D4 histograms  
- **Step 4:** Querying using normalized feature vectors  
- **Step 5:** Fast retrieval with ANN + t-SNE for visualization  
- **Step 6:** Evaluation using precision/recall  
- **UI:** Modern Streamlit viewer for 3D CBSR  

---

## ✔️ Notes

- Mesh paths in the Streamlit app must point to `MR System/data/<ClassName>/`.
- Evaluation results are stored in:
  - `all_classes_steps5&6_all_results/`
  - `10classes_steps5&6_all_results/`
