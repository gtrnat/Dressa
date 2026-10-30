"""Builds dressa_ml_classifier.ipynb. Run from this folder: python build_notebook.py

The notebook embeds a copy of services/ml-engine/app/core/features.py. Re-run this script
whenever that file changes so training and serving keep computing identical features.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FEATURES_SRC = (ROOT / "services/ml-engine/app/core/features.py").read_text(encoding="utf-8")
CELLS: list[tuple[str, str]] = []
md = lambda t: CELLS.append(("markdown", t.strip("\n")))
code = lambda t: CELLS.append(("code", t.strip("\n")))

md(r"""
# Dressa: Klasifikasi Atribut Pakaian dengan Machine Learning

Notebook ini melatih model **machine learning klasik** (scikit-learn) untuk menebak tiga atribut dari foto satu pakaian:

| Head | Contoh kelas | Dipakai Dressa untuk |
|---|---|---|
| `item_type` | t-shirt, jeans, sneakers, bag | Kategori (atasan/bawahan/sepatu) dan ketebalan baju (WeatherFit) |
| `style` | casual, formal, sporty, ethnic | Kecocokan dengan agenda (CalendarFit) |
| `season` | summer, fall, winter, spring | Penyesuaian ketebalan baju |

### Posisi deep learning dan machine learning

```
foto ─► prepare() ─┬─► [model pretrained, DIBEKUKAN] ─► embedding (512–2048 angka) ─┐
                   └─► [rumus buatan: warna, bentuk, tekstur] ─► 38 angka ─────────┴─► tabel numerik ─► MODEL ML (dilatih di sini)
```

- **Deep learning hanya dipakai sebagai *feature extractor***. Bobot model pretrained tidak diubah sama sekali, dan fungsinya hanya mengubah foto menjadi deretan angka.
- **Yang dilatih adalah model ML klasik pada data numerik**: Logistic Regression, KNN, SVM, Random Forest, Gradient Boosting. Notebook ini membandingkan semuanya, lalu memilih yang terbaik untuk setiap head.

### Alur notebook
1. Setup
2. Dataset dan pemetaan label
3. Ekstraksi fitur (sekali saja, hasilnya disimpan ke file)
4. Eksplorasi fitur
5. Perbandingan ekstraktor
6. Perbandingan set fitur (ablation)
7. Perbandingan algoritma ML dan tuning hyperparameter
8. Evaluasi akhir di test set, termasuk baseline
9. Analisis: cross-validation, feature importance, contoh kesalahan
10. Export model ke ml-engine

**Sebelum mulai di Colab:** pilih **Runtime → Change runtime type → T4 GPU**. GPU hanya dipakai untuk mempercepat ekstraksi fitur. Training model ML sendiri berjalan di CPU.
""")

md(r"""
## 1. Setup

Library yang dipakai:
- **scikit-learn**: library ML utama. Versinya dikunci (`1.9.1`) karena file model hanya bisa dibuka dengan versi yang sama di ml-engine.
- **fastembed**: menjalankan model pretrained (CLIP dan lainnya) dalam format ONNX. ml-engine Dressa memakai library yang sama, sehingga angka fitur saat training dan saat aplikasi berjalan identik. Versi `-gpu` memakai GPU kalau tersedia.
""")

code(r"""
!pip install -q "scikit-learn==1.9.1" "fastembed-gpu==0.9.1" kagglehub
""")

code(r"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from PIL import Image
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

assert sklearn.__version__ == "1.9.1", f"scikit-learn {sklearn.__version__}: jalankan sel pip di atas lalu Runtime -> Restart session"
import onnxruntime as ort
print("scikit-learn", sklearn.__version__, "| ONNX providers:", ort.get_available_providers())
""")

md(r"""
### Konfigurasi

- `cap_per_class`: jumlah foto maksimum per kelas `item_type`. Dataset sangat tidak seimbang (t-shirt sekitar 7000, skirt sekitar 130). Pembatasan ini mengurangi ketimpangan sekaligus mempercepat ekstraksi fitur.
- `extractor`: model pretrained yang dipakai. Bagian 5 akan memilihnya otomatis dari `extractor_candidates`.
- `fast_mode`: grid hyperparameter yang lebih kecil, untuk percobaan cepat.
""")

code(r"""
CONFIG = {
    "seed": 42,
    "cap_per_class": 1500,
    "extractor": "Qdrant/clip-ViT-B-32-vision",  # default; bagian 5 bisa menggantinya
    "extractor_candidates": [
        "Qdrant/clip-ViT-B-32-vision",       # CLIP (2021), 512 dimensi, sudah dipakai ml-engine
        "Qdrant/resnet50-onnx",              # CNN ImageNet (2016), 2048 dimensi
        "google/siglip2-base-patch16-224",   # SigLIP 2 (2025), 768 dimensi
    ],
    "extractor_compare_n": 4000,  # jumlah foto untuk perbandingan ekstraktor
    "fast_mode": False,
    "run_cv": True,
}
SEED = CONFIG["seed"]
np.random.seed(SEED)
WORK_DIR = Path("/content/dressa_ml" if IN_COLAB else "work")
WORK_DIR.mkdir(parents=True, exist_ok=True)
""")

# ---------------------------------------------------------------------------
md(r"""
## 2. Dataset

Dataset yang dipakai adalah [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) dari Kaggle: sekitar 44 ribu foto produk Myntra berlatar putih, dengan label di `styles.csv`. Dataset ini bisa diunduh tanpa login. Kalau muncul error autentikasi, isi `KAGGLE_USERNAME` dan `KAGGLE_KEY` di menu **Secrets** Colab.
""")

code(r"""
import kagglehub

DATA_DIR = Path(os.environ.get("DRESSA_DATA_DIR") or kagglehub.dataset_download("paramaggarwal/fashion-product-images-small"))
CSV_PATH = next(DATA_DIR.rglob("styles.csv"))
IMG_DIR = next(p for p in DATA_DIR.rglob("images") if p.is_dir())
df = pd.read_csv(CSV_PATH, on_bad_lines="skip")
df = df[df["id"].apply(lambda i: (IMG_DIR / f"{i}.jpg").exists())].reset_index(drop=True)
print(len(df), "foto dengan gambar")
""")

md(r"""
### Pemetaan label

`articleType` di dataset punya lebih dari 140 nilai. Tipe yang mirip digabung, misalnya `Casual Shoes` + `Sports Shoes` → `sneakers`. Tipe yang tidak relevan untuk lemari pakaian dibuang (jam, parfum, perhiasan, pakaian dalam).

Untuk `style` dan `season`, label yang kosong atau langka diberi nilai **-1** (tidak diketahui). Setiap head dilatih **hanya dengan baris yang labelnya diketahui**.

**Catatan untuk laporan:** `season` adalah musim katalog penjualan, bukan tebal-tipisnya bahan, jadi label ini hanya proksi kasar.
""")

code(r"""
ITEM_TYPE_MAP = {
    "t-shirt": ["Tshirts", "Lounge Tshirts"], "shirt": ["Shirts"], "top": ["Tops", "Tunics", "Camisoles"],
    "kurta": ["Kurtas", "Kurtis", "Kurta Sets"], "sweatshirt": ["Sweatshirts"], "sweater": ["Sweaters"],
    "jacket": ["Jackets", "Rain Jacket", "Blazers", "Waistcoat", "Nehru Jackets"],
    "jeans": ["Jeans", "Jeggings"], "trousers": ["Trousers", "Capris"], "shorts": ["Shorts", "Lounge Shorts"],
    "track pants": ["Track Pants", "Lounge Pants"], "leggings": ["Leggings", "Tights"], "skirt": ["Skirts"],
    "dress": ["Dresses", "Jumpsuit", "Rompers"], "sneakers": ["Casual Shoes", "Sports Shoes"],
    "formal shoes": ["Formal Shoes"], "heels": ["Heels"], "flats": ["Flats"],
    "sandals": ["Sandals", "Sports Sandals", "Flip Flops"],
    "bag": ["Handbags", "Backpacks", "Clutches", "Duffel Bag", "Laptop Bag", "Messenger Bag", "Rucksacks"],
    "cap": ["Caps", "Hat"], "belt": ["Belts"], "scarf": ["Scarves", "Stoles", "Mufflers"],
}
ITEM_TYPE_TO_CATEGORY = {
    **dict.fromkeys(["t-shirt", "shirt", "top", "kurta", "sweatshirt", "sweater"], "TOP"), "jacket": "OUTERWEAR",
    **dict.fromkeys(["jeans", "trousers", "shorts", "track pants", "leggings", "skirt"], "BOTTOM"), "dress": "DRESS",
    **dict.fromkeys(["sneakers", "formal shoes", "heels", "flats", "sandals"], "FOOTWEAR"),
    **dict.fromkeys(["bag", "cap", "belt", "scarf"], "ACCESSORY"),
}
STYLE_MAP = {"Casual": "casual", "Formal": "formal", "Sports": "sporty", "Ethnic": "ethnic"}
SEASON_MAP = {"Summer": "summer", "Fall": "fall", "Winter": "winter", "Spring": "spring"}
CLASSES = {"item_type": list(ITEM_TYPE_MAP), "style": list(STYLE_MAP.values()), "season": list(SEASON_MAP.values())}
HEADS = list(CLASSES)

article_to_item = {a: k for k, v in ITEM_TYPE_MAP.items() for a in v}
data = df[df["articleType"].isin(article_to_item)].copy()
idx = lambda col, mapping, h: data[col].map(lambda v: {c: i for i, c in enumerate(CLASSES[h])}.get(mapping.get(v), -1)).astype(int)
data["item_type"] = idx("articleType", article_to_item, "item_type")
data["style"] = idx("usage", STYLE_MAP, "style")
data["season"] = idx("season", SEASON_MAP, "season")

# Batasi jumlah foto per kelas (sampling acak, tetap reproducible dengan seed)
data = (data.sample(frac=1.0, random_state=SEED)       # acak urutan
            .groupby("item_type").head(CONFIG["cap_per_class"])  # ambil maksimal N per kelas
            .reset_index(drop=True))
print(f"Dipakai {len(data)} foto, {len(CLASSES['item_type'])} kelas item_type")
counts = data["item_type"].value_counts().sort_index()
plt.figure(figsize=(12, 3.2)); plt.bar([CLASSES["item_type"][i] for i in counts.index], counts.values)
plt.xticks(rotation=60, ha="right"); plt.title("Jumlah foto per item_type (setelah cap)"); plt.tight_layout(); plt.show()
for h in ("style", "season"):
    vc = data[h].map(lambda i: CLASSES[h][i] if i >= 0 else "(tidak diketahui)").value_counts()
    print(h, vc.to_dict())
""")

md(r"""
### Split train / validation / test (70 / 15 / 15)

- **Train**: data untuk melatih model.
- **Validation**: data untuk memilih algoritma, hyperparameter, ekstraktor, dan set fitur.
- **Test**: dipakai **satu kali di akhir** untuk angka laporan. Kalau test set ikut dipakai untuk memilih setelan, hasilnya jadi terlalu optimis.

`stratify` menjaga proporsi setiap kelas tetap sama di ketiga bagian.
""")

code(r"""
train_df, rest = train_test_split(data, test_size=0.30, stratify=data["item_type"], random_state=SEED)
val_df, test_df = train_test_split(rest, test_size=0.50, stratify=rest["item_type"], random_state=SEED)
SPLITS = {"train": train_df.reset_index(drop=True), "val": val_df.reset_index(drop=True), "test": test_df.reset_index(drop=True)}
Y = {s: d[HEADS].to_numpy() for s, d in SPLITS.items()}
print({s: len(d) for s, d in SPLITS.items()})
""")

# ---------------------------------------------------------------------------
md(r"""
## 3. Ekstraksi fitur

Model ML tidak bisa membaca foto secara langsung. Ia butuh **fitur**, yaitu angka-angka yang mewakili isi foto. Ada dua jenis fitur:

**a. Embedding dari model pretrained (deep learning, dibekukan)**

Model seperti CLIP sudah dilatih orang lain dengan ratusan juta foto. Kita hanya memakainya untuk mengubah foto menjadi vektor, misalnya 512 angka, tanpa melatihnya ulang. Foto yang mirip menghasilkan vektor yang berdekatan. Vektor dinormalisasi ke panjang 1 (L2-normalize).

**b. Fitur buatan tangan (*handcrafted*, 38 angka)**

Fitur ini dihitung dengan rumus biasa dari piksel pakaian saja. Latar putih dipisahkan dulu dengan *flood fill* dari tepi gambar.
- **Warna** (22): rata-rata dan sebaran warna di ruang CIELAB, histogram hue 12 bin, rasio piksel abu-abu, gelap, dan terang, serta saturasi.
- **Bentuk** (14): luas, rasio lebar/tinggi, kepadatan, titik pusat, simetri, profil lebar di bagian atas, tengah, dan bawah, serta **celah di antara kaki** (khas celana).
- **Tekstur** (2): kepadatan tepi (proksi motif atau pola).

Fungsi `prepare()` (latar putih, persegi, 80×80) **sama persis** dengan yang dipakai ml-engine. Sel di bawah ini adalah salinan otomatis dari `services/ml-engine/app/core/features.py`. **Jangan diubah di sini**, karena kalau berbeda, model akan menerima angka yang berbeda saat dipakai di aplikasi.
""")

code("# === SALINAN OTOMATIS dari services/ml-engine/app/core/features.py (jangan diubah di sini) ===\n" + FEATURES_SRC)

code(r"""
def load_prepared(split):
    # Semua foto split -> array uint8 (N, 80, 80, 3), disimpan ke cache
    cache = WORK_DIR / f"prepared_{split}_{len(SPLITS[split])}_v{FEATURE_VERSION}.npy"
    if cache.exists():
        return np.load(cache)
    def one(i):
        with Image.open(IMG_DIR / f"{i}.jpg") as im:
            return np.asarray(prepare(im))
    with ThreadPoolExecutor(8) as pool:
        arr = np.stack(list(pool.map(one, SPLITS[split]["id"])))
    np.save(cache, arr)
    return arr

def extract_handcrafted(split):
    cache = WORK_DIR / f"hc_{split}_{len(SPLITS[split])}_v{FEATURE_VERSION}.npy"
    if cache.exists():
        return np.load(cache)
    with ThreadPoolExecutor(8) as pool:
        X = np.stack(list(pool.map(lambda a: handcrafted(Image.fromarray(a)), PREPARED[split])))
    np.save(cache, X)
    return X

_embedders, EXTRACT_MS = {}, {}
def extract_embeddings(model_name, split, rows=None):
    # rows: subset index (untuk perbandingan ekstraktor); None = semua
    from fastembed import ImageEmbedding
    n = len(PREPARED[split]) if rows is None else len(rows)
    safe = model_name.replace("/", "_")
    cache = WORK_DIR / f"emb_{safe}_{split}_{n}_{'all' if rows is None else 'sub'}_v{FEATURE_VERSION}.npy"
    if cache.exists():
        return np.load(cache)
    if model_name not in _embedders:
        _embedders[model_name] = ImageEmbedding(model_name=model_name)  # pakai GPU kalau tersedia
    model = _embedders[model_name]
    arrs = PREPARED[split] if rows is None else PREPARED[split][rows]
    t0, out = time.time(), []
    for s in range(0, len(arrs), 512):
        out.extend(model.embed([Image.fromarray(a) for a in arrs[s:s + 512]], batch_size=64))
        print(f"\r  {model_name}: {min(s + 512, len(arrs))}/{len(arrs)} ({time.time() - t0:.0f}s)", end="")
    print()
    EXTRACT_MS[model_name] = (time.time() - t0) / len(arrs) * 1000  # tanpa waktu memuat model
    X = np.stack([l2_normalize(v) for v in out]).astype(np.float32)
    np.save(cache, X)
    return X

t0 = time.time()
PREPARED = {s: load_prepared(s) for s in SPLITS}
HC = {s: extract_handcrafted(s) for s in SPLITS}
print({s: HC[s].shape for s in SPLITS}, f"{time.time() - t0:.0f}s")
""")

# ---------------------------------------------------------------------------
md(r"""
## 4. Eksplorasi fitur

Sebelum melatih model, kita lihat dulu apakah fitur buatan bisa membedakan kelas. Contohnya, `gap_bottom` (celah di antara kaki) seharusnya tinggi untuk celana dan rendah untuk kaos.
""")

code(r"""
hc_df = pd.DataFrame(HC["train"], columns=HANDCRAFTED_NAMES)
hc_df["item_type"] = [CLASSES["item_type"][i] for i in Y["train"][:, 0]]
show = ["bbox_aspect", "gap_bottom", "width_top", "lab_L_mean"]
order = hc_df.groupby("item_type")["gap_bottom"].mean().sort_values().index
fig, axes = plt.subplots(1, len(show), figsize=(18, 5), sharey=True)
for ax, f in zip(axes, show):
    ax.barh(order, hc_df.groupby("item_type")[f].mean().loc[order])
    ax.set_title(f"rata-rata {f}")
plt.tight_layout(); plt.show()

fig, axes = plt.subplots(1, 8, figsize=(16, 2.4))
for ax, j in zip(axes, np.random.choice(len(PREPARED["train"]), 8, replace=False)):
    ax.imshow(PREPARED["train"][j]); ax.axis("off")
    ax.set_title(f"{CLASSES['item_type'][Y['train'][j, 0]]}\ngap={HC['train'][j, HANDCRAFTED_NAMES.index('gap_bottom')]:.2f}", fontsize=8)
plt.suptitle("Contoh hasil prepare() (80x80, latar putih)"); plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------------------
md(r"""
## 5. Perbandingan ekstraktor

Mana model pretrained yang menghasilkan fitur terbaik untuk pakaian? Untuk menghemat waktu, setiap kandidat dicoba pada sebagian data train (`extractor_compare_n` foto) dengan model sederhana dan cepat (Logistic Regression). Skornya diukur di validation set.

Ekstraktor terbaik dipakai untuk semua langkah berikutnya. Pertimbangkan juga biayanya: model yang lebih besar membuat aplikasi lebih lambat. Kolom `ms/foto` menunjukkan waktu ekstraksi per foto di runtime ini.

**Perkiraan waktu:** dengan GPU T4, ekstraksi seluruh data (sekitar 17 ribu foto) memakan beberapa menit per ekstraktor. Tanpa GPU (CPU saja), prosesnya bisa sekitar 1 jam per ekstraktor. Kalau terpaksa memakai CPU, kurangi `extractor_candidates` menjadi satu model saja.
""")

code(r"""
def metrics(y_true, y_pred):
    return {"acc": accuracy_score(y_true, y_pred), "macro_f1": f1_score(y_true, y_pred, average="macro"),
            "balanced_acc": balanced_accuracy_score(y_true, y_pred)}

def labeled(split, h, X):
    m = Y[split][:, HEADS.index(h)] >= 0
    return X[m], Y[split][m, HEADS.index(h)]

def make_pipeline(estimator, n_emb, n_hc, pca=None):
    # Embedding: StandardScaler (+ PCA opsional). Handcrafted: StandardScaler.
    parts = []
    if n_emb:
        steps = [("scale", StandardScaler())] + ([("pca", PCA(n_components=pca, random_state=SEED))] if pca else [])
        parts.append(("emb", Pipeline(steps), list(range(n_emb))))
    if n_hc:
        parts.append(("hc", StandardScaler(), list(range(n_emb, n_emb + n_hc))))
    return Pipeline([("features", ColumnTransformer(parts)), ("model", estimator)])

logreg = lambda C=1.0: LogisticRegression(C=C, max_iter=3000, class_weight="balanced")

rng = np.random.default_rng(SEED)
sub = np.sort(rng.choice(len(SPLITS["train"]), min(CONFIG["extractor_compare_n"], len(SPLITS["train"])), replace=False))
rows = []
for name in CONFIG["extractor_candidates"]:
    E_tr, E_va = extract_embeddings(name, "train", sub), extract_embeddings(name, "val")
    ms = EXTRACT_MS.get(name)  # None kalau diambil dari cache
    row = {"extractor": name, "dim": E_tr.shape[1], "ms/foto": round(ms, 1) if ms else None}
    for h in HEADS:
        y_sub = Y["train"][sub, HEADS.index(h)]
        Xtr, ytr = E_tr[y_sub >= 0], y_sub[y_sub >= 0]
        Xva, yva = labeled("val", h, E_va)
        pipe = make_pipeline(logreg(), E_tr.shape[1], 0).fit(Xtr, ytr)
        row[f"{h} F1"] = round(metrics(yva, pipe.predict(Xva))["macro_f1"], 3)
    rows.append(row)
    print(row)
ext_table = pd.DataFrame(rows)
ext_table["rata-rata F1"] = ext_table[[f"{h} F1" for h in HEADS]].mean(1).round(3)
print(ext_table.to_string(index=False))
CONFIG["extractor"] = ext_table.sort_values("rata-rata F1", ascending=False).iloc[0]["extractor"]
print("\nEkstraktor terpilih:", CONFIG["extractor"])
""")

md(r"""
Sekarang embedding diekstrak untuk **semua** foto dengan ekstraktor terpilih. Hasilnya disimpan, jadi kalau sel ini dijalankan ulang, prosesnya langsung selesai.
""")

code(r"""
EXTRACTOR = CONFIG["extractor"]
EMB = {s: extract_embeddings(EXTRACTOR, s) for s in SPLITS}
N_EMB, N_HC = EMB["train"].shape[1], HC["train"].shape[1]
print({s: EMB[s].shape for s in SPLITS})
""")

code(r"""
# Visualisasi: embedding 512+ dimensi diproyeksikan ke 2D dengan t-SNE (2000 foto acak)
pick = np.random.default_rng(1).choice(len(EMB["train"]), min(2000, len(EMB["train"])), replace=False)
xy = TSNE(n_components=2, init="pca", random_state=SEED, perplexity=30).fit_transform(PCA(50, random_state=SEED).fit_transform(EMB["train"][pick]))
cats = np.array([ITEM_TYPE_TO_CATEGORY[CLASSES["item_type"][i]] for i in Y["train"][pick, 0]])
plt.figure(figsize=(8, 6.5))
for c in sorted(set(cats)):
    plt.scatter(*xy[cats == c].T, s=6, label=c)
plt.legend(markerscale=3); plt.title(f"t-SNE embedding {EXTRACTOR.split('/')[-1]} (warna = kategori)"); plt.axis("off"); plt.show()
""")

md(r"""
Kalau titik-titik dengan warna yang sama berkumpul, artinya embedding sudah memisahkan kategori dengan baik bahkan sebelum ada training. Tugas model ML adalah menarik garis batas di antara kelompok-kelompok itu.
""")

# ---------------------------------------------------------------------------
md(r"""
## 6. Perbandingan set fitur (ablation)

Apakah fitur buatan tangan membantu? Model yang sama (Logistic Regression) dilatih dengan tiga set fitur:
1. embedding saja
2. handcrafted saja
3. gabungan keduanya

Set fitur dengan rata-rata macro-F1 validation tertinggi dipakai untuk langkah berikutnya.
""")

code(r"""
FEATURE_SETS = {
    "embedding": (lambda s: EMB[s], N_EMB, 0),
    "handcrafted": (lambda s: HC[s], 0, N_HC),
    "embedding+handcrafted": (lambda s: np.hstack([EMB[s], HC[s]]), N_EMB, N_HC),
}
rows = []
for fs, (getX, ne, nh) in FEATURE_SETS.items():
    row = {"feature_set": fs, "n_features": ne + nh}
    for h in HEADS:
        Xtr, ytr = labeled("train", h, getX("train")); Xva, yva = labeled("val", h, getX("val"))
        row[f"{h} F1"] = round(metrics(yva, make_pipeline(logreg(), ne, nh).fit(Xtr, ytr).predict(Xva))["macro_f1"], 3)
    rows.append(row); print(row)
fs_table = pd.DataFrame(rows)
fs_table["rata-rata F1"] = fs_table[[f"{h} F1" for h in HEADS]].mean(1).round(3)
print(fs_table.to_string(index=False))
FEATURE_SET = fs_table.sort_values("rata-rata F1", ascending=False).iloc[0]["feature_set"]
getX, NE, NH = FEATURE_SETS[FEATURE_SET]
X = {s: getX(s) for s in SPLITS}
print("\nSet fitur terpilih:", FEATURE_SET, X["train"].shape)
""")

# ---------------------------------------------------------------------------
md(r"""
## 7. Perbandingan algoritma ML

Setiap head dilatih dengan beberapa algoritma. Setiap algoritma dicoba dengan beberapa nilai hyperparameter (*grid search* manual di validation set).

| Algoritma | Cara kerja singkat | Hyperparameter yang dicoba |
|---|---|---|
| **Logistic Regression** | Mencari batas linear di antara kelas, lalu mengeluarkan probabilitas | `C` (kekuatan regularisasi; makin kecil, makin sederhana modelnya) |
| **KNN** | Mencari *k* foto training dengan fitur paling mirip, lalu mengambil suara terbanyak | `k` |
| **SVM (RBF)** | Mencari batas dengan margin terlebar. Kernel RBF membuat batasnya bisa melengkung | `C` |
| **Random Forest** | Ratusan pohon keputusan, masing-masing dilatih dengan data dan fitur acak, lalu voting | `min_samples_leaf` |
| **Gradient Boosting** | Pohon-pohon kecil dibuat berurutan; setiap pohon baru memperbaiki kesalahan pohon sebelumnya | `learning_rate` |

Detail tambahan:
- **StandardScaler**: setiap kolom dibuat ber-mean 0 dan ber-std 1, supaya fitur dengan skala besar tidak mendominasi. Ini penting untuk LogReg, KNN, dan SVM.
- **PCA**: untuk KNN, SVM, dan model pohon, embedding dikompres menjadi 64–128 komponen utama. Training jadi jauh lebih cepat, dan noise ikut berkurang.
- **class_weight="balanced"**: kesalahan pada kelas kecil diberi bobot lebih berat, supaya model tidak hanya pandai menebak kelas besar.
- **Metrik pemilihan: macro-F1**, yaitu rata-rata F1 semua kelas dengan bobot sama. Accuracy biasa bisa tinggi hanya karena model pandai menebak kelas yang dominan.
""")

code(r"""
fast = CONFIG["fast_mode"]
CANDIDATES = {
    "LogisticRegression": (lambda: logreg(), None, [{"model__C": c} for c in ([1.0] if fast else [0.1, 1.0, 10.0])]),
    "KNN": (lambda: KNeighborsClassifier(weights="distance", n_jobs=-1), 128,
            [{"model__n_neighbors": k} for k in ([15] if fast else [5, 15, 31])]),
    "SVM (RBF)": (lambda: SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED), 128,
                  [{"model__C": c} for c in ([3.0] if fast else [1.0, 10.0])]),
    "RandomForest": (lambda: RandomForestClassifier(n_estimators=300, max_features="sqrt", class_weight="balanced_subsample",
                                                    n_jobs=-1, random_state=SEED), 64,
                     [{"model__min_samples_leaf": m} for m in ([1] if fast else [1, 3])]),
    "GradientBoosting": (lambda: HistGradientBoostingClassifier(max_iter=300, early_stopping=True, class_weight="balanced",
                                                                random_state=SEED), 64,
                         [{"model__learning_rate": lr} for lr in ([0.1] if fast else [0.05, 0.15])]),
}

results, best = [], {}
for h in HEADS:
    Xtr, ytr = labeled("train", h, X["train"]); Xva, yva = labeled("val", h, X["val"])
    for name, (factory, pca, grid) in CANDIDATES.items():
        for params in grid:
            pipe = make_pipeline(factory(), NE, NH, pca=pca if NE else None).set_params(**params)
            t0 = time.time(); pipe.fit(Xtr, ytr); fit_s = time.time() - t0
            t0 = time.time(); pred = pipe.predict(Xva); ms = (time.time() - t0) / len(Xva) * 1000
            m = metrics(yva, pred)
            results.append({"head": h, "model": name, "params": json.dumps({k.split("__")[1]: v for k, v in params.items()}),
                             **{k: round(v, 3) for k, v in m.items()}, "fit_s": round(fit_s, 1), "ms/foto": round(ms, 3)})
            print(f"[{h}] {name:18s} {results[-1]['params']:28s} macro-F1={m['macro_f1']:.3f} acc={m['acc']:.3f} ({fit_s:.0f}s)")
            if m["macro_f1"] > best.get(h, {}).get("macro_f1", -1):
                best[h] = {"name": name, "params": params, "pca": pca, "factory": factory, "macro_f1": m["macro_f1"]}
results = pd.DataFrame(results)
for h in HEADS:
    print(f"\n=== {h}: terbaik = {best[h]['name']} {best[h]['params']} (val macro-F1 {best[h]['macro_f1']:.3f})")
""")

code(r"""
pivot = results.groupby(["head", "model"])["macro_f1"].max().unstack(0)[HEADS]
ax = pivot.plot.bar(figsize=(11, 4), rot=0)
ax.set_ylabel("validation macro-F1 (setelan terbaik)"); ax.set_title(f"Perbandingan algoritma ({FEATURE_SET}, {EXTRACTOR.split('/')[-1]})")
plt.tight_layout(); plt.show()
print(pivot.round(3))
""")

# ---------------------------------------------------------------------------
md(r"""
## 8. Evaluasi akhir di test set

Model terbaik setiap head dilatih ulang dengan **train + validation**, karena datanya lebih banyak. Setelah itu model dievaluasi **sekali** di test set.

Sebagai pembanding, ada dua baseline:
- **Dummy (most frequent)**: selalu menebak kelas terbanyak. Model yang tidak bisa mengalahkan baseline ini tidak belajar apa-apa.
- **CLIP zero-shot**: tanpa training sama sekali. Embedding foto dibandingkan dengan embedding kalimat seperti "a photo of a t-shirt". Cara ini dipakai Dressa sebelum ada model ML.
""")

code(r"""
def trainval(h):
    Xa, ya = labeled("train", h, X["train"]); Xb, yb = labeled("val", h, X["val"])
    return np.vstack([Xa, Xb]), np.concatenate([ya, yb])

FINAL, TEST_RESULTS, TEST_PRED = {}, {}, {}
for h in HEADS:
    b = best[h]
    Xtv, ytv = trainval(h)
    FINAL[h] = make_pipeline(b["factory"](), NE, NH, pca=b["pca"] if NE else None).set_params(**b["params"]).fit(Xtv, ytv)
    Xte, yte = labeled("test", h, X["test"])
    pred = FINAL[h].predict(Xte); TEST_PRED[h] = (yte, pred)
    dummy = DummyClassifier(strategy="most_frequent").fit(Xtv, ytv).predict(Xte)
    TEST_RESULTS[h] = {"model": b["name"], **{k: round(v, 4) for k, v in metrics(yte, pred).items()},
                       "dummy_macro_f1": round(metrics(yte, dummy)["macro_f1"], 4)}

# Baseline CLIP zero-shot (selalu memakai CLIP, apa pun ekstraktor terpilih)
from fastembed import TextEmbedding
clip_test = extract_embeddings("Qdrant/clip-ViT-B-32-vision", "test")
text = TextEmbedding("Qdrant/clip-ViT-B-32-text")
PROMPTS = {"item_type": "a product photo of a {}", "style": "a photo of {} clothing", "season": "clothing for {}"}
for h in HEADS:
    T = np.stack([l2_normalize(v) for v in text.embed([PROMPTS[h].format(c) for c in CLASSES[h]])])
    m = Y["test"][:, HEADS.index(h)] >= 0
    zs = (clip_test[m] @ T.T).argmax(1)
    TEST_RESULTS[h]["clip_zero_shot_macro_f1"] = round(metrics(Y["test"][m, HEADS.index(h)], zs)["macro_f1"], 4)

print(pd.DataFrame(TEST_RESULTS).T.to_string())
""")

code(r"""
for h in HEADS:
    yte, pred = TEST_PRED[h]
    print(f"\n=== {h} ({TEST_RESULTS[h]['model']}) ===")
    print(classification_report(yte, pred, labels=range(len(CLASSES[h])), target_names=CLASSES[h], digits=3, zero_division=0))

def plot_confusion(h, figsize):
    yte, pred = TEST_PRED[h]
    cm = confusion_matrix(yte, pred, labels=range(len(CLASSES[h])), normalize="true")
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES[h])), CLASSES[h], rotation=70, ha="right"); ax.set_yticks(range(len(CLASSES[h])), CLASSES[h])
    ax.set_xlabel("tebakan"); ax.set_ylabel("sebenarnya"); ax.set_title(f"Confusion matrix {h} (per baris)")
    plt.colorbar(im); plt.tight_layout(); plt.show()

plot_confusion("item_type", (11, 9)); plot_confusion("style", (5, 4)); plot_confusion("season", (5, 4))
""")

md(r"""
Cara membaca:
- **Precision** kelas X: dari semua yang ditebak X, berapa persen yang benar.
- **Recall** kelas X: dari semua X yang sebenarnya, berapa persen yang berhasil ditemukan.
- **Confusion matrix**: kotak terang di luar diagonal menunjukkan kelas yang sering tertukar. Contohnya *top* dengan *t-shirt*, atau *flats* dengan *sandals*. Untuk pasangan seperti itu, foto produknya memang sangat mirip.
""")

# ---------------------------------------------------------------------------
md(r"""
## 9. Analisis

### 9a. Cross-validation (5-fold)

Satu kali split train/val bisa saja kebetulan "beruntung". **K-fold cross-validation** membagi data train+val menjadi 5 bagian, lalu melatih dan menguji 5 kali dengan bagian uji yang bergiliran. Hasilnya berupa rata-rata ± standar deviasi, yang lebih bisa dipercaya untuk laporan.
""")

code(r"""
if CONFIG["run_cv"]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for h in HEADS:
        Xtv, ytv = trainval(h)
        t0 = time.time()
        s = cross_val_score(clone(FINAL[h]), Xtv, ytv, cv=cv, scoring="f1_macro", n_jobs=1)
        TEST_RESULTS[h]["cv_macro_f1_mean"], TEST_RESULTS[h]["cv_macro_f1_std"] = round(s.mean(), 4), round(s.std(), 4)
        print(f"{h:10s} {TEST_RESULTS[h]['model']:18s} CV macro-F1 = {s.mean():.3f} ± {s.std():.3f}  ({time.time() - t0:.0f}s)")
""")

md(r"""
### 9b. Feature importance (permutation importance)

Pertanyaannya: **fitur mana yang paling dipakai model?** Caranya, acak nilai satu kelompok fitur di test set (barisnya diacak supaya hubungannya dengan label hilang), lalu ukur seberapa besar macro-F1 turun. Makin besar penurunannya, makin penting kelompok fitur itu. Metode ini bisa dipakai untuk algoritma apa pun.
""")

code(r"""
def permutation_drop(h, cols, repeats=3):
    Xte, yte = labeled("test", h, X["test"])
    base = f1_score(yte, FINAL[h].predict(Xte), average="macro")
    rng = np.random.default_rng(SEED); drops = []
    for _ in range(repeats):
        Xp = Xte.copy(); Xp[:, cols] = Xp[rng.permutation(len(Xp))][:, cols]
        drops.append(base - f1_score(yte, FINAL[h].predict(Xp), average="macro"))
    return float(np.mean(drops))

off = NE  # kolom handcrafted dimulai setelah embedding
groups = {}
if NE: groups["embedding"] = list(range(NE))
if NH:
    groups["warna"] = [off + i for i, n in enumerate(HANDCRAFTED_NAMES) if n.startswith(("lab_", "hue_", "gray", "dark", "bright", "saturation"))]
    groups["bentuk"] = [off + i for i, n in enumerate(HANDCRAFTED_NAMES) if n.startswith(("fg_", "bbox", "centroid", "sym", "width", "height", "gap"))]
    groups["tekstur"] = [off + i for i, n in enumerate(HANDCRAFTED_NAMES) if n.startswith("edge")]
imp = pd.DataFrame({h: {g: permutation_drop(h, c) for g, c in groups.items()} for h in HEADS})
print("Penurunan macro-F1 kalau kelompok fitur diacak:\n", imp.round(3))
imp.plot.bar(figsize=(8, 3.5), rot=0, title="Permutation importance per kelompok fitur"); plt.ylabel("penurunan macro-F1"); plt.tight_layout(); plt.show()

if NH:
    single = pd.Series({n: permutation_drop("item_type", [off + i], repeats=2) for i, n in enumerate(HANDCRAFTED_NAMES)})
    single.sort_values().tail(15).plot.barh(figsize=(7, 5), title="15 fitur handcrafted terpenting (item_type)")
    plt.xlabel("penurunan macro-F1"); plt.tight_layout(); plt.show()
""")

md(r"""
### 9c. Contoh kesalahan

Melihat foto yang salah ditebak membantu memahami **kenapa** model salah: labelnya memang ambigu, kelasnya sangat mirip, atau fiturnya kurang informatif.
""")

code(r"""
yte, pred = TEST_PRED["item_type"]
m = Y["test"][:, 0] >= 0
imgs = PREPARED["test"][m]
proba = FINAL["item_type"].predict_proba(X["test"][m]).max(1)
wrong = np.where(pred != yte)[0]
show = wrong[np.argsort(-proba[wrong])][:16]
fig, axes = plt.subplots(2, 8, figsize=(16, 5))
for ax, j in zip(axes.flat, show):
    ax.imshow(imgs[j]); ax.axis("off")
    ax.set_title(f"asli: {CLASSES['item_type'][yte[j]]}\ntebak: {CLASSES['item_type'][pred[j]]} ({proba[j]:.0%})", fontsize=8)
plt.suptitle(f"Kesalahan paling 'yakin': {len(wrong)} dari {len(yte)} foto test salah"); plt.tight_layout(); plt.show()
""")

md(r"""
### Coba foto sendiri (opsional)

Untuk hasil terbaik, upload foto yang **background-nya sudah dihapus**. Contohnya `cutout.png` dari Dressa, di folder `services/api-gateway/uploads/<user>/<item>/`. Alasannya, dataset training berlatar putih, sehingga foto dengan latar ramai kemungkinan besar ditebak salah. Perbedaan seperti ini disebut *domain gap*.
""")

code(r"""
from fastembed import ImageEmbedding

def predict_photo(img):
    p = prepare(img)
    parts = []
    if NE:
        emb_model = _embedders.get(EXTRACTOR) or ImageEmbedding(model_name=EXTRACTOR)
        parts.append(l2_normalize(next(iter(emb_model.embed([p])))))
    if NH:
        parts.append(handcrafted(p))
    x = np.concatenate(parts)[None]
    out = {}
    for h in HEADS:
        pr = FINAL[h].predict_proba(x)[0]
        top = np.argsort(-pr)[:3]
        out[h] = [(CLASSES[h][int(FINAL[h].classes_[k])], float(pr[k])) for k in top]
    return p, out

if IN_COLAB:
    import io
    from google.colab import files
    for name, content in files.upload().items():
        p, out = predict_photo(Image.open(io.BytesIO(content)))
        plt.figure(figsize=(2.5, 2.5)); plt.imshow(p); plt.axis("off"); plt.title("yang dilihat model"); plt.show()
        for h, top in out.items():
            print(f"  {h:10s}", ", ".join(f"{c} {v:.0%}" for c, v in top))
""")

# ---------------------------------------------------------------------------
md(r"""
## 10. Export ke ml-engine

Semua model terbaik disimpan dalam **satu file** `dressa_ml.joblib`. Isinya model per head (lengkap dengan scaler dan PCA di dalam pipeline), nama ekstraktor, set fitur, daftar kelas, versi scikit-learn, dan skor test. ml-engine memeriksa kecocokan semua informasi ini saat start.

`dressa_ml_report.json` berisi ringkasan hasil dalam format yang mudah dibaca, untuk keperluan laporan.

**Keamanan:** file `.joblib` bisa menjalankan kode saat dibuka, jadi hanya buka file yang kamu latih sendiri.
""")

code(r"""
bundle = {
    "format": "dressa-ml-v1",
    "feature_version": FEATURE_VERSION,
    "handcrafted_names": HANDCRAFTED_NAMES,
    "extractor": EXTRACTOR,
    "feature_set": FEATURE_SET,
    "sklearn_version": sklearn.__version__,
    "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "heads": {h: {"name": best[h]["name"], "params": {k.split("__")[1]: v for k, v in best[h]["params"].items()},
                  "model": FINAL[h], "classes": CLASSES[h]} for h in HEADS},
    "item_type_to_category": ITEM_TYPE_TO_CATEGORY,
    "metrics": TEST_RESULTS,
}
model_path = WORK_DIR / "dressa_ml.joblib"
joblib.dump(bundle, model_path, compress=3)

report = {k: v for k, v in bundle.items() if k != "heads"}
report["heads"] = {h: {k: v for k, v in b.items() if k != "model"} for h, b in bundle["heads"].items()}
report["extractor_comparison"] = ext_table.to_dict("records")
report["feature_set_comparison"] = fs_table.to_dict("records")
report["model_comparison"] = results.to_dict("records")
report["n_samples"] = {s: len(d) for s, d in SPLITS.items()}
report_path = WORK_DIR / "dressa_ml_report.json"
report_path.write_text(json.dumps(report, indent=2, default=str))

# Verifikasi: file bisa dibuka ulang dan prediksinya sama persis
loaded = joblib.load(model_path)
for h in HEADS:
    Xte, _ = labeled("test", h, X["test"])
    same = (loaded["heads"][h]["model"].predict(Xte) == FINAL[h].predict(Xte)).mean()
    print(f"{h:10s} {loaded['heads'][h]['name']:18s} prediksi sama setelah reload: {same:.0%}")
print(f"\n{model_path.name}: {model_path.stat().st_size / 1e6:.1f} MB | ekstraktor={EXTRACTOR} | fitur={FEATURE_SET}")

if IN_COLAB:
    from google.colab import files
    files.download(str(model_path)); files.download(str(report_path))
""")

md(r"""
Simpan `dressa_ml.joblib` ke `services/ml-engine/weights/`, lalu restart ml-engine. Kalau berhasil, `GET /health` akan menampilkan `"attribute_model": "ml-classifier"`.

## Ide eksperimen untuk laporan

- **Tabel utama:** perbandingan algoritma (bagian 7), perbandingan dengan baseline Dummy dan CLIP zero-shot (bagian 8), dan CV mean ± std (bagian 9a).
- **Ablation:** pengaruh ekstraktor (bagian 5), set fitur (bagian 6), PCA (coba ubah angka PCA di `CANDIDATES`), `class_weight` (coba hapus), dan `cap_per_class`.
- **Interpretasi:** permutation importance (bagian 9b), yaitu apakah fitur bentuk seperti `gap_bottom` benar-benar membantu membedakan celana.
- **Domain gap:** kumpulkan 30–50 foto baju sendiri lewat Dressa, beri label manual, lalu bandingkan akurasinya dengan test set Kaggle.
""")


def to_source(text: str) -> list[str]:
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


nb = {
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": to_source(s)} if k == "markdown"
        else {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": to_source(s)}
        for k, s in CELLS
    ],
    "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                 "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 0,
}
out = Path(__file__).with_name("dressa_ml_classifier.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", out, len(CELLS), "cells")
