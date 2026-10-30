# Training model atribut pakaian (machine learning)

`dressa_ml_classifier.ipynb` melatih model **machine learning klasik** (scikit-learn) untuk menebak `item_type`, `style`, dan `season` dari foto pakaian:

- **Fitur:** embedding dari model pretrained yang dibekukan (CLIP / ResNet50 / SigLIP 2, dipilih otomatis), ditambah 38 fitur buatan tangan (warna, bentuk, tekstur).
- **Model:** Logistic Regression, KNN, SVM, Random Forest, dan Gradient Boosting dibandingkan. Yang terbaik dipilih untuk setiap head.
- **Data:** Fashion Product Images (Small) dari Kaggle.

Deep learning hanya dipakai untuk ekstraksi fitur dan tidak dilatih. Model yang dilatih adalah model ML pada data numerik.

## Menjalankan di Google Colab

1. Buka https://colab.research.google.com, pilih **File → Upload notebook**, lalu pilih `dressa_ml_classifier.ipynb`.
2. Pilih **Runtime → Change runtime type → T4 GPU**. GPU hanya mempercepat ekstraksi fitur.
3. Jalankan sel satu per satu dengan Shift+Enter, atau pilih **Runtime → Run all**.
4. Kalau sel pengecekan versi gagal setelah `pip install`, pilih **Runtime → Restart session**, lalu jalankan ulang dari sel kedua.
5. Di akhir, browser mengunduh `dressa_ml.joblib` (model) dan `dressa_ml_report.json` (ringkasan hasil untuk laporan). Simpan `dressa_ml.joblib` ke `services/ml-engine/weights/`, lalu restart ml-engine.

Hasil ekstraksi fitur disimpan di `/content/dressa_ml`, jadi sel yang dijalankan ulang langsung selesai. Kalau runtime Colab terputus, folder itu ikut hilang. Supaya aman, ganti `WORK_DIR` ke Google Drive (`from google.colab import drive; drive.mount("/content/drive")`).

## Mengubah fitur

Fungsi fitur hanya ditulis di satu tempat, yaitu `services/ml-engine/app/core/features.py`. Notebook berisi salinan otomatisnya. Kalau file itu diubah:

1. Naikkan `FEATURE_VERSION` di file tersebut.
2. Jalankan `python build_notebook.py` dari folder ini untuk membuat ulang notebook.
3. Latih ulang modelnya.

ml-engine menolak model yang `FEATURE_VERSION`-nya berbeda.
