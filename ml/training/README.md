# Training model atribut pakaian

`dressa_attribute_classifier.ipynb` melatih CNN dari nol (DressaNet, 2,8 juta parameter) untuk menebak `item_type`, `style`, dan `season` dari foto pakaian. Datanya Fashion Product Images (Small) dari Kaggle.

## Menjalankan di Google Colab

1. Buka https://colab.research.google.com, pilih **File → Upload notebook**, lalu pilih file `.ipynb` ini.
2. Pilih **Runtime → Change runtime type → T4 GPU**.
3. Pilih **Runtime → Run all**, atau jalankan sel satu per satu dengan Shift+Enter supaya bisa membaca penjelasannya.
4. Di akhir, browser akan mengunduh `dressa_attr.onnx` dan `dressa_attr_labels.json`. Simpan keduanya ke `services/ml-engine/weights/`.

Perkiraan waktu dengan T4: memuat data sekitar 2–3 menit, DressaNet 30 epoch sekitar 10 menit, dan pembanding ResNet18 sekitar 5–10 menit.

Hasil di Colab akan hilang kalau runtime terputus. Untuk eksperimen panjang, ganti `WORK_DIR` ke Google Drive (`from google.colab import drive; drive.mount("/content/drive")`).
