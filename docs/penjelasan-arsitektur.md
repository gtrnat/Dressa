# Penjelasan Arsitektur Dressa

Dokumen ini menjelaskan apa saja yang sudah dipakai di Dressa, untuk apa, dan bagaimana bagian-bagiannya saling terhubung. Penjelasan training model ada di notebook `ml/training/dressa_attribute_classifier.ipynb`.

## Gambaran besar

```
 HP / Browser
     │  POST /items (foto)
     ▼
 api-gateway  (Node.js + TypeScript, port 3000)
     │  1. kirim foto ───────────────► ml-engine /process-image
     │                                   hapus background + tebak tag
     │  2. simpan foto & cutout ke disk (folder uploads/)
     │  3. kirim cutout ─────────────► ml-engine /embed ──► Qdrant (vektor)
     │  4. simpan data item ─────────► PostgreSQL (tabel wardrobe_items)
     ▼
 respons: data item + needsReview
```

Ada dua service yang dipisah karena bebannya beda:

- **api-gateway** melayani permintaan aplikasi: login (nanti), daftar baju, edit tag. Permintaan seperti ini harus cepat dan ringan.
- **ml-engine** menjalankan model ML. Prosesnya berat (beberapa detik per foto) dan memakai Python, karena hampir semua library ML ada di Python.

## Komponen infrastruktur (docker-compose)

**Docker** menjalankan program di dalam "container", yaitu lingkungan terisolasi yang sudah berisi semua yang dibutuhkan program itu. Jadi kamu tidak perlu memasang PostgreSQL, Qdrant, dan Redis satu per satu di Windows. `docker-compose.yml` mendaftar container mana saja yang dijalankan dan setelannya.

| Container | Fungsi di Dressa |
|---|---|
| **PostgreSQL** | Database relasional (tabel). Menyimpan user, item pakaian, dan riwayat outfit. |
| **Qdrant** | *Vector database*. Menyimpan embedding tiap pakaian dan bisa mencari "pakaian yang paling mirip" dengan cepat. |
| **Redis** | Cache di memori. Nanti dipakai untuk menyimpan data cuaca sementara supaya tidak memanggil API cuaca terus-menerus. Belum dipakai. |

## api-gateway (Node.js + TypeScript)

- **Node.js** menjalankan JavaScript di server. **TypeScript** adalah JavaScript yang punya tipe, sehingga banyak kesalahan ketahuan sebelum program dijalankan (`npm run typecheck`).
- **Fastify** adalah framework web untuk membuat endpoint seperti `GET /items`.
- **Prisma** adalah ORM: kamu menulis struktur tabel di `prisma/schema.prisma`, lalu Prisma membuat tabelnya (`npm run db:migrate`) dan menyediakan fungsi seperti `prisma.wardrobeItem.findMany(...)`, jadi tidak perlu menulis SQL manual.
- **zod** memeriksa input dari user, misalnya warna harus berformat `#rrggbb`. Input yang salah ditolak dengan error 400 sebelum menyentuh database.

Susunan folder `services/api-gateway/src`:

| Folder | Isi |
|---|---|
| `routes/` | Daftar URL dan fungsi yang menanganinya |
| `controllers/` | Membaca dan memvalidasi request, lalu menyusun respons |
| `services/` | Logika utama: alur tambah item, panggilan ke ml-engine, penyimpanan file |
| `plugins/auth.ts` | Autentikasi sementara (lihat di bawah) |

**Autentikasi sementara.** Belum ada login. Untuk development, gateway percaya header `X-User-Id` yang dikirim client. Ini tidak aman, karena siapa pun bisa mengaku sebagai user lain. Karena itu fitur ini otomatis mati kalau `NODE_ENV=production`. Sebelum aplikasi dipakai orang lain, ini harus diganti login sungguhan (misalnya JWT atau Supabase Auth). Untuk membuat user uji, pakai `POST /dev/users`.

### Endpoint item

Semua butuh header `X-User-Id`.

| Method | Path | Fungsi |
|---|---|---|
| POST | `/items` | Upload foto (multipart, field `file`). Menjalankan seluruh alur di diagram atas. |
| GET | `/items?category=TOP&limit=50&offset=0` | Daftar item milik user |
| GET | `/items/:id` | Detail satu item |
| PATCH | `/items/:id` | Koreksi tag oleh user, misalnya `{"itemType": "polo shirt"}` |
| DELETE | `/items/:id` | Hapus item beserta vektor di Qdrant dan file gambarnya |

Kalau salah satu langkah di `POST /items` gagal, langkah sebelumnya dibatalkan: file gambar dihapus dan vektor di Qdrant dihapus. Tujuannya supaya tidak ada data setengah jadi.

## ml-engine (Python + FastAPI)

- **FastAPI** adalah framework web Python. Kelebihannya, halaman dokumentasi interaktif dibuat otomatis di `http://localhost:8000/docs`.
- **ONNX Runtime** menjalankan model yang sudah di-export ke format ONNX. Format ini tidak bergantung pada library training-nya, dan cukup cepat di CPU. Model buatanmu nanti juga akan di-export ke ONNX.

### Langkah di `/process-image`

1. **Hapus background** dengan model **IS-Net** lewat library **rembg**. Ini model *segmentasi*: untuk setiap piksel, model menebak apakah piksel itu bagian objek utama atau latar. Hasilnya *mask* (peta 0–1), lalu latar dibuat transparan.
2. **Warna dominan** dengan **k-means clustering**. Semua piksel pakaian dikelompokkan menjadi beberapa kelompok warna. Pusat tiap kelompok menjadi warna dominan, dan ukuran kelompok menjadi porsinya. Perhitungannya dilakukan di ruang warna **CIELAB** karena jarak di ruang ini lebih mirip cara mata manusia membedakan warna dibanding RGB.
3. **Embedding** dengan **CLIP** (lihat di bawah).
4. **Tag** (tipe, pola, style) dengan **CLIP zero-shot**. Bagian inilah yang akan diganti dengan model buatanmu sendiri.

### Embedding dan CLIP

**Embedding** adalah daftar angka (di sini 512 angka) yang mewakili "isi" sebuah gambar. Gambar yang mirip menghasilkan embedding yang berdekatan. Kedekatannya diukur dengan **cosine similarity**: 1 berarti arahnya sama persis, 0 berarti tidak berhubungan.

**CLIP** (dari OpenAI) dilatih dengan 400 juta pasangan gambar dan teks, sehingga gambar dan kalimat dipetakan ke ruang embedding yang sama. Dari situ muncul trik **zero-shot**: untuk menebak tipe baju, embedding gambar dibandingkan dengan embedding kalimat "a photo of a t-shirt", "a photo of a jacket", dan seterusnya. Kalimat yang paling dekat menjadi tebakannya. Cara ini tidak butuh training sama sekali, tapi akurasinya terbatas, seperti yang terlihat di hasil uji sebelumnya.

**Qdrant** menyimpan embedding semua pakaian. ID tiap titik di Qdrant sama dengan `WardrobeItem.id` di PostgreSQL, sehingga keduanya mudah dihubungkan. Setiap titik juga menyimpan `user_id`, dan setiap pencarian selalu difilter per user.

## Bagian mana yang pretrained dan mana yang buatan sendiri

| Bagian | Sekarang | Rencana |
|---|---|---|
| Hapus background | IS-Net pretrained | Tetap pretrained |
| Embedding (untuk kecocokan antar pakaian) | CLIP pretrained | Tetap untuk sementara |
| Tipe pakaian, style, musim | CLIP zero-shot | **Diganti CNN yang kamu latih dari nol** |
| Pola (polos, garis, dst.) | CLIP zero-shot | Tetap, karena datasetnya tidak punya label pola |
| Warna dominan | k-means, kode sendiri | Tetap |
