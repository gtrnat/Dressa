# Mesin Rekomendasi Outfit (Phase 2)

Kode utamanya ada di `services/api-gateway/src/services/`:

| File | Isi |
|---|---|
| `scoring.ts` | Semua rumus skor. Berupa fungsi murni, jadi mudah diuji dan dijelaskan. |
| `recommender.ts` | Mengambil konteks, menyusun kombinasi outfit, memberi skor, memilih alternatif yang beragam |
| `weather.ts` | OpenWeatherMap forecast 5 hari, di-cache di Redis 30 menit |
| `google-calendar.ts` | OAuth Google dan pengambilan agenda (opsional) |
| `event-classifier.ts` | Menebak tipe agenda (formal / casual / sport) dari judul event |

## Alur `GET /recommendations`

1. **Konteks hari itu:**
   - **Cuaca**: diambil dari lokasi user (`PATCH /me`) atau dari query `lat` dan `lon`. Yang dirangkum adalah jam 06:00–22:00 waktu lokal: rata-rata suhu "feels like", peluang hujan tertinggi, dan total curah hujan.
   - **Agenda**: diambil dari Google Calendar kalau sudah terhubung, atau dari query `eventType=formal|casual|sport` kalau ingin diisi manual.
2. **Kandidat outfit**: (atasan + bawahan) atau dress, ditambah sepatu, ditambah outer (opsional). Outer hanya dipertimbangkan kalau terasa ≤ 24°C, kemungkinan hujan, atau cuaca tidak diketahui. Supaya jumlah kombinasi tetap kecil, setiap kategori dibatasi 12 item paling menjanjikan berdasarkan skor rotasi dan kecocokan agenda.
3. **Skor setiap kandidat**, lalu kandidat diurutkan.
4. **Pemilihan alternatif yang beragam**: dua outfit yang ditampilkan boleh berbagi paling banyak setengah item-nya. Parameter `reroll=1, 2, ...` mengambil halaman alternatif berikutnya.

## Rumus

$$\text{Score} = w_1 \cdot \text{StyleMatch} + w_2 \cdot \text{WeatherFit} + w_3 \cdot \text{CalendarFit} + w_4 \cdot \text{RotationWeight}$$

Setiap komponen bernilai 0–1. Bobot default sama rata, $w_i = 0.25$. **Komponen yang tidak tersedia dibuang, lalu bobot sisanya dinormalisasi ulang** supaya jumlahnya tetap 1:
- Tanpa kalender dan tanpa `eventType`: CalendarFit dibuang, dan ketiga komponen lain masing-masing berbobot 1/3.
- Tanpa lokasi atau API key cuaca: WeatherFit dibuang.

Alasan komponen dibuang dilaporkan di `context.unavailable`.

### StyleMatch, yaitu keserasian antar item

$$\text{StyleMatch} = 0.4 \cdot \text{Warna} + 0.3 \cdot \text{Visual} + 0.3 \cdot \text{Gaya}$$

- **Warna**: dihitung dari warna dominan tiap item (hasil k-means), lalu diubah ke HSL.
  - Warna netral (hitam, putih, abu-abu, navy, denim, beige/khaki/coklat) cocok dengan warna apa pun dan mendapat skor 1.0.
  - Untuk dua warna yang sama-sama tidak netral, skornya ditentukan jarak hue: analog (≤30°) 0.9, komplementer (>140°) 0.8, triadik (~120°) 0.6, sedangkan 60–100° cenderung bertabrakan, 0.4.
  - Kalau ada lebih dari 2 item berwarna terang, skor dikurangi 0.15 untuk setiap item tambahan.
- **Visual**: rata-rata cosine similarity embedding CLIP antar item, dipetakan dari rentang [0.6, 0.9] ke [0, 1]. Ini heuristik sederhana. Model kecocokan buatan sendiri (opsi 2) bisa menggantikan bagian ini nanti.
- **Gaya**: kecocokan tag style antar item. Contohnya, formal dengan formal bernilai 1, formal dengan casual 0.45, dan formal dengan sporty 0.1.

### WeatherFit, yaitu kecocokan dengan cuaca

Kehangatan yang diharapkan dihitung dari suhu "feels like" $T$ (°C):

$$\text{target}_{atas} = \text{clamp}\left(1 + \frac{30 - T}{3},\ 1,\ 9\right) \qquad \text{target}_{bawah} = \text{clamp}\left(1 + \frac{30 - T}{4},\ 1,\ 5\right)$$

Contohnya, 30°C menghasilkan target atas 1 (kaos), dan 18°C menghasilkan 5 (butuh lapisan). Kehangatan outfit dihitung dari `warmthLevel` tiap item, yang ditebak model ML: atas = atasan + 0.8 × outer, bawah = bawahan. Kecocokannya diukur dengan fungsi Gaussian:

$$\text{fit} = 0.6 \cdot e^{-\frac{(atas - target_{atas})^2}{2 \cdot 1.5^2}} + 0.4 \cdot e^{-\frac{(bawah - target_{bawah})^2}{2 \cdot 1.2^2}}$$

Penyesuaian tambahan:
- Kalau hujan (peluang ≥ 50% atau ≥ 1 mm) dan ada outer atau sepatu yang tahan air: skor +0.1. Kalau tidak ada: skor ×0.9.
- Sandal, flats, atau heels saat hujan atau saat terasa < 18°C: skor ×0.85.

### CalendarFit, yaitu kecocokan dengan agenda

Untuk setiap item: $0.5 \cdot \text{skor style} + 0.5 \cdot \text{skor tipe}$, dari dua tabel di `scoring.ts`. Contohnya, untuk agenda formal, kemeja = 1, kaos = 0.2, dan celana training = 0. CalendarFit adalah rata-rata skor semua item.

Kalau dalam sehari ada beberapa agenda: kalau ada satu saja yang **formal**, hari itu dianggap formal. Hari dianggap **sport** hanya kalau agenda olahraga mengambil sebagian besar jam di hari itu. Selain itu, hari dianggap **casual**. Agenda seharian penuh (all-day) dan agenda berstatus "free", misalnya ulang tahun, diabaikan.

### RotationWeight, yaitu prioritas baju yang jarang dipakai

Untuk setiap item, dengan $d$ = jumlah hari sejak terakhir dipakai (dianggap 30 kalau belum pernah dipakai):

$$r = 0.7 \cdot \left(1 - e^{-d/7}\right) + 0.3 \cdot \left(1 - \frac{\text{wearCount}}{\max \text{wearCount}}\right)$$

Item yang dipakai kemarin mendapat ≈0.1, seminggu lalu ≈0.6, dan sebulan lalu ≈1.0. Nilai ini diperbarui setiap kali user mencatat pemakaian lewat `POST /outfits`.

## Contoh pemakaian

Semua permintaan memakai header `X-User-Id: <id>`.

```http
PATCH /me                      {"latitude": -6.2, "longitude": 106.8}
GET   /recommendations         → 3 outfit terbaik hari ini
GET   /recommendations?reroll=1
GET   /recommendations?eventType=formal&date=2026-10-31
GET   /recommendations?wWeather=0.5&wRotation=0.1   → ubah bobot untuk eksperimen
POST  /outfits                 {"itemIds": ["...", "..."], "recommended": true, "score": 0.91}
DELETE /outfits/<id>           → batalkan catatan (wearCount dan lastWornAt dikembalikan)
```

Setiap outfit di respons berisi `score` (total, keempat komponen, dan rincian StyleMatch) serta `reasons`, yaitu alasan dalam bahasa Indonesia yang bisa langsung ditampilkan di aplikasi.

## Menghubungkan Google Calendar (opsional)

1. Buka [Google Cloud Console](https://console.cloud.google.com/), lalu buat project.
2. **APIs & Services → Library**: aktifkan **Google Calendar API**.
3. **OAuth consent screen**: pilih tipe External. Tambahkan scope `.../auth/calendar.events.readonly`, lalu tambahkan email kamu sebagai **Test user**.
4. **Credentials → Create credentials → OAuth client ID**: pilih tipe *Web application*. Isi Authorized redirect URI dengan `http://localhost:3000/integrations/google/callback`.
5. Isi `GOOGLE_CLIENT_ID` dan `GOOGLE_CLIENT_SECRET` di `.env`, lalu restart api-gateway.
6. Panggil `GET /integrations/google/auth-url`, buka URL yang dikembalikan di browser, lalu izinkan akses.
7. Cek `GET /integrations/google`. Hasilnya harus `"connected": true`.

Dressa hanya meminta izin **membaca** event. Refresh token disimpan terenkripsi (AES-256-GCM). Parameter `state` OAuth ditandatangani dengan HMAC dan kedaluwarsa dalam 10 menit, untuk mencegah CSRF. `DELETE /integrations/google` mencabut akses di Google dan menghapus token.

## Ide eksperimen untuk laporan

- Bandingkan bobot berbeda dengan parameter `w*`. Misalnya, seberapa sering rekomendasi berubah kalau bobot rotasi dinaikkan?
- Ablation: matikan satu komponen (bobot 0), lalu minta beberapa teman menilai outfit yang dihasilkan (skor 1–5).
- Ukur rotasi lemari setelah N hari: berapa persen item yang pernah dipakai, dengan dan tanpa RotationWeight.
