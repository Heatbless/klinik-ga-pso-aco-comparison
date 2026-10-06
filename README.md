# Klinik Optima — GA, PSO, dan ACO

Aplikasi desktop Tkinter berbahasa Indonesia untuk membandingkan optimasi antrean pasien menggunakan **Genetic Algorithm**, **Particle Swarm Optimization**, dan **Ant Colony Optimization**. Tersedia replay proses algoritma, animasi pelayanan klinik, eksperimen berulang, serta ekspor CSV dan JSON.

## Menjalankan

Memerlukan Python 3.10+ dengan Tkinter. Tidak ada dependensi pip. Pada lingkungan pengembangan, aplikasi diuji dengan Python 3.14 dan Tk 9 di Windows.

```powershell
python main.py
```

1. Di **Skenario**, isi jumlah pasien/dokter, rentang kedatangan, dan seed, lalu klik **Bangkitkan data**. Angka yang diubah baru diterapkan ke dataset setelah tombol tersebut diklik. Durasi pemeriksaan 5–20 menit; prioritas 1/2/4 dibangkitkan dengan probabilitas 70%/20%/10%.
2. Pilih fungsi fitness dan anggaran evaluasi. Bobot Gabungan dinormalisasi otomatis; seluruh bobot nol tidak diperbolehkan.
3. Di **Demo**, klik **Jalankan demo**. Setelah perhitungan selesai, pilih **Proses optimasi** atau **Pelayanan klinik**, lalu **Putar**. Kontrol Jeda, Langkah berikutnya, Ulangi, dan kecepatan berlaku bersama pada ketiga panel.
4. Di **Eksperimen**, pilih skenario aktif atau ketiga preset, fitness terpilih atau keempat fitness, dan jumlah pengulangan. Klik **Jalankan eksperimen**. Pilih kelompok grafik untuk melihat konvergensi dan baseline.
5. Tombol ekspor membuat subfolder baru berisi `hasil.csv`, `konvergensi.csv`, `ringkasan.csv`, dan `manifest.json`. Ekspor tidak menimpa folder hasil sebelumnya.

Mengubah skenario atau fitness tidak mengubah hasil yang telah dihitung. Jalankan ulang untuk menghasilkan solusi dengan konfigurasi baru. Header replay menyebutkan konfigurasi asal hasil.

## Model dan fitness

Satu layanan dengan beberapa dokter setara, tanpa interupsi pemeriksaan. Seluruh kedatangan diketahui sebelum optimasi. Permutasi solusi menentukan peringkat pemilihan **di antara pasien yang sudah datang**. Dokter tidak menunggu pasien masa depan ketika ada pasien yang bisa dilayani. Kedatangan dan selesainya pemeriksaan pada waktu yang sama diproses sebelum penugasan dokter; dokter dengan ID terkecil ditugaskan lebih dahulu.

Untuk pasien i: `A_i` = kedatangan, `S_i` = mulai pemeriksaan, `C_i` = selesai pemeriksaan, `w_i` = bobot prioritas. Semua waktu dalam menit sejak klinik dibuka pukul 08:00.

- **Waktu tunggu berbobot:** `F1 = sum(w_i * (S_i - A_i)) / sum(w_i)`.
- **Waktu tunggu maksimum:** `F2 = max(S_i - A_i)`.
- **Waktu seluruh pelayanan selesai:** `F3 = max(C_i)` sejak pembukaan.
- **Gabungan:** `F4 = alpha * F1_hat + beta * F2_hat + gamma * F3_hat`, dengan `Fk_hat = Fk / max(Fk_FCFS, 1 menit)`. Bobot awal 0,50 / 0,30 / 0,20. Penyebut dihitung sekali dari FCFS dataset yang sama dan tetap selama pencarian.

**Seluruh fungsi diminimalkan.** F1–F3 bersatuan menit; F4 tanpa satuan. Aplikasi selalu melaporkan metrik asli. Skor dan peringkat dari mode fitness berbeda tidak dibandingkan langsung. Bobot prioritas adalah atribut simulasi, bukan aturan triase atau jaminan batas waktu tunggu.

## Algoritma dan perbandingan

| Algoritma | Implementasi | Parameter tetap |
|---|---|---|
| GA | Permutasi, seleksi turnamen, order crossover, mutasi swap, elitisme | Populasi 20; turnamen 3; crossover 0,9; mutasi per anak 0,2; elite 2 |
| PSO | Random keys; pengurutan key menghasilkan permutasi; personal/global best | 20 partikel; inertia 0,7; c1=c2=1,5; kecepatan ±0,2; posisi 0–1 |
| ACO | Ant System; transisi feromon, heuristik bobot/durasi, simpul awal buatan | 20 semut; alpha=1; beta=2; evaporasi 0,2; feromon awal 1; deposit per semut `1/(1+skor)` |

PSO memecah key seri berdasarkan ID. Pada batas posisi PSO, komponen kecepatan yang keluar batas menjadi nol. Feromon ACO memiliki batas bawah numerik 1e-12. ACO menguapkan dan mendepositkan feromon setelah setiap kelompok, termasuk kelompok terakhir yang belum penuh. GA memakai skor elite kembali tanpa evaluasi ulang.

Setiap panggilan evaluator selama optimasi dihitung, termasuk inisialisasi dan solusi duplikat. Ketiga algoritma berhenti tepat pada anggaran (default 2.000). Baseline FCFS dan Prioritas/Durasi dihitung terpisah, tidak memakai anggaran algoritma. Tidak ada early stopping.

Eksperimen memakai dataset tetap per skenario dan seed optimizer 0…n−1 (default 10 pengulangan). Urutan algoritma dirotasi setiap pengulangan. Komputasi berlangsung berurutan, bukan bersamaan. Waktu mencakup pencarian dan evaluasi, tanpa rendering, snapshot visual, atau ekspor. Rata-rata, simpangan baku sampel, dan nilai terbaik dilaporkan per skenario/fitness/algoritma. Pengukuran waktu bergantung pada kondisi komputer; simpangan baku skor menunjukkan variasi optimizer pada dataset tetap.

Peringkat memakai rata-rata skor pada kelompok yang sama. Skor seri mendapat peringkat yang sama (toleransi numerik 1e-9), dengan peringkat berikutnya mengikuti posisi, misalnya 1, 1, 3. Pilih baris pengulangan untuk melihat rata-rata waktu tunggu tanpa bobot dan waktu tunggu setiap kelas prioritas. Panel animasi dapat digulir vertikal jika semua keterangan tidak muat.

Preset: Kecil (20 pasien, 2 dokter, kedatangan 0–60, seed 42), Sedang (50, 3, 0–120, seed 43), Besar (100, 5, 0–180, seed 44).

Replay optimasi disinkronkan berdasarkan jumlah evaluasi. Snapshot ditampilkan pada akhir generasi/kelompok, sehingga nomor snapshot GA bisa berbeda dari PSO/ACO pada titik replay yang sama. Grafik menyimpan skor terbaik pada setiap evaluasi. Plot PSO adalah proyeksi dua key pasien, bukan posisi fisik. Heatmap ACO menampilkan hingga 20 pasien pertama; kromosom GA hingga 10. Antrean/selesai menampilkan hingga 24 ikon per kategori dengan jumlah sisanya tertulis. Seluruh pasien tetap dihitung dan jadwal lengkap diekspor.

## Eksperimen tanpa antarmuka

```powershell
python benchmark.py --patients 20 --doctors 2 --budget 2000 --repeats 10
python benchmark.py --presets --all-fitness --repeats 10 --budget 2000
python benchmark.py --fitness combined --weights 5 3 2 --output exports\percobaan-baru
```

Folder `--output` harus belum ada. Tanpa opsi tersebut, folder berstempel waktu dibuat dalam `exports/`. JSON menyimpan dataset, konfigurasi algoritma, bobot asli dan ternormalisasi, penyebut normalisasi, seed, jadwal terbaik, metrik, dan riwayat konvergensi.

## Pengujian

```powershell
python -m unittest discover -v
```

Pengujian mencakup skor manual, enumerasi kasus kecil, validitas jadwal dan permutasi, tidak menganggurkan dokter ketika ada antrean, kedatangan bersamaan, satu pasien, normalisasi baseline nol, anggaran yang tidak habis dalam satu kelompok, reproduksi seed, pembatalan, ketidakbergantungan pencarian pada observer animasi, pemisahan kelompok eksperimen, serta konsistensi ekspor.

## Struktur

- `klinik/model.py`: data pasien, generator, simulator, baseline, dan evaluator fitness.
- `klinik/algorithms.py`: GA, PSO, ACO, hasil, snapshot, dan pembatalan.
- `klinik/experiments.py`: pengulangan, agregasi, dan ekspor.
- `klinik/ui.py`: aplikasi Tkinter, chart Canvas, worker queue, dan replay.
- `main.py`: aplikasi desktop; `benchmark.py`: CLI eksperimen.

Referensi representasi: [PSO random keys untuk penjadwalan — Sankaran](https://open.clemson.edu/all_theses/690/) dan [bibliografi Ant System — Dorigo](https://iridia.ulb.ac.be/~mdorigo/ACO/publications.html). Implementasi di sini mengadaptasi pendekatan tersebut ke antrean klinik; parameter awal bukan hasil tuning untuk menjamin satu algoritma menang.
