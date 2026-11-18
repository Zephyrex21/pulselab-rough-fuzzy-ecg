# Additional ECG CSV test data

PulseLab's saved MIT-BIH model expects **two ECG leads** and accepts CSV/TXT waveform samples at 100–2000 Hz. For the most representative upload demo, use real MIT-BIH records from the project's patient-disjoint held-out test split.

## Export multiple held-out MIT-BIH records already on your computer

The project includes a helper that reads your local `data/mitdb` WFDB files and creates two-lead CSV files without retraining or changing any model:

```powershell
python tools/export_mitbih_demo_csv.py --max-records 5
```

Files are written to `data/demo_ecg_csv/`. The default selection comes from the configured patient-disjoint test split, and the exporter refuses training-split records. It preserves the native 360 Hz sampling rate and first two recorded leads. In the upload form, set **Sampling frequency = 360 Hz**.

You can choose specific held-out records too:

```powershell
python tools/export_mitbih_demo_csv.py --records 100 103 105 111 113
```

Only IDs in the configured patient-disjoint test split are accepted. These CSVs are useful for demonstrating repeatable end-to-end inference on multiple real records; upload predictions are not a new classification evaluation unless you compare them against the reference annotations.

## Additional open CSV source: Chapman–Shaoxing ECG

- [Official PhysioNet database page](https://www.physionet.org/content/ecg-arrhythmia/1.0.0/)
- [Original Figshare CSV collection](https://figshare.com/collections/ChapmanECG/4560497/2)
- [Dataset publication (Scientific Data)](https://doi.org/10.1038/s41597-020-0386-x)

The original Figshare release contains 10,646 de-identified, 12-lead ECG recordings as individual CSV files. Each recording is 10 seconds at 500 Hz, with 5,000 rows and 12 lead columns; sample amplitudes are in microvolts. The current complete PhysioNet release has 45,152 ECGs, but its files are WFDB/MAT rather than ready-to-upload CSVs. The original 10,646-record Figshare subset is therefore the most convenient CSV source.

For PulseLab's two-channel input, export exactly two ECG lead columns from a downloaded CSV (for example, `II` and `V1`) and retain the header. Set the upload sampling frequency to **500 Hz**. This is a different dataset and lead setup from MIT-BIH: use it as an out-of-domain robustness/demo example, not as a fair MIT-BIH benchmark. The app's feature-drift warnings should be retained and reported.

## Another public option: PTB-XL

[PTB-XL on PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/) provides 21,837 ten-second 12-lead ECGs. The native waveforms are WFDB at 500 Hz, with a 100 Hz downsampled version also available; `ptbxl_database.csv` is metadata, not the ECG waveform itself. To use a PTB-XL waveform, read one record with WFDB, export two selected leads to a CSV, and set the corresponding sampling frequency (500 or 100 Hz).

## Interpretation

Mean heart rate in PulseLab upload results is calculated from the automatically detected consecutive R-peak intervals. It is a signal-derived estimate and may be affected by missed or extra detections. It is not an annotated reference value or a clinically validated measurement.