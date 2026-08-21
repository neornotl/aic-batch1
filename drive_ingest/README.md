# Official BTC video ingestion

This isolated job reads only the 14 official `aic-data.ledo.io.vn` ZIP URLs listed in the AIC2026 submission sheet. It uses HTTP Range requests to inspect ZIP central directories and streams each MP4 through `rclone rcat`; it never downloads a complete ZIP or stores a local video. Existing Drive files are skipped by path. No semantic labels or frame extraction are performed.

## One-time setup

1. Create a Google Cloud service account with Drive API access and download its JSON key.
2. Share Drive folder `01_Videos_Original` with the service-account email as Content manager.
3. Add the complete JSON as GitHub Actions secret `GDRIVE_SERVICE_ACCOUNT_JSON`.
4. Run `Upload official videos to Drive` manually with `GDRIVE_FOLDER_ID` set to the folder ID. Use one archive per dispatch (`archive`) so runs can resume.

The workflow refuses to run without the secret/folder ID. Do not put credentials in repository files or logs. The generated JSONL catalog records archive/member, official video ID, Drive path/link, duration when `ffprobe` is available, and the BTC timestamp-index reference; it contains no semantic labels.
