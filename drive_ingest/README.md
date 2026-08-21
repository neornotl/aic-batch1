# Official BTC video ingestion

This isolated job reads only the 14 official Videos and 14 official Keyframes ZIP URLs listed in the AIC2026 submission sheet. It uses HTTP Range requests to inspect ZIP central directories and streams each original member through `rclone rcat`; it never downloads a complete ZIP or stores a local video. Existing Drive files are skipped by path. No semantic labels or frame extraction are performed.

The logical hierarchy under the AIC root is `01_Videos_Original/L21/L21_V001/L21_V001.mp4`, `01_Videos_Original/L21/L21_V001/timestamps.jsonl`, and `01_Videos_Original/L21/L21_V001/keyframes/<official member name>`. L26 archive parts are merged logically under `L26`; their source archive/member names remain in the catalog. Keyframe JPEGs are uploaded as supplied by BTC, never regenerated.

## One-time setup

1. Create a Google Cloud service account with Drive API access and download its JSON key.
2. Share Drive folder `01_Videos_Original` with the service-account email as Content manager.
3. Add the complete JSON as GitHub Actions secret `GDRIVE_SERVICE_ACCOUNT_JSON`.
4. Run `Upload official videos to Drive` manually for each logical batch: `L21`–`L25`, `L26_a`–`L26_e`, and `L27`–`L30`. Each batch includes its official video and keyframe archives; exact archive filenames remain supported for targeted retries. The workflow uses the AIC root ID `1E8u-YURTRexdR1Ax4Hu2aq7Vt646VQuN` and fetches the exact timestamp map to runner temporary storage before ingestion.

The workflow refuses to run without the secret. Share the parent `AIC2026_VideoRAG` folder (ID `1E8u-YURTRexdR1Ax4Hu2aq7Vt646VQuN`) with the service account as Content manager. The job reads `02_Timelines/btc_keyframe_timestamps.jsonl` temporarily through rclone and writes exact per-video sidecars; it does not invent timestamps or YouTube chapters.
