# Official BTC video ingestion

This isolated job reads only the 14 official Videos and 14 official Keyframes ZIP URLs listed in the AIC2026 submission sheet. It uses HTTP Range requests to inspect ZIP central directories and streams each original member through `rclone rcat`; it never downloads a complete ZIP or stores a local video. Existing Drive files are skipped by path. No semantic labels or frame extraction are performed.

The logical hierarchy under the AIC root is `01_Videos_Original/L21/L21_V001/L21_V001.mp4`, `01_Videos_Original/L21/L21_V001/timestamps.jsonl`, and `01_Videos_Original/L21/L21_V001/keyframes/<official member name>`. L26 archive parts are merged logically under `L26`; their source archive/member names remain in the catalog. Keyframe JPEGs are uploaded as supplied by BTC, never regenerated.

## One-time setup

1. Preferred for a personal My Drive: on a trusted machine, run `rclone config` and create a Google Drive remote named `drive` using your user OAuth login. Set its root folder to `1E8u-YURTRexdR1Ax4Hu2aq7Vt646VQuN` (or ensure the generated config contains `root_folder_id = 1E8u-YURTRexdR1Ax4Hu2aq7Vt646VQuN`). Copy the complete config file contents, then add them as the GitHub Actions secret `RCLONE_CONFIG_DRIVE` (Settings → Secrets and variables → Actions). Do not paste or commit the config/token anywhere else.
2. Fallback for a Shared Drive only: create a Google Cloud service account with Drive API access and download its JSON key, share the target Shared Drive/folder with its email as Content manager, and add the complete JSON as `GDRIVE_SERVICE_ACCOUNT_JSON`. Service accounts cannot upload to a personal My Drive after their storage quota is exhausted.
3. Run `Upload official videos to Drive` manually for each logical batch: `L21`–`L25`, `L26_a`–`L26_e`, and `L27`–`L30`. Each batch includes its official video and keyframe archives; exact archive filenames remain supported for targeted retries. The workflow uses the AIC root ID `1E8u-YURTRexdR1Ax4Hu2aq7Vt646VQuN` and fetches the exact timestamp map to runner temporary storage before ingestion.

Run only 1–2 logical batches concurrently. The uploader applies conservative Google Drive pacing (`2` requests per second per rclone process, burst `1`, with a 5-second minimum Drive pacer sleep); higher parallelism can still trigger Drive project rate limits. Canceled or interrupted batches can be safely resumed because existing files are skipped by path.

The workflow refuses to run without one of the two secrets. With OAuth, share the parent `AIC2026_VideoRAG` folder (ID `1E8u-YURTRexdR1Ax4Hu2aq7Vt646VQuN`) with the user who authorized rclone. With the service-account fallback, share it with the service-account email as Content manager. The job reads `02_Timelines/btc_keyframe_timestamps.jsonl` temporarily through rclone and writes exact per-video sidecars; it does not invent timestamps or YouTube chapters.
