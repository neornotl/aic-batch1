# AIC2026 / Video-RAG — tổng hợp cuộc thi, dự án và GitHub Actions

> Tài liệu này mô tả những gì repository đang làm, dữ liệu đi qua hệ thống,
> các lỗi đã quan sát và hai luồng chạy mới (repair/resume). Đây là tài liệu
> kỹ thuật vận hành, không phải file nộp đáp án. Trạng thái run là snapshot tại
> lúc tạo và cần đối chiếu lại GitHub trước khi dùng làm báo cáo cuối.

## 1. Bối cảnh cuộc thi

Cuộc thi là Hội thi Thử thách Trí tuệ Nhân tạo Thành phố Hồ Chí Minh năm 2026
(AIC2026). Phần đang xử lý là vòng sơ tuyển, dữ liệu đợt 1 (Batch 1). Website
vòng sơ tuyển được đội sử dụng là `https://sotuyenaic.oj.io.vn/`.

### 1.1. Ba loại truy vấn

1. **Textual Known Item Search (Textual KIS)**: từ mô tả tự nhiên, tìm đúng một
   video và nộp một frame bất kỳ thuộc đoạn đáp án đúng. Định dạng:
   `<video_id>, <frame_id>`.
2. **Q&A / Visual Question Answering**: tìm đúng video, frame liên quan và trả
   lời câu hỏi bằng tiếng Việt hoặc tiếng Anh. Định dạng:
   `<video_id>, <frame_id>, <answer>`.
3. **TRAKE (Temporal Retrieval and Alignment of Key Events)**: tìm đúng một
   video và một frame cho từng mốc ngữ nghĩa của chuỗi sự kiện. Nếu có `N` mốc,
   định dạng là `<video_id>, <frame_id_1>, ..., <frame_id_N>`.

“Semantic keyframe” trong TRAKE là khoảnh khắc có ý nghĩa nội dung, không phải
I-frame kỹ thuật của codec.

### 1.2. Cách chấm

- Mỗi câu có thể nhận tối đa 100 câu trả lời.
- Với KIS/Q&A, video phải đúng và frame phải nằm trong khoảng frame đáp án
  `[s, e]`; Q&A còn phải đúng ngữ nghĩa câu trả lời.
- Với TRAKE, sai `video_id` thì câu nhận 0 ngay; nếu video đúng, điểm là tỷ lệ
  số mốc có frame nằm trong đúng khoảng `[s_j, e_j]`.
- `R@k` là điểm R-Score tốt nhất trong k câu trả lời đầu tiên, với
  `k ∈ {1, 5, 20, 50, 100}`.
- Final Score là trung bình của năm giá trị `R@1, R@5, R@20, R@50, R@100`.
  Vì vậy thứ tự xếp hạng rất quan trọng: đáp án đúng ở vị trí đầu giúp `R@1`,
  còn đáp án đúng ở vị trí sâu hơn chỉ giúp các mốc top lớn hơn.

> Ghi chú vận hành từ đội: giai đoạn thử nghiệm cho phép gửi nhiều lần hơn;
> khi thi thật số lượt ít hơn và chỉ nên nộp sau khi rà soát. Các workflow trong
> tài liệu này chỉ xử lý dữ liệu/RAG, không tự động nộp lên hệ thống thi.

### 1.3. Dữ liệu BTC

BTC cung cấp:

- **Videos**: dữ liệu video chính thức, là thành phần chấm trực tiếp.
- **Keyframes**: frame BTC đã trích xuất; tên frame theo video và metadata chứa
  frame index chính thức.
- **Objects**: JSON object detection từ Faster R-CNN trên OpenImages V4, một
  JSON tương ứng với mỗi keyframe.
- **CLIP features**: vector CLIP ViT-B/32 theo đúng thứ tự keyframe.
- **Metadata**: metadata nguồn video (thường từ YouTube); một số video không có.

Keyframes, objects, CLIP và metadata là dữ liệu hỗ trợ xây dựng hệ thống mẫu;
video vẫn là nguồn chính thức để chấm. Frame trả lời phải tra theo
metadata/keyframe BTC, không tự cắt lại rồi coi đó là frame chính thức.

Các archive video logic đang dùng: L21, L22, L23, L24, L25, L27, L28, L29,
L30 có archive riêng; L26 được chia thành `Videos_L26_a.zip` tới
`Videos_L26_e.zip`. Script giữ nguyên tên member/archive nguồn.

## 2. Kiến trúc dự án hiện tại

### 2.1. Luồng dữ liệu chính

```text
BTC Video ZIP (HTTP Range)
        │
        ├─ RangeZip đọc central directory, không tải cả ZIP về máy
        ├─ tải MP4 hiện tại vào thư mục tạm của runner
        ├─ upload MP4 tạm lên GCS
        ├─ Vertex Gemini 2.5 Flash xem video qua gs:// URI
        ├─ xóa object GCS trong finally
        ├─ ghi một dòng JSONL theo video
        └─ upload artifact + publish JSONL lên Google Drive
                         │
                         └─ RAG đọc summary và truy hồi video/frame ứng viên
```

Thiết kế này giữ máy cá nhân nhẹ: GitHub-hosted runner xử lý, video ZIP được
đọc theo HTTP Range và chỉ MP4 hiện tại nằm trên đĩa tạm. GCS chỉ là vùng trung
gian cho Vertex, không phải kho lưu trữ vĩnh viễn. Keyframe ZIP chính thức được
giữ nguyên; không tạo frame BTC giả.

### 2.2. Vị trí Drive

Root folder đang cấu hình: `1E8u-YURTRexdR1Ax4Hu2aq7Vt646VQuN`.

- `03_Video_Summaries/video_summaries_<BATCH>.jsonl`: summary chính theo batch.
- `03_Video_Summaries/repairs/video_summaries_repair_<BATCH>.jsonl`: output sửa
  chọn lọc, không ghi đè summary chính.
- Uploader video/keyframe tổ chức video theo batch/video và giữ timestamp JSONL,
  keyframe ZIP theo logical batch.

## 3. Mô tả từng GitHub Action

| Workflow | Mục đích và đầu ra |
|---|---|
| **Upload official videos to Drive** | Smoke test/đẩy một archive batch lên Drive; kiểm tra quyền và cấu hình rclone. |
| **Upload full official database sequentially** | Upload toàn bộ 14 logical batch tuần tự để giảm Drive/API rate limit. Mỗi batch gồm video MP4 riêng, keyframe ZIP chính thức và timestamp JSONL. |
| **Summarize full official Batch 1 videos** | Đọc video ZIP theo RangeZip, đẩy MP4 tạm lên GCS, gọi Vertex Gemini 2.5 Flash, ghi JSONL, upload artifact và publish summary lên Drive. Matrix gồm 14 batch. |
| **Smoke test Vertex video summary via GCS** | Kiểm tra đường đi service account → GCS → Vertex → summary trước khi dùng quota lớn. |
| **AIC Batch 1 - Finish Remaining v4** | Luồng batch-1 cũ/tiếp tục các phần còn lại theo shard và artifact. Không phải parser summary mới. |
| **AIC Cascade Processing** | Chạy pipeline cascade/ranking nhiều tầng trên các shard overnight, dùng cache/artifact để không tính lại. |
| **Enrich Candidate Videos** | Tải nhóm candidate, chạy enrichment/evidence generation và đóng gói output candidate. |
| **Extract Frames from Candidate Videos** | Tải candidate và tách frame phục vụ review; không thay keyframe BTC khi nộp. |
| **Publish public source-video links** | Tạo/publish link nguồn video candidate cho review theo cấu hình action. |
| **AIC Evidence Video Bundles** | Gom frame evidence, README frame list và source video rank-1 thành artifact ZIP để nhóm kiểm tra tay. |
| **AIC Evidence Review** | Render/đóng gói review evidence theo kế hoạch để kiểm tra frame/video liên quan. |
| **AIC Video Window Review** | Tạo artifact các cửa sổ video quanh frame ứng viên, hỗ trợ xem trước/sau khoảnh khắc. |

Các action review/evidence không tự quyết đáp án. Chúng tạo gói kiểm chứng có
nguồn; quyết định frame cuối vẫn phải bám keyframe/metadata BTC.

## 4. Lỗi đã quan sát và nguyên nhân

### 4.1. JSON malformed bị ghi nhầm là thành công

Phiên bản cũ của `scripts/summarize_official_videos.py` chỉ gọi `json.loads`;
nếu parse thất bại, nó biến raw text thành `summary` và gán `confidence=low`,
sau đó vẫn trả `status=ok`. Vì vậy “low confidence” trước đây phần lớn có
nghĩa là **parser không đọc được JSON**, không phải Vertex tự đánh giá hình ảnh
thấp.

Các dạng lỗi thực tế: dấu ngoặc kép trong nội dung không escape; thiếu dấu phẩy
hoặc dấu ngoặc ở `timeline`/`visual_entities`; code fence/trailing text; phản
hồi thiếu trường schema.

### 4.2. Lỗi API/quota là nhóm khác

Artifact đã kiểm tra có lỗi `429` (quota/rate limit) và `403 Spend cap breached`.
Đây không phải lỗi parser. Budget mới giúp nhóm Spend cap nếu billing/quota đã
được cấp, nhưng vẫn cần giới hạn workers và pacing để tránh 429.

Snapshot artifact trước khi sửa:

| Batch | Số dòng | `status=ok` | `status=error` | Low-confidence |
|---|---:|---:|---:|---:|
| L23 | 25 | 25 | 0 | 3 |
| L24 | 43 | 40 | 3 | 2 |
| L27 | 16 | 15 | 1 | 0 |
| L28 | 24 | 10 | 14 | 1 |
| L26_d | 100 | 55 | 45 | 4 |
| **Tổng** | **208** | **145** | **63** | **10** |

10 ID cần repair: `L23_V001, L23_V005, L23_V023, L24_V019, L24_V028,
L28_V008, L26_V301, L26_V318, L26_V330, L26_V333`.

### 4.3. Timeout của run full

Run full cũ `33073572170` dùng `timeout-minutes: 350`. Các batch
`L21, L22, L25, L26_a, L26_b, L26_c` kết thúc `cancelled` sau thời gian xấp xỉ
timeout; đây là lý do cần luồng resume. L23, L24, L27, L28, L26_d đã thành
công; L26_e, L29, L30 từng còn chạy tại snapshot.

## 5. Thay đổi trên nhánh riêng

Nhánh: `codex/summary-parser-repair`  
Worktree: `C:\Users\laptopppp\aic-batch1-fix`

### 5.1. Parser trung thực và schema gate

`scripts/summarize_official_videos.py` hiện:

- chấp nhận object JSON hợp lệ có code fence hoặc trailing explanation;
- kiểm tra đủ `summary`, `opening_scene`, `timeline`, `visual_entities`,
  `actions`, `on_screen_text`, `locations`, `closing_scene`,
  `search_keywords`, `confidence`;
- kiểm tra kiểu dữ liệu và confidence `high|medium|low`;
- nếu parse/schema lỗi, ghi `status=error`, `parse_error`, raw response tối đa
  12.000 ký tự và cờ `raw_response_truncated`;
- không biến raw text lỗi thành summary thành công;
- thêm `--video-ids` để retry ID chọn lọc;
- yêu cầu Vertex `application/json`, `temperature=0`, và nhắc escape dấu ngoặc;
- vẫn xóa GCS object trong `finally`, kể cả khi generation/parse lỗi.

Nguyên tắc là **không tự sửa nội dung bằng cách đoán**. JSON hỏng được đưa vào
hàng retry/review để evidence không bị “làm đẹp” giả.

### 5.2. Timeout và tách luồng

Workflow `Summarize full official Batch 1 videos` trên nhánh này:

- tăng timeout mỗi job từ 350 lên 720 phút;
- giữ checkpoint theo JSONL và cleanup GCS;
- thêm input `repair`, `targets`, `resume`;
- dùng concurrency group riêng cho full, repair và resume.

### 5.3. Repair mode

- Matrix gồm `L22, L23, L24, L26_d, L28`; các batch không có target sẽ no-op.
- Tối đa 2 batch song song, mặc định `workers=1` để hạn chế 429.
- Chỉ xử lý 10 ID low-confidence đã xác định.
- Với response bị cắt, input `compact=true` dùng prompt ngắn hơn nhưng vẫn bắt
  buộc đủ schema; output repair không ghi đè summary chính.
- Artifact `video-summaries-repair-<BATCH>`.
- Drive `03_Video_Summaries/repairs/video_summaries_repair_<BATCH>.jsonl`.
- Không ghi đè summary chính.

### 5.4. Resume mode

- Matrix chỉ gồm sáu batch bị timeout: `L21, L22, L25, L26_a, L26_b, L26_c`.
- Tối đa 2 batch song song, mặc định `workers=1`.
- Publish lại đúng tên summary chính của batch còn thiếu.
- Dùng parser mới, structured JSON response và timeout 720 phút.
- L29 có job resume riêng (`resume_l29=true`) và concurrency group riêng để
  không phải chờ sáu batch cũ.

## 6. Các run đã mở

| Run | Vai trò | SHA/nhánh | Trạng thái snapshot |
|---|---|---|---|
| [33073572170](https://github.com/neornotl/aic-batch1/actions/runs/33073572170) | Full gốc | `main` / `13e34a7` | **Đã cancelled** lúc L29 chạm timeout cũ; L23, L24, L26_d, L26_e, L27, L28, L30 success; sáu batch cũ và L29 chưa có summary hoàn chỉnh. |
| [33141325971](https://github.com/neornotl/aic-batch1/actions/runs/33141325971) | Repair 10 ID | `codex/summary-parser-repair` / `16f9bb4` | **Hoàn tất success**; 10/10 rows `status=ok`, đủ schema, không còn error trong artifact repair. |
| [33141595264](https://github.com/neornotl/aic-batch1/actions/runs/33141595264) | Dispatch nhầm | `codex/summary-parser-repair` | Đã hủy ngay vì điều kiện ban đầu tạo thêm full matrix; không phải run full gốc. |
| [33141702955](https://github.com/neornotl/aic-batch1/actions/runs/33141702955) | Resume sáu batch | `codex/summary-parser-repair` / `7308203` | L21 và L22 success; L25/L26_a đang chạy; L26_b/L26_c queued. L22 cũ còn 6 lỗi do response bị cắt. |
| [33152076394](https://github.com/neornotl/aic-batch1/actions/runs/33152076394) | Compact repair L22 | `codex/summary-parser-repair` / `6c6ea54` | **Success**; artifact có đúng 6 target, 6/6 `status=ok`, đủ schema, không còn `raw_response_truncated`; Drive publish thành công. |
| [33152078590](https://github.com/neornotl/aic-batch1/actions/runs/33152078590) | Resume riêng L29 | `codex/summary-parser-repair` / `6c6ea54` | **Đang chạy**; 23 video L29, timeout 720 phút, concurrency độc lập. |

Hậu kiểm GCS lúc `2026-08-28T08:25Z`: prefix `aic-batch1-summary/` có 9 object,
tổng khoảng 1.17 GiB. Ba object mới tương ứng worker đang chạy (L25, L26_a,
L29); sáu object cũ (L23, L24, L25_V026, L26_b, L26_c, L29_V023) có dấu hiệu
rò và chỉ nên dọn sau khi mọi worker kết thúc. Không xóa trong lúc job còn chạy.

## 7. Tiêu chí “đủ đưa vào RAG”

1. Mỗi logical batch có JSONL, số video khớp manifest/archive.
2. Không còn `status=error` ngoài các dòng được đánh dấu retry/review.
3. Mọi `status=ok` có đủ schema và confidence hợp lệ.
4. Repair rows được hợp nhất/đọc ưu tiên mà không che raw evidence.
5. Artifact GitHub tải được và file Drive đúng thư mục.
6. Prefix tạm `aic-batch1-summary/` trong GCS không còn object rò sau jobs.
7. RAG index lưu `video_id`, batch/archive, summary, timeline, entities,
   actions, on-screen text, locations, keywords và confidence.
8. Frame cuối vẫn lấy từ keyframe/metadata BTC; summary chỉ giúp truy hồi và
   định vị thời gian, không thay ground truth.

## 8. Cách RAG tận dụng output

- **Lọc nhanh**: normalize tiếng Việt và BM25/FTS trên summary, keywords,
  entities, actions, locations, on-screen text.
- **Xếp hạng**: ưu tiên exact video ID/địa danh/hành động/chữ, sau đó rerank
  semantic; giữ nhiều candidate thay vì chốt sớm.
- **Thời gian**: dùng `timeline.approx_time` để mở cửa sổ video rồi đối chiếu
  keyframe BTC và frame range đáp án.
- **Q&A/TRAKE**: kiểm tra lại answer/mốc semantic bằng frame/video; không lấy
  confidence của summary làm điểm thi.
- **Audit**: lưu source member, archive batch, model, parse error và artifact.

## 9. Việc còn lại sau các run

1. Đọc artifact repair, đếm `ok/error`, kiểm tra 10 ID còn parse error, 429 hay
   403 không.
2. Đọc artifact sáu batch resume; xác nhận không timeout và Drive đã nhận file.
3. Kiểm tra GCS prefix tạm, đặc biệt sau job bị hủy.
4. Hợp nhất repair rows theo `video_id`; row lỗi giữ audit, không đưa vào positive
   index.
5. Smoke retrieval trên Q3, Q9, Q11, Q12, Q13, Q14, Q15 và review tay frame.
6. Chỉ sau checklist mới tạo submission variant; pipeline không tự nộp thay đội.

## 10. Tệp và liên kết

- Script: `scripts/summarize_official_videos.py`
- Workflow: `.github/workflows/summarize-full-batch1.yml`
- Evidence frames: `competition_sotuyen1/EVIDENCE_FRAMES.md`
- Evidence artifacts: `competition_sotuyen1/EVIDENCE_ARTIFACTS.md`
- Drive notes: `drive_ingest/README.md`
- Branch sửa: `https://github.com/neornotl/aic-batch1/tree/codex/summary-parser-repair`
- Root Drive: `https://drive.google.com/drive/folders/1E8u-YURTRexdR1Ax4Hu2aq7Vt646VQuN`
