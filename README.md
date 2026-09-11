# AIC Video Retrieval Pipeline

Bộ công cụ Python để lập chỉ mục, tìm kiếm, đánh giá và đóng gói kết quả cho dữ liệu video AIC. Repo này ưu tiên mã nguồn và hướng dẫn tái sử dụng; các file chạy tạm, checkpoint và gói kết quả cũ không được lưu trong Git.

> **Phạm vi bằng chứng:** mã nguồn và kết quả chạy cục bộ trong repo không tự chứng minh đây là kết quả nộp chính thức. Khi tạo submission, chỉ dùng `frame_id` / `OFFICIAL_FRAME_ID` lấy từ cơ sở dữ liệu chuẩn. Không suy ra ID từ tên file, số thứ tự keyframe, FPS hoặc timestamp.

## Thành phần chính

| Thư mục / file | Vai trò |
| --- | --- |
| `aic_pipeline/` | CLI chính: manifest, index, search, evaluate, submission và package |
| `competition_sotuyen1/` | Mã và cấu hình phục vụ vòng sơ tuyển |
| `drive_ingest/` | Nhập dữ liệu từ Google Drive |
| `evidence_review/` | Công cụ rà soát bằng chứng/kết quả |
| `scripts/` | Tiện ích xử lý dữ liệu có thể chạy độc lập |
| `run_batch.py` | Bộ xử lý ảnh hàng loạt cũ; cần ba đối số và biến môi trường riêng |
| `work/` | Dữ liệu sinh ra khi chạy; không phải mã nguồn và không commit |

## Bắt đầu nhanh

Yêu cầu Python 3.12. FFmpeg và CUDA chỉ cần cho các bước video/GPU tương ứng.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m aic_pipeline --help
```

Luồng cơ bản:

```powershell
# 1. Tạo manifest từ các nguồn dữ liệu
python -m aic_pipeline manifest --results <thu-muc-ket-qua> --maps <thu-muc-map> --output work/aic_pipeline/manifest.jsonl

# 2. Lập chỉ mục SQLite
python -m aic_pipeline index --manifest work/aic_pipeline/manifest.jsonl --database work/aic_pipeline/aic.sqlite

# 3. Tìm kiếm
python -m aic_pipeline search "noi dung can tim" --database work/aic_pipeline/aic.sqlite --limit 20

# 4. Chạy danh sách truy vấn và đóng gói CSV
python -m aic_pipeline batch --queries <queries.json> --database work/aic_pipeline/aic.sqlite --output <thu-muc-csv>
python -m aic_pipeline package --csv-dir <thu-muc-csv> --output <submission.zip>
```

Xem đúng tham số của từng lệnh bằng `python -m aic_pipeline <lenh> --help`.

## Dữ liệu và thông tin nhạy cảm

- Đặt dữ liệu tải về, chỉ mục, vector, checkpoint và kết quả chạy trong `work/`.
- Không commit khóa API, file `.env`, service-account JSON hoặc dữ liệu cuộc thi không được phép công khai.
- `run_batch.py` đọc khóa từ `AIC2` và `AIC3`; cú pháp là `python run_batch.py <thu-muc-anh> <output.jsonl> <failed.json>`. Đây là script batch cũ, không phải CLI khuyến nghị cho người mới.
- Luôn kiểm tra CSV/ZIP bằng dữ liệu chuẩn trước khi nộp. File được tạo thành công chỉ chứng minh cấu trúc cục bộ, không chứng minh đã nộp lên cổng thi.

## Kiểm tra nhanh cho người phát triển

```powershell
python -m compileall -q aic_pipeline scripts
python -m pytest
```

Nếu chỉ muốn tìm hiểu repo, hãy bắt đầu từ `aic_pipeline/__main__.py`, sau đó đọc module tương ứng với lệnh cần dùng.
