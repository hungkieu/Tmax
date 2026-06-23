# Hướng dẫn sử dụng — Dự báo nhiệt METAR kế tiếp (Next-METAR)

Tài liệu này hướng dẫn vận hành thực tế hệ thống dự báo nhiệt độ METAR kế tiếp
cho `RKSI` / `RKPK`, đọc dữ liệu **trực tiếp từ MongoDB** (không còn dùng HTTP API).

Tài liệu thiết kế kỹ thuật: [next-metar-temp.md](next-metar-temp.md).

---

## 1. Tổng quan luồng

```
MongoDB (nhiệt quan trắc KMA)
  ├─ airporttemperaturecurrents  ──► PREDICT (nhiệt live đầu vào)
  └─ airporttemperaturehistories ──► VERIFY  (nhiệt thực tế để chấm điểm)

predict  → ghi 1 dòng vào prediction log (status: pending)
verify   → tìm nhiệt thực tế, điền actual + đánh giá (status: verified) + tính health
```

- **Predict**: lấy nhiệt mới nhất của 1 sân bay, dự báo nhiệt (số nguyên °C) của
  METAR đều đặn kế tiếp (`:00`/`:30` với RKSI, `:00` với RKPK).
- **Verify**: sau khi qua mốc METAR, đối chiếu dự báo với nhiệt quan trắc thật.
- Tất cả ghi chung vào file log `artifacts/next_metar_temp/next_metar_temp_predictions.jsonl`.

---

## 1b. Cách dùng nhanh nhất — qua UI (1 nút Run)

Mở dashboard và dùng tab **Next-METAR** — bạn chỉ cần bấm **một nút**:

```powershell
uv run rksi-ui
```

Trong tab **Next-METAR**, bấm **▶ Run**. Hệ thống tự động làm tất cả bên dưới:

1. Đọc nhiệt live từ MongoDB cho mọi sân bay Korea (RKSI, RKPK).
2. Dự báo nhiệt METAR kế tiếp cho từng sân bay.
3. Chấm điểm các dự báo đang chờ (so với nhiệt quan trắc thật trong MongoDB).

UI hiển thị rõ **từng việc đã làm** (✅/⚠️/⏭️/❌), **sức khỏe model** (MAE, độ
chính xác…) và **bảng dự báo gần đây**. Bạn không phải bấm predict/verify riêng,
cũng không cần gõ tham số. Các mục dưới đây dành cho khi bạn muốn chạy bằng dòng
lệnh hoặc tự động hóa.

## 2. Chuẩn bị (làm 1 lần)

### 2.1. Khai báo kết nối MongoDB

Biến môi trường **duy nhất**: `MONGODB_URI` (tên database nằm trong URI).

Cách 1 — file `.env` ở gốc repo (khuyên dùng cho máy cá nhân; đã được gitignore):

```text
# E:\Git\ML\.env
MONGODB_URI=mongodb+srv://user:pass@cluster.xxxxx.mongodb.net/<dbName>?retryWrites=true&w=majority
```

Tham khảo mẫu: [.env.example](../.env.example). File được nạp tự động bởi
`rksi_tmax.mongo_source` (qua `python-dotenv`).

Cách 2 — biến môi trường hệ thống (CI / scheduled task):

```powershell
$env:MONGODB_URI = "mongodb+srv://..."   # phiên hiện tại
setx MONGODB_URI "mongodb+srv://..."     # lưu vĩnh viễn (mở terminal mới mới có hiệu lực)
```

> Project chỉ **đọc** → nên tạo user **read-only** trên Atlas cho URI này.

### 2.2. Cài đặt phụ thuộc / script CLI

```powershell
uv sync
```

Lệnh này cài project vào venv và sinh các file script `.exe` trong
`.venv\Scripts\` (vd `rksi-predict-next-metar-temp.exe`).

> **Lưu ý Windows:** `uv sync` ghi đè toàn bộ `.exe` của project, kể cả
> `rksi-ui.exe`. Nếu **UI Streamlit đang chạy** thì file này bị khóa và sync sẽ
> lỗi `being used by another process`. → **Tắt UI trước khi `uv sync`.**
> Chi tiết ở mục [Xử lý sự cố](#7-xử-lý-sự-cố).
>
> Chỉ cần `uv sync` lại khi đổi `pyproject.toml` (thêm script / dependency).
> Chạy predict/verify hằng ngày **không** cần sync và **không** cần tắt UI.

---

## 3. Tham chiếu lệnh CLI

### 3.1. `rksi-predict-next-metar-temp` — dự báo 1-4 METAR kế tiếp

Lệnh dự báo **nhiều mốc** (mặc định 4 METAR kế tiếp). Kết quả trả về mảng
`predictions`, mỗi phần tử là 1 horizon (bước), và **mỗi horizon ghi 1 dòng**
vào log để verify độc lập.

| Tham số | Bắt buộc | Mặc định | Ý nghĩa |
|---|---|---|---|
| `--station` | ✅ | — | `RKSI` hoặc `RKPK` |
| `--horizons` | ❌ | `1 2 3 4` | Các bước cần dự báo (vd `--horizons 1 2`) |
| `--temp-c` | ❌ | đọc từ Mongo | Nhiệt live (°C). Bỏ trống → lấy mới nhất từ `airporttemperaturecurrents` |
| `--observed-at` | ❌ | đọc từ Mongo | Thời điểm quan trắc ISO-8601 có timezone (vd `2026-06-23T13:00:00+09:00`) |
| `--tmax-signal-c` | ❌ | model tự impute | Tín hiệu Tmax dự báo trong ngày (tùy chọn) |
| `--model` | ❌ | `artifacts/next_metar_temp/next_metar_temp_model.joblib` | Đường dẫn model |
| `--log` | ❌ | `artifacts/next_metar_temp/next_metar_temp_predictions.jsonl` | File log dự báo (append) |

Khoảng cách giữa các mốc theo lịch METAR: **RKSI 30 phút** (vd 14:00, 14:30,
15:00, 15:30), **RKPK 60 phút** (vd 14:00, 15:00, 16:00, 17:00).

Nếu bỏ cả `--temp-c` lẫn `--observed-at` mà Mongo không có dữ liệu cho sân bay
đó → lệnh báo lỗi rõ ràng và dừng.

### 3.2. `rksi-verify-next-metar-temp` — đối chiếu dự báo với thực tế

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--station` | `ALL` | `ALL` / `RKSI` / `RKPK` |
| `--from-db` | tắt | **Bật để đọc nhiệt thực tế từ Mongo** (`airporttemperaturehistories`) |
| `--hours` | 48 | Số giờ history nạp từ Mongo (chỉ dùng khi `--from-db`) |
| `--tolerance-seconds` | 300 | Khoảng lệch tối đa khi khớp slot METAR với observation gần nhất |
| `--window` | 100 | Số dự báo gần nhất dùng để tính health |
| `--log` | mặc định | File log cần verify |
| `--fetch` | tắt | (Đường METAR cũ) tải METAR mới trước khi verify |
| `--metar-file` | `data/shared/metar.txt` | (Đường METAR cũ) file METAR |
| `--reference-date` | hôm nay (UTC) | (Đường METAR cũ) ngày tham chiếu khi import METAR |

> Dữ liệu Mongo là **stream liên tục**, không rơi đúng mốc `:00/:30`. Vì vậy mỗi
> dự báo được khớp với observation **gần nhất** `next_metar_at` trong khoảng
> `--tolerance-seconds`. Nếu không có observation nào trong cửa sổ đó → bỏ qua
> (giữ `pending`), lần verify sau sẽ khớp khi dữ liệu về.

### 3.3. `rksi-build-next-metar-dataset` — dựng dataset huấn luyện

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--station` | `ALL` | Build 1 trạm hay gộp cả Korea |
| `--output` | parquet mặc định | Nơi ghi dataset |

### 3.4. `rksi-train-next-metar-temp` — huấn luyện + (tùy chọn) promote

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--dataset` | parquet mặc định | Dataset đầu vào |
| `--model` | joblib mặc định | Nơi ghi model active |
| `--metrics` | json mặc định | Nơi ghi metrics active |
| `--promote` / `--no-promote` | promote bật | Có ghi đè model active nếu candidate không tệ hơn |

> Build & train dùng **dữ liệu lịch sử local** (CSV/duckdb), không phải Mongo.
> Mongo chỉ thay phần **predict + verify** vận hành live.

---

## 4. Quy trình vận hành hằng ngày

```powershell
# 1) Dự báo nhiệt METAR kế tiếp cho RKSI (đọc nhiệt live từ Mongo)
uv run rksi-predict-next-metar-temp --station RKSI

# 2) Vài phút sau, khi đã qua mốc :00/:30 → chấm điểm dự báo
uv run rksi-verify-next-metar-temp --from-db --hours 48
```

Kết quả verify trả về khối `health` (mae, bias, exact/within-1C accuracy, coverage…)
để theo dõi chất lượng model gần đây. Ngưỡng "unhealthy" xem ở
[next-metar-temp.md](next-metar-temp.md#verification-and-health).

Ví dụ truyền tham số tùy chọn:

```powershell
# Có tín hiệu Tmax trong ngày
uv run rksi-predict-next-metar-temp --station RKSI --tmax-signal-c 31

# Nhập tay nhiệt + thời điểm (không đọc Mongo)
uv run rksi-predict-next-metar-temp --station RKSI --observed-at 2026-06-23T13:00:00+09:00 --temp-c 29.5

# Verify chỉ RKSI, nới cửa sổ khớp thời gian lên 10 phút
uv run rksi-verify-next-metar-temp --station RKSI --from-db --tolerance-seconds 600
```

---

## 5. Định kỳ huấn luyện lại (không bắt buộc thường xuyên)

```powershell
uv run rksi-build-next-metar-dataset            # dựng lại dataset từ dữ liệu lịch sử
uv run rksi-train-next-metar-temp               # train + promote nếu không xấu đi
uv run rksi-train-next-metar-temp --no-promote  # chỉ train ra candidate, không thay model
```

---

## 6. Đọc dữ liệu Mongo bằng Python (khi cần kiểm tra nhanh)

```python
from datetime import datetime, timezone, timedelta
from rksi_tmax.mongo_source import get_current_temperature, get_temperature_history

cur = get_current_temperature("RKSI")
print(cur.temp_c, cur.observed_at_local_iso)

since = datetime.now(timezone.utc) - timedelta(hours=24)
hist = get_temperature_history("RKSI", since=since)   # DataFrame: observed_at_utc, valid_local, temp_c
print(len(hist), hist.tail())
```

ICAO luôn được `.upper()` trước khi query. Client MongoDB được cache (singleton)
theo URI để tránh mở nhiều pool.

---

## 7. Xử lý sự cố

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `uv sync` lỗi `...rksi-ui.exe ... being used by another process` | UI Streamlit đang chạy khóa file `.exe` (Windows) | Tắt UI (Ctrl+C cửa sổ chạy `uv run rksi-ui`, hoặc `Stop-Process`), rồi `uv sync` lại |
| `uv run rksi-...` báo `program not found` | Script chưa được cài (chưa `uv sync` sau khi thêm vào pyproject) | Tắt UI → `uv sync`. Hoặc chạy tạm qua module (xem dưới) |
| `MONGODB_URI is not set` | Thiếu env | Tạo `.env` ở gốc repo hoặc set biến môi trường |
| `No live temperature found in MongoDB for RKPK` | Mongo chưa có doc cho sân bay đó | Upstream chưa ghi ICAO này; tạm thời truyền `--temp-c`/`--observed-at` thủ công |
| Verify luôn `pending`, `verified: 0` | Chưa qua mốc METAR, hoặc lệch thời gian > tolerance | Chạy lại sau mốc `:00/:30`; tăng `--tolerance-seconds` nếu cần |

### Chạy tạm khi không muốn tắt UI (không cần sync)

```powershell
uv run --no-sync python -c "from rksi_tmax.cli import predict_next_metar_temp_main as m; import sys; sys.argv=['x','--station','RKSI']; m()"

uv run --no-sync python -c "from rksi_tmax.cli import verify_next_metar_temp_main as m; import sys; sys.argv=['x','--from-db','--hours','48']; m()"
```

Cách này gọi thẳng hàm Python trong source, không đụng tới file `.exe` → không
xung đột khóa file, không cần tắt UI. Truyền tham số bằng cách chèn vào `sys.argv`.

---

## 8. File / vị trí liên quan

| Thành phần | Đường dẫn |
|---|---|
| Truy cập MongoDB | [src/rksi_tmax/mongo_source.py](../src/rksi_tmax/mongo_source.py) |
| Logic model / predict / verify | [src/rksi_tmax/next_metar_temp.py](../src/rksi_tmax/next_metar_temp.py) |
| Điều phối 1-nút-Run (UI) | [src/rksi_tmax/services/next_metar_service.py](../src/rksi_tmax/services/next_metar_service.py) |
| Tab UI Next-METAR | [src/rksi_tmax/ui_tabs/next_metar_tab.py](../src/rksi_tmax/ui_tabs/next_metar_tab.py) |
| Khai báo lệnh CLI | [src/rksi_tmax/cli.py](../src/rksi_tmax/cli.py) |
| Tài liệu thiết kế | [docs/next-metar-temp.md](next-metar-temp.md) |
| Mẫu env | [.env.example](../.env.example) |
| Prediction log | `artifacts/next_metar_temp/next_metar_temp_predictions.jsonl` |
