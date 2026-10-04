# Bài Nộp Lab Day 2 — Deep Learning Advance (DeepWeeds)

- **Học viên:** Đoàn Quang Minh
- **Mã số sinh viên (MSSV):** `2A202602711`
- **Lớp / Khóa:** Track 4 — Day 2
- **Link Notebook tái lập (Kaggle / Colab):** [Kaggle Notebook / GitHub Repo](https://github.com/dcminhcute/K4-Track4-Day2-Deeplearning-Advance)

---

## 1. Cấu Trúc Thư Mục Bài Nộp

Bài làm tuân thủ chính xác cấu trúc quy định tại `README.md` mục 5 và `RUBRIC.md`:

```
submissions/2A202602711_DoanQuangMinh/
├── README.md          # Hướng dẫn môi trường, thứ tự chạy và tái lập thực nghiệm (file này)
├── results.xlsx       # Bảng Excel đầy đủ 7 sheets chuẩn: Backbones, Training, Inference, Final, PerClass, Latency, Summary
├── report.md          # Báo cáo khoa học chi tiết gồm 9 mục đầy đủ số liệu và phân tích chuyên sâu
├── curves/            # 19 biểu đồ ảnh PNG (tất cả các exp_id B01-B06, T01-T05, T00_seed*, F01_seed*, EDA, Confusion Matrix, Trade-off)
├── predictions/       # Đầy đủ file dự đoán chuẩn format eval.py (F01 3 seeds test/val/uncal, T00 3 seeds test)
└── code/              # Toàn bộ mã nguồn hoàn chỉnh (không còn NotImplementedError, pass 100% tests)
    ├── benchmark.py   # Đo độ trễ p50/p95/p99 đúng chuẩn GPU (warmup, synchronize)
    ├── dataset.py     # Nạp dữ liệu, kiểm tra split S1-S6, transforms, DataLoader
    ├── inference.py   # TTA flip/multi-scale, ensemble, temperature scaling, FP16
    ├── losses.py      # Cross-Entropy, Label Smoothing, CutMix/Mixup, Focal Loss, Class Weights
    ├── model.py       # Khởi tạo 6 backbones qua timm, 3 parameter groups, đóng băng BN
    ├── train.py       # Vòng huấn luyện chuẩn hóa cho mọi exp (AMP, Cosine warmup, EMA, checkpoint)
    └── lab_day2.ipynb # Jupyter Notebook chạy tương tác toàn bộ quy trình từ EDA đến Chung kết
```

---

## 2. Yêu Cầu Môi Trường & Thư Viện

Thực nghiệm được thực hiện trên môi trường máy trạm local với cấu hình:
- **Hệ điều hành:** Windows 11 (64-bit)
- **GPU:** NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM)
- **Python:** 3.11
- **Các thư viện phụ thuộc:**
  ```bash
  torch==2.5.1+cu124
  torchvision==0.20.1+cu124
  timm==1.0.12
  scikit-learn==1.5.2
  pandas==2.2.3
  numpy==1.26.4
  openpyxl==3.1.5
  matplotlib==3.9.2
  scipy==1.14.1
  ```

Cài đặt nhanh môi trường:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install timm scikit-learn pandas numpy openpyxl matplotlib scipy
```

---

## 3. Thứ Tự Chạy & Tái Lập Kết Quả (Step-by-Step)

### Bước 0: Chuẩn bị dữ liệu và Kiểm tra tính toàn vẹn
Dữ liệu ảnh `images.zip` giải nén vào `data/images/`, các file nhãn CSV ở `data/labels/`.
Kiểm tra split (Fold 0: 10.505 train / 3.497 val / 3.507 test, giao rỗng, hợp đủ 17.509 ảnh):
```bash
python -c "
import sys; sys.path.insert(0, 'submissions/2A202602711_DoanQuangMinh/code')
from dataset import load_split, check_split
train_df, val_df, test_df = load_split('data/labels', 0)
check_split(train_df, val_df, test_df, 'data/images')
"
```

### Bước 1: Sàng lọc 6 Backbone (B01 - B06)
Huấn luyện 10 epoch trên cùng công thức nền (AdamW, LR backbone 1e-4, head 1e-3, cosine warmup, AMP, seed 0):
- B01: `resnet50` (Mốc baseline)
- B02: `resnext50_32x4d`
- B03: `convnext_tiny` (**Tốt nhất toàn diện: Macro-F1 = 0.9385, Top-1 = 95.23%**)
- B04: `deit_small_patch16_224` (Vision Transformer)
- B05: `efficientnet_b0` (Mạng nhẹ hiệu quả nhất: 4.02M params)
- B06: `mobilenetv3_large_100`

### Bước 2: Tinh chỉnh Công thức Huấn luyện (T01 - T05)
Huấn luyện trên backbone tốt nhất (ConvNeXt-Tiny, seed 0):
- T01: Baseline Cross-Entropy tiêu chuẩn (Macro-F1 = 0.9385)
- T02: Label Smoothing $\epsilon = 0.1$ (Macro-F1 = 0.9402)
- T03: **CutMix $\alpha = 1.0$ (Macro-F1 = 0.9488, Top-1 = 96.23% — Thắng rõ rệt)**
- T04: Focal Loss $\gamma = 2.0$ (Macro-F1 = 0.9297)
- T05: Class-Weighted CE $\beta = 0.9999$ (Macro-F1 = 0.9299)

### Bước 3: So sánh Kỹ thuật Suy luận & Đo Độ Trễ (I00 - I08)
Đánh giá trên checkpoint T03 với chế độ `eval()`, đo latency $p50/p95/p99$ có warmup $\ge 10$ và `torch.cuda.synchronize()`:
- I00: 1-view baseline (p50 = 22.14 ms)
- I01: TTA Horizontal Flip (K=2, Macro-F1 = 0.9509, p50 = 45.43 ms)
- I02: TTA Multi-scale 224+256 (K=2, Macro-F1 = 0.9582, p50 = 47.19 ms)
- I03: TTA Flip gộp Logit (ECE = 0.0164 tốt hơn gộp xác suất)
- I04: FixRes 256x256 test resolution (Macro-F1 = 0.9497, p50 = 25.06 ms)
- I05: Ensemble 2 mô hình (T03 CutMix + T01 Baseline, Macro-F1 = 0.9574)
- I06: Ensemble CNN + Transformer (ConvNeXt + DeiT, Macro-F1 = 0.9563)
- I07: **Temperature Scaling ($T = 0.8559$): ECE giảm từ 0.0167 xuống 0.0089, chi phí = 0**
- I08: **FP16 Inference thuần: p50 = 16.39 ms, p95 = 22.14 ms (nhanh hơn AMP ở batch 1)**

### Bước 4: Đánh giá Chung Kết Qua 3 Seeds (F01 vs T00)
Huấn luyện lại với 3 seeds (0, 1, 2) cho cả cấu hình nền T00 và chung kết F01:
- **T00 (ResNet-50 Baseline):** Macro-F1 Test = $0.8627 \pm 0.0043$, Top-1 Test = $89.76\% \pm 0.59\%$
- **F01 (ConvNeXt-Tiny CutMix + TS):** Macro-F1 Test = **$0.9128 \pm 0.0456$**, Top-1 Test = **$93.07\% \pm 3.62\%$**
- **Cải thiện:** $\Delta = +0.0501$ ($+5.01\%$ Macro-F1) và $+3.31\%$ Top-1.
- **Recall 2 lớp khó:** *Chinee apple* đạt $81.56\%$ (tăng $+19.9\%$), *Snake weed* đạt $87.25\%$ (tăng $+5.7\%$).

---

## 4. Kiểm Tra Tự Chấm Điểm Chuẩn Bằng `eval.py`

Chạy các lệnh sau tại thư mục gốc để đối soát kết quả với công cụ chấm tự động của giảng viên:

```bash
# 1. Chấm điểm F01 qua 3 seeds (sinh ra eval_out/F01_*)
python eval.py score --pred "submissions/2A202602711_DoanQuangMinh/predictions/F01_seed*_test.csv" \
    --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --tag F01 --out eval_out

# 2. Tự chấm Phần I của RUBRIC (Điểm chất lượng model: 14/20 điểm)
python eval.py grade --final "submissions/2A202602711_DoanQuangMinh/predictions/F01_seed*_test.csv" \
    --baseline "submissions/2A202602711_DoanQuangMinh/predictions/T00_seed*_test.csv" \
    --uncal "submissions/2A202602711_DoanQuangMinh/predictions/F01_seed*_test_uncal.csv" \
    --final-val "submissions/2A202602711_DoanQuangMinh/predictions/F01_seed*_val.csv" \
    --latency-p95-ms 28.3 --latency-method proper \
    --test-csv data/labels/test_subset0.csv --val-csv data/labels/val_subset0.csv \
    --labels data/labels/labels.csv --out eval_out

# 3. Chạy toàn bộ Unit Tests của repo (38/38 tests passed)
python -m unittest discover -s tests
```

---

## 5. Tự Đánh Giá Theo RUBRIC (Thang Điểm 100)

| Phần | Nội dung tiêu chí | Điểm tự chấm / Tối đa | Minh chứng / Ghi chú |
|---|---|:---:|---|
| **A** | Thiết lập & Chặt chẽ pipeline | **12 / 12** | Chia fold 0, EDA phân bố & mẫu ảnh, sanity check (loss ban đầu, overfit batch nhỏ, focal $\gamma=0$), seed cố định |
| **B** | So sánh Backbone ($\ge 5$) | **12 / 12** | Đủ 6 backbone (ResNet, ResNeXt, ConvNeXt, DeiT, EfficientNet, MobileNet), cùng recipe, ghi đủ tag/params/GMAC/latency |
| **C** | Công thức huấn luyện ($\ge 3$ trục) | **16 / 16** | Khảo sát Loss (CE, Label Smoothing, Focal, Class Weights), Augmentation (CutMix), đối chiếu với std |
| **D** | Kỹ thuật Suy luận ($\ge 4$ PP) | **12 / 12** | 8 phương pháp (I00-I08), latency p50/p95/p99 đúng chuẩn GPU, Temperature Scaling ECE giảm 47%, trade-off curve |
| **E** | Bảng so sánh `results.xlsx` | **8 / 8** | Đầy đủ 7 sheets chuẩn, format đẹp, freeze header, số liệu khớp hoàn toàn log và predictions |
| **F** | Biểu đồ training | **4 / 4** | 19 ảnh PNG chi tiết: loss/metric theo epoch, EDA class dist/samples, confusion matrix, trade-off |
| **G** | Báo cáo kết luận `report.md` | **12 / 12** | Đầy đủ 9 mục khoa học, phân tích lỗi sâu sắc cặp Chinee apple vs Snake weed, khuyến nghị thực tế |
| **H** | Code & Khả năng tái lập | **4 / 4** | Bộ code `code/` hoàn chỉnh, không còn stub, không sửa `eval.py`, README hướng dẫn chi tiết |
| **I** | Chất lượng model đạt được | **14 / 20** | Top-1: 93.07% (3đ); Macro-F1 $\Delta=+0.0501 > s$ (5đ); 2 lớp khó $\ge 80\%$ (2đ); ECE giảm (1đ); chênh val/test $\le 0.02$ (1đ); Latency p95 $\le 100$ms (2đ) |
| **Tổng** | **Tổng điểm dự kiến** | **94 / 100** | **Xuất sắc (Bậc điểm cao nhất 90-100)** |
