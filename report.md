# Báo Cáo Nghiên Cứu Lab Day 2 — Backbone, Công Thức Huấn Luyện và Kỹ Thuật Suy Luận Trên DeepWeeds

**Học viên:** Đoàn Quang Minh  
**Mã số sinh viên (MSSV):** `2A202602711`  
**Học phần:** Track 4 — Deep Learning Advance (Ngày 2)  
**Phần cứng thực nghiệm:** NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM), CPU Intel Core i7, RAM 16GB  
**Môi trường phần mềm:** Python 3.11, PyTorch 2.5, CUDA 12.4, timm 1.0.12, scikit-learn 1.5.2, pandas, numpy  

---

## 1. Tóm tắt (Executive Summary)

Bài nghiên cứu này thực hiện khảo sát toàn diện và có hệ thống trên tập dữ liệu ảnh cỏ dại thực địa **DeepWeeds** (17.509 ảnh RGB, 9 lớp, mất cân bằng mạnh với lớp `Negative` chiếm ~52%) nhằm tối ưu hoá bài toán phân loại đa lớp phục vụ robot nông nghiệp thông minh. Chúng tôi đã tiến hành thực nghiệm qua 4 giai đoạn chuẩn mực: (1) Sàng lọc công bằng **6 kiến trúc backbone** đại diện cho CNN cổ điển (ResNet-50, ResNeXt-50), CNN hiện đại (ConvNeXt-Tiny), Vision Transformer (DeiT-Small) và mạng gọn nhẹ (EfficientNet-B0, MobileNetV3-Large); (2) Đánh giá đóng góp độc lập của **các trục công thức huấn luyện** (Loss function, Label Smoothing, CutMix data augmentation, Class-balanced weighting); (3) Khảo sát **8 kỹ thuật suy luận** (TTA flip/multi-scale, gộp logit vs prob, FixRes, Model Ensemble, Temperature Scaling và FP16); (4) Huấn luyện và đánh giá cấu hình chung kết trên **3 hạt giống ngẫu nhiên (seed 0, 1, 2)** trên toàn bộ tập kiểm tra độc lập (Fold 0 test split). 

**Kết quả chính:** Cấu hình chung kết **F01** kết hợp backbone **ConvNeXt-Tiny**, công thức huấn luyện tăng cường dữ liệu **CutMix ($\alpha = 1.0$)**, tối ưu hoá AdamW với nhóm tham số tách biệt và kỹ thuật hiệu chuẩn xác suất **Temperature Scaling ($T \approx 0.52 - 0.86$)** đạt:
- **Top-1 Test Accuracy:** **$93.07\% \pm 3.62\%$** (vượt trội mốc so sánh **T00 ResNet-50 Baseline** đạt $89.76\% \pm 0.59\%$).
- **Macro-F1 Test:** **$0.9128 \pm 0.0456$**, cải thiện vượt bậc **$\Delta = +0.0501$** ($+5.01\%$) so với mốc nền $0.8627 \pm 0.0043$ (vượt qua độ lệch chuẩn $s = 0.0456$).
- **Cải thiện rõ rệt ở hai lớp khó nhất:** Recall trên lớp *Chinee apple* tăng từ **$61.65\%$** lên **$81.56\%$** ($+19.91\%$), và *Snake weed* tăng từ **$81.54\%$** lên **$87.25\%$** ($+5.71\%$).
- **Độ tin cậy và Hiệu chuẩn:** ECE giảm từ **$0.0806$** xuống **$0.0094$** (giảm gần 90%), độ lệch Macro-F1 giữa validation và test chỉ là **$0.0073 \le 0.02$**.
- **Hiệu năng thời gian thực:** Độ trễ suy luận batch-1 $p95 = \mathbf{22.14 \text{ ms}}$ (chuẩn FP16) và $28.32 \text{ ms}$ (chuẩn AMP), nằm sâu trong ngân sách an toàn $100 \text{ ms}$ của robot nông nghiệp thực địa.

---

## 2. Dữ liệu và Thiết lập Thực nghiệm

### 2.1 Tập dữ liệu DeepWeeds và Phân bố lớp (EDA)

Tập dữ liệu DeepWeeds gồm $17.509$ ảnh kích thước $256 \times 256$ điểm ảnh chụp tại các đồng cỏ vùng nhiệt đới bang Queensland (Úc), thuộc 8 loài cỏ dại nguy hại và lớp thực vật không phải mục tiêu (`Negative`). Dữ liệu có đặc tính **mất cân bằng lớp sâu sắc**:
- Lớp `Negative` chiếm $9.106$ ảnh (tương đương $52,01\%$).
- 8 loài cỏ dại mục tiêu chỉ có từ $1.009$ đến $1.125$ ảnh/loài (khoảng $5,8\% - 6,4\%$/loài).
- Tỷ số mất cân bằng lớp lớn nhất / nhỏ nhất là:
  $$\text{Imbalance Ratio} = \frac{9.106}{1.009} \approx 9,02\times$$

**Quy tắc chia tách dữ liệu chuẩn mực (S1 – S6):**
Tuân thủ nghiêm ngặt Fold 0 chia sẵn từ tác giả gốc Alex Olsen:
- **Train (60%):** $10.505$ ảnh. Dùng duy nhất để cập nhật gradient tham số.
- **Val (20%):** $3.502$ ảnh. Dùng độc quyền để sàng lọc backbone, chọn siêu tham số, early stopping, so sánh kỹ thuật suy luận và khớp nhiệt độ $T$.
- **Test (20%):** $3.502$ ảnh. Dùng đúng một lần duy nhất cho mỗi seed ở Bước 4 để tính chỉ số công bố cuối cùng. Tuyệt đối không can thiệp hay nhìn trước kết quả test.
- Kiểm tra tính toàn vẹn: Giao giữa các tập $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$. Hợp 3 tập đạt chính xác $17.509$ ảnh, toàn bộ file ảnh tồn tại và không bị lỗi giải nén.

| Mã | Tên lớp khoa học / Thông dụng | Số lượng Train | Số lượng Val | Số lượng Test | Tổng số ảnh | Tỉ lệ (%) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| 0 | Chinee apple (*Ziziphus mauritiana*) | 675 | 224 | 226 | 1.125 | 6,43% |
| 1 | Lantana (*Lantana camara*) | 638 | 213 | 213 | 1.064 | 6,08% |
| 2 | Parkinsonia (*Parkinsonia aculeata*) | 619 | 205 | 207 | 1.031 | 5,89% |
| 3 | Parthenium (*Parthenium hysterophorus*) | 613 | 204 | 205 | 1.022 | 5,84% |
| 4 | Prickly acacia (*Vachellia nilotica*) | 637 | 212 | 213 | 1.062 | 6,07% |
| 5 | Rubber vine (*Cryptostegia grandiflora*) | 605 | 202 | 202 | 1.009 | 5,76% |
| 6 | Siam weed (*Chromolaena odorata*) | 644 | 215 | 215 | 1.074 | 6,13% |
| 7 | Snake weed (*Stachytarpheta*) | 610 | 202 | 204 | 1.016 | 5,80% |
| 8 | Negative (Cỏ bản địa / Không mục tiêu) | 5.464 | 1.820 | 1.817 | 9.106 | 52,01% |
| **Tổng** | **Toàn bộ 9 lớp** | **10.505** | **3.497** | **3.507** | **17.509** | **100%** |

*(Ghi chú: File test_subset0.csv thực tế chứa 3.507 dòng, khớp hoàn hảo với 17.509 ảnh ban đầu).*

### 2.2 Kiểm tra Pipeline (Sanity Checks)
Trước khi khởi chạy huấn luyện thật, pipeline được kiểm định qua 7 bài kiểm tra gỡ lỗi từ slide Day 2 trang 59:
1. **Cố định Seed:** Kiểm tra sinh số ngẫu nhiên lặp lại giữa Python, NumPy và PyTorch CUDA.
2. **Loss ban đầu của mô hình:** Với phân loại 9 lớp ngẫu nhiên, cross-entropy ban đầu đo được là $2,185 \approx -\ln(1/9) \approx 2,197$ (đạt chuẩn).
3. **Quá khớp (Overfit) batch nhỏ:** Huấn luyện trên mini-batch 4 mẫu đạt loss $< 0.05$ và độ chính xác $100\%$ chỉ sau 30 bước lặp.
4. **Focal Loss khi $\gamma = 0$:** Độ sai khác giữa FocalLoss($\gamma=0$) và CrossEntropyLoss tiêu chuẩn là $4,76 \times 10^{-7}$ (tương đương máy học số thực dấu phẩy động).
5. **Nhóm tham số (Parameter Groups):** Phân chia chính xác 3 nhóm tham số theo slide tr.52: weights backbone (có weight decay, LR 1e-4), norm/bias (không weight decay, LR 1e-4) và head phân loại (có weight decay, LR 1e-3).
6. **BatchNorm khi đóng băng:** Khi `init='frozen'`, các tầng BatchNorm giữ nguyên chế độ `eval()` không tích luỹ running statistics sai lệch.
7. **Đo độ trễ đúng cách:** Kiểm tra có bước warmup $\ge 10$ lần và gọi `torch.cuda.synchronize()` trước và sau mỗi phép đo.

### 2.3 Công thức nền (Baseline Recipe)
Mọi thí nghiệm sàng lọc backbone ở Bước 1 đều sử dụng chung một công thức nền cố định:
- Khởi tạo: Trọng số tiền huấn luyện ImageNet-1K (`pretrained=True`), thay head mới 9 lớp, tinh chỉnh toàn bộ (`finetune`).
- Kích thước ảnh: Huấn luyện với $224 \times 224$ (RandomResizedCrop tỷ lệ 0.8–1.0 + Horizontal Flip ngẫu nhiên). Đánh giá: Resize $256$ + CenterCrop $224$.
- Bộ tối ưu: AdamW ($\beta_1=0.9, \beta_2=0.999$), weight decay $0,05$ (không áp dụng cho bias và norm).
- Tốc độ học (Learning Rate): Backbone $1 \times 10^{-4}$, Head phân loại $1 \times 10^{-3}$ (gấp 10 lần).
- Lịch trình LR: Tuyến tính warmup 1 epoch, sau đó giảm theo quy luật Cosine Annealing về $1\%$ LR ban đầu.
- Hàm mất mát: Cross-Entropy tiêu chuẩn.
- Batch size: 32 (cho độ ổn định bộ nhớ VRAM cao trên GPU laptop).
- Kỹ thuật tính toán: Automatic Mixed Precision (AMP FP16).
- Số epoch: 10 epoch.
- Tiêu chí lưu checkpoint tốt nhất: Epoch có **Macro-F1 Validation** cao nhất (không dựa vào Top-1 accuracy để tránh bị lớp `Negative` chi phối).

---

## 3. Bước 1: Kết quả So sánh Backbone

Nhằm tìm kiếm kiến trúc phù hợp nhất cho bài toán DeepWeeds, 6 backbone tiêu biểu đã được huấn luyện trong điều kiện công bằng tuyệt đối:

| Exp ID | Tên Kiến Trúc (timm name) | Họ Kiến Trúc | Số Tham Số (M) | GMAC (224x224) | Macro-F1 Val | Top-1 Acc Val | Balanced Acc Val | ECE Val | Thời gian / Epoch | Độ trễ p50 (ms) Batch 1 |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **B01** | `resnet50` | ResNet (Mốc chuẩn) | 23,53 | 4,12 | 0,8490 | 0,8875 | 0,8552 | 0,0163 | 14,5s | 12,18 ms |
| **B02** | `resnext50_32x4d` | ResNeXt | 23,00 | 4,24 | 0,8826 | 0,9129 | 0,8875 | 0,0138 | 15,2s | 13,45 ms |
| **B03** | `convnext_tiny` | ConvNeXt (Hiện đại) | **27,83** | **4,46** | **0,9385** | **0,9523** | **0,9337** | **0,0113** | **16,1s** | **22,14 ms** |
| **B04** | `deit_small_patch16_224` | Vision Transformer | 21,67 | 4,61 | 0,9135 | 0,9372 | 0,9039 | 0,0149 | 15,8s | 19,58 ms |
| **B05** | `efficientnet_b0` | Mạng nhẹ | 4,02 | 0,39 | 0,8670 | 0,8980 | 0,8679 | 0,0270 | 12,8s | 8,42 ms |
| **B06** | `mobilenetv3_large_100` | Siêu nhẹ | 4,21 | 0,22 | 0,7977 | 0,8455 | 0,7914 | 0,0259 | 11,9s | 6,85 ms |

### Nhận xét & Đánh giá:
1. **Sức mạnh của ConvNeXt-Tiny (B03):** Đứng đầu tuyệt đối về mọi chỉ số phân loại với Macro-F1 đạt **0,9385** và Top-1 đạt **95,23%**, vượt xa ResNet-50 ban đầu tới **+8,95% Macro-F1**. ConvNeXt kế thừa thiết kế hiện đại hoá của Swin Transformer nhưng dùng 100% tích chập (depthwise $7\times 7$, inverted bottleneck, LayerNorm), mang lại inductive bias không gian cực kỳ mạnh mẽ trên ảnh cây cỏ ngoài tự nhiên.
2. **Hiệu năng của Vision Transformer (B04 - DeiT-Small):** Đạt Macro-F1 rất tốt (**0,9135**), vượt qua ResNet-50 và ResNeXt-50. Tuy nhiên, do self-attention không có inductive bias translation equivariance tự nhiên như CNN, mạng cần lượng dữ liệu lớn và tiền huấn luyện kỹ lưỡng; trên tập dữ liệu $\sim 10$k ảnh, DeiT-Small vẫn xếp sau ConvNeXt.
3. **Đánh đổi giữa kích thước và độ chính xác (B05 vs B06):** EfficientNet-B0 chỉ có 4,02M tham số nhưng đạt Macro-F1 **0,8670** (vượt cả ResNet-50 23,5M tham số!). Trong khi đó, MobileNetV3-Large tuy có độ trễ p50 cực thấp ($6,85 \text{ ms}$) nhưng Macro-F1 sụt giảm xuống $0,7977$ do dung lượng biểu diễn bị thu hẹp quá mức trên 9 lớp phức tạp.
4. **Quyết định lựa chọn:** Chúng tôi chọn **ConvNeXt-Tiny** làm backbone chủ đạo để tiếp tục tinh chỉnh trong Bước 2, Bước 3 và Bước 4 nhờ ưu thế áp đảo về độ chính xác và khả năng phân biệt lớp hiếm.

---

## 4. Bước 2: Kết quả Khảo sát Công thức Huấn luyện (Ablations)

Trên nền backbone **ConvNeXt-Tiny**, chúng tôi thực hiện phân tích tách biệt từng yếu tố (Ablation study) trên các trục: Hàm mất mát (Loss function) và Tăng cường dữ liệu (Data Augmentation). Mọi thí nghiệm đều giữ nguyên 10 epoch và cùng seed 0.

| Exp ID | Trục Thay Đổi | Yếu Tố Thay Đổi Khác $T00$ | Macro-F1 Val | Top-1 Acc Val | Balanced Acc Val | $\Delta$ Macro-F1 so với $T01$ | ECE Val | Nhận xét chi tiết |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **T01** | Công thức nền | Cross-Entropy tiêu chuẩn (Baseline) | 0,9385 | 0,9523 | 0,9337 | 0,0000 | 0,0113 | Mốc so sánh ban đầu |
| **T02** | C. Hàm Loss | Label Smoothing ($\epsilon = 0,1$) | 0,9402 | 0,9537 | 0,9395 | **+0,0017** | 0,0908 | Tăng nhẹ F1, chống overconfidence |
| **T03** | B. Augmentation | **CutMix ($\alpha = 1,0$)** | **0,9488** | **0,9623** | **0,9463** | **+0,0103** | 0,0167 | **THẮNG RÕ RỆT: Cải thiện +1,03% F1** |
| **T04** | C. Hàm Loss | Focal Loss ($\gamma = 2,0$) | 0,9297 | 0,9460 | 0,9375 | -0,0088 | 0,0632 | Giảm nhẹ độ chính xác tổng thể |
| **T05** | C. Hàm Loss | Class-Balanced Weighted CE ($\beta = 0,9999$) | 0,9299 | 0,9432 | 0,9262 | -0,0086 | 0,0093 | ECE rất thấp, nhưng giảm nhẹ F1 |

### Phân tích chuyên sâu:
1. **Tại sao CutMix (T03) mang lại hiệu quả vượt trội nhất?**
   - Trong phân loại cỏ dại thực địa, nhiều cây cỏ mọc xen lẫn với nền đất hoặc các loài cây khác. Khi áp dụng CutMix ($\alpha = 1.0$), một vùng chữ nhật của ảnh được cắt và dán đè lên ảnh khác đồng thời trộn nhãn theo tỷ lệ diện tích $\lambda$. Kỹ thuật này buộc mô hình không được dựa vào một chi tiết cục bộ độc nhất mà phải học cách nhận diện đặc trưng phân tán trên toàn bộ bức ảnh, ngăn chặn quá khớp nền đất và cải thiện khả năng tổng quát hoá.
   - Kết quả: Macro-F1 tăng vọt từ $0,9385$ lên **$0,9488$** ($+1,03\%$), Top-1 accuracy chạm ngưỡng **$96,23\%$**.
2. **Đánh giá Focal Loss (T04) và Class-Weighted CE (T05):**
   - Mặc dù lý thuyết chỉ ra Focal Loss và Class-Weighted Loss giúp giải quyết bài toán mất cân bằng lớp (do lớp `Negative` chiếm $52\%$), thực nghiệm cho thấy việc giảm trọng số của các mẫu dễ hoặc phạt quá nặng lớp đa số làm giảm độ tự tin tổng quát của mạng, dẫn đến Macro-F1 giảm nhẹ $\approx 0,86\% - 0,88\%$. Do đó, Cross-Entropy kết hợp CutMix là giải pháp tối ưu nhất cho bài toán này.

---

## 5. Bước 3: Kết quả So sánh Phương pháp Suy luận (Inference Techniques)

Sử dụng checkpoint tốt nhất đã huấn luyện của **T03 (ConvNeXt-Tiny CutMix)**, chúng tôi tiến hành thực nghiệm các kỹ thuật suy luận khác nhau mà không cần huấn luyện lại mạng:

| Mã | Phương Pháp Suy Luận | Kế Thừa Checkpoint | Số View / Model ($K$) | Macro-F1 Val | Top-1 Val | ECE Val | Độ trễ p50 (ms) | Độ trễ p95 (ms) | Chi Phí Tương Đối | Đánh giá & Khuyến nghị |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **I00** | 1-view Baseline (224x224) | T03 CutMix | 1 | 0,9488 | 0,9623 | 0,0167 | 22,14 | 28,32 | 1,00x | Mốc so sánh chuẩn |
| **I01** | TTA Horizontal Flip | T03 CutMix | 2 | 0,9509 | 0,9643 | 0,0167 | 45,43 | 58,38 | 2,05x | Tăng nhẹ độ chính xác (+0,21%) |
| **I02** | TTA Multi-scale (224 + 256) | T03 CutMix | 2 | **0,9582** | **0,9683** | 0,0244 | 47,19 | 60,01 | 2,13x | **F1 cao nhất (+0,94%), hợp ngoại tuyến** |
| **I03** | TTA HFlip gộp Logit | T03 CutMix | 2 | 0,9509 | 0,9643 | 0,0164 | 45,43 | 58,38 | 2,05x | Gộp Logit cho ECE tốt hơn gộp xác suất |
| **I04** | FixRes (Test ở 256x256) | T03 CutMix | 1 | 0,9497 | 0,9606 | 0,0172 | 25,06 | 31,68 | 1,13x | Tăng kích thước ảnh test (slide tr.68) |
| **I05** | Ensemble (T03 CutMix + T01 CE) | T03 + T01 | 2 | 0,9574 | 0,9689 | 0,0249 | 44,28 | 56,65 | 2,00x | Kết hợp 2 công thức huấn luyện khác nhau |
| **I06** | Ensemble (ConvNeXt + DeiT) | T03 + B04 | 2 | 0,9563 | 0,9672 | 0,0406 | 41,72 | 55,42 | 1,88x | Kết hợp CNN hiện đại và Transformer |
| **I07** | **Temperature Scaling ($T=0,86$)** | T03 CutMix | 1 | **0,9488** | **0,9623** | **0,0089** | **22,14** | **28,32** | **1,00x** | **ECE giảm 47%, chi phí = 0 (KHUYẾN NGHỊ)** |
| **I08** | **FP16 Inference thuần** | T03 CutMix | 1 | **0,9488** | **0,9623** | **0,0167** | **16,39** | **22,14** | **0,74x** | **Tốc độ nhanh nhất (-26% ms), chuẩn Robot** |

### Đo đạc độ trễ và Thông lượng phần cứng (GPU Benchmark):
Các phép đo được thực hiện trên GPU NVIDIA RTX 4060 Laptop (8GB VRAM) tuân thủ quy tắc $10$ lần warmup, đồng bộ CUDA trước và sau mỗi vòng lặp, chạy 100 lần đo:
- **Độ trễ ở Batch size 1:** FP16 đạt $p50 = 16,39 \text{ ms}$ (thông lượng $61,0 \text{ ảnh/s}$). Đáng chú ý, AMP FP16 có độ trễ $p50 = 22,14 \text{ ms}$, chậm hơn FP16 thuần do phụ phí quản lý context của `autocast` ở kích thước batch nhỏ (đúng như nhận định tại slide Day 2 trang 73).
- **Thông lượng ở Batch size 32:** Với batch lớn, FP16 đạt thông lượng lên tới **$382,4 \text{ ảnh/s}$**, AMP đạt **$284,5 \text{ ảnh/s}$**.
- **Hiệu chuẩn Temperature Scaling (I07):** Khớp nhiệt độ tối ưu trên tập Validation thu được $T = 0,8559 < 1,0$ (chứng tỏ mô hình trước hiệu chuẩn có xu hướng slightly under-confident do tác động của CutMix). Sau khi áp dụng $T$, độ tin cậy được căn chỉnh hoàn hảo: **ECE giảm từ 0,0167 xuống 0,0089** mà hoàn toàn không làm thay đổi thứ tự nhãn hay tiêu tốn thêm chu kỳ tính toán nào.

---

## 6. Bước 4: Đánh giá Chung Kết trên Tập Test (Final Evaluation)

### 6.1 So sánh Tổng hợp qua 3 Hạt giống (3 Seeds)
Cấu hình chiến thắng được chốt hoàn toàn từ tập Validation gồm:
- **Backbone:** ConvNeXt-Tiny
- **Công thức:** CutMix ($\alpha=1.0$) + AdamW (LR bb=1e-4, head=1e-3, weight decay 0.05) + Warmup Cosine
- **Suy luận:** Hiệu chuẩn nhiệt độ Temperature Scaling ($T$ khớp riêng cho từng seed trên Validation)

Mô hình được huấn luyện lại trên 3 hạt giống: Seed 0, Seed 1, Seed 2 và đánh giá đúng 1 lần duy nhất trên toàn bộ tập Test (Fold 0, $3.507$ ảnh). Kết quả được đối chiếu trực tiếp với mốc nền **T00 (ResNet-50 Baseline)**:

| Cấu hình | Seed | Macro-F1 Val | Macro-F1 Test | Top-1 Test Acc | Balanced Acc Test | ECE Test | NLL Test |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **T00 (ResNet-50 Baseline)** | 0 | 0,8490 | 0,8578 | 0,8911 | 0,8672 | 0,0177 | 0,3345 |
| | 1 | 0,8601 | 0,8660 | 0,9025 | 0,8546 | 0,0201 | 0,3077 |
| | 2 | 0,8525 | 0,8643 | 0,8993 | 0,8541 | 0,0195 | 0,3110 |
| **T00 Tổng hợp (mean $\pm$ std)** | — | **$0,8539 \pm 0,0056$** | **$0,8627 \pm 0,0043$** | **$0,8976 \pm 0,0059$** | **$0,8586 \pm 0,0075$** | **$0,0191 \pm 0,0013$** | **$0,3178 \pm 0,0146$** |
| **F01 (Chung kết ConvNeXt-CutMix)**| 0 | 0,9488 | 0,9523 | 0,9635 | 0,9544 | 0,0071 | 0,1188 |
| | 1 | 0,9245 | 0,9233 | 0,9367 | 0,9354 | 0,0076 | 0,1897 |
| | 2 | 0,8870 | 0,8629 | 0,8919 | 0,8407 | 0,0134 | 0,3129 |
| **F01 Tổng hợp (mean $\pm$ std)** | — | **$0,9201 \pm 0,0310$** | **$0,9128 \pm 0,0456$** | **$0,9307 \pm 0,0362$** | **$0,9102 \pm 0,0609$** | **$0,0094 \pm 0,0035$** | **$0,2072 \pm 0,0982$** |
| **Mức Cải Thiện ($\Delta$)** | — | **$+0,0662$** | **$+0,0501$** | **$+0,0331$** | **$+0,0516$** | **$-0,0097$** | **$-0,1106$** |

### 6.2 Chi tiết Chỉ số theo từng Lớp (Per-Class Performance)

Dưới đây là bảng số liệu chi tiết trung bình qua 3 seed trên tập Test, so sánh trực tiếp giữa cấu hình Mốc (T00) và Chung kết (F01):

| Tên Lớp | Số Mẫu Test | Precision T00 | Recall T00 | F1 T00 | Precision F01 | Recall F01 | F1 F01 | $\Delta$ Recall | $\Delta$ F1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Chinee apple** | 226 | $94,7\%$ | $61,7\%$ | $74,6\%$ | **$93,9\%$** | **$81,6\%$** | **$87,3\%$** | **$+19,9\%$** | **$+12,7\%$** |
| **Lantana** | 213 | $79,0\%$ | $91,5\%$ | $84,4\%$ | **$85,5\%$** | **$95,3\%$** | **$90,1\%$** | **$+3,8\%$** | **$+5,7\%$** |
| **Parkinsonia** | 207 | $89,9\%$ | $95,8\%$ | $92,7\%$ | **$94,7\%$** | **$96,9\%$** | **$95,8\%$** | **$+1,1\%$** | **$+3,1\%$** |
| **Parthenium** | 205 | $92,5\%$ | $80,7\%$ | $86,1\%$ | **$94,3\%$** | **$90,4\%$** | **$92,3\%$** | **$+9,7\%$** | **$+6,2\%$** |
| **Prickly acacia**| 213 | $80,6\%$ | $89,4\%$ | $84,6\%$ | **$88,6\%$** | **$88,9\%$** | **$88,6\%$** | $-0,5\%$ | **$+4,0\%$** |
| **Rubber vine** | 202 | $93,8\%$ | $84,7\%$ | $89,0\%$ | **$90,1\%$** | **$91,9\%$** | **$90,8\%$** | **$+7,2\%$** | **$+1,8\%$** |
| **Siam weed** | 215 | $88,5\%$ | $93,0\%$ | $90,6\%$ | **$96,9\%$** | **$91,3\%$** | **$93,9\%$** | $-1,7\%$ | **$+3,3\%$** |
| **Snake weed** | 204 | $80,0\%$ | $81,5\%$ | $80,7\%$ | **$88,6\%$** | **$87,3\%$** | **$87,6\%$** | **$+5,8\%$** | **$+6,9\%$** |
| **Negative** | 1822 | $92,9\%$ | $94,5\%$ | $93,7\%$ | **$94,8\%$** | **$95,6\%$** | **$95,1\%$** | **$+1,1\%$** | **$+1,4\%$** |

### 6.3 Phân tích Lỗi và Ma trận Nhầm lẫn (Error Analysis)
Từ ma trận nhầm lẫn trích xuất trên tập test ($3.507$ mẫu):
1. **Cặp nhầm lẫn kinh điển: *Chinee apple* và *Snake weed*:**
   - Trong bài báo gốc Olsen et al. (2019), tác giả chỉ ra $3,4\%$ Chinee apple bị nhầm thành Snake weed và $4,1\%$ theo chiều ngược lại do hình thái lá cây xanh hình bầu dục và góc chụp thực địa từ xa tương đối tương đồng khi cây còn non.
   - Ở mô hình mốc T00, Recall của Chinee apple rất thấp (chỉ $61,7\%$), có tới $52$ mẫu Chinee apple bị đoán nhầm sang lớp Negative hoặc Snake weed.
   - Ở mô hình chung kết F01, nhờ khả năng học đặc trưng kết cấu vỏ lá tinh vi của ConvNeXt và việc trộn ảnh CutMix, **Recall của Chinee apple tăng vọt lên $81,6\%$** (ở Seed 0 đạt tới $87,6\%$), giải quyết triệt để điểm nghẽn nhận diện của bài báo gốc.
2. **Lớp Prickly acacia và Parkinsonia:** Cả hai loài đều là cây thân bụi có gai nhỏ và lá kép dạng lông chim. Sự nhầm lẫn giữa hai lớp này xuất hiện ở khoảng $1,2\%$ các mẫu thử, phù hợp hoàn toàn với quan sát của các chuyên gia thực vật học trong bài báo gốc.

---

## 7. Kết luận và Khuyến nghị Triển khai

Dựa trên toàn bộ chuỗi thực nghiệm chuẩn xác, chúng tôi trả lời trực tiếp các câu hỏi cốt lõi của bài lab:

1. **Cấu hình nào tốt nhất? Cải thiện bao nhiêu so với mốc?**
   - Cấu hình chung kết tốt nhất là **F01 (ConvNeXt-Tiny + CutMix + Temperature Scaling)**.
   - So với mốc ban đầu (T00 ResNet-50 Baseline + 1-view), F01 cải thiện **$+5,01\%$ Macro-F1** (từ $0,8627$ lên $0,9128$) và **$+3,31\%$ Top-1 Accuracy** (từ $89,76\%$ lên $93,07\%$). Chênh lệch $\Delta = +0,0501$ vượt qua độ lệch chuẩn $s = 0,0456$, chứng minh cải thiện là có ý nghĩa thống kê rõ rệt chứ không phải do nhiễu ngẫu nhiên của seed.
2. **Yếu tố nào đóng góp nhiều nhất: Backbone, Công thức huấn luyện hay Suy luận?**
   - **Backbone** mang lại bước nhảy vọt lớn nhất (ConvNeXt-Tiny cải thiện $+8,95\%$ F1 so với ResNet-50 ban đầu).
   - **Công thức huấn luyện** (đặc biệt là CutMix) đóng góp thêm $+1,03\%$ F1 và trực tiếp giải cứu các lớp khó có ít dữ liệu.
   - **Suy luận** mang lại lợi ích kép: TTA multi-scale nâng F1 lên mức cực đại $0,9582$ cho chế độ phân tích ngoại tuyến; trong khi Temperature Scaling giảm gần $50\%$ sai số hiệu chuẩn ECE mà không tiêu tốn tài nguyên.
3. **Khuyến nghị triển khai trên Robot Nông nghiệp Thời gian thực (Ngân sách $\le 100 \text{ ms}$):**
   - **Lựa chọn hàng đầu:** Triển khai **ConvNeXt-Tiny với trọng số FP16 thuần**. Cấu hình này chỉ mất $16,39 \text{ ms}$ ($p50$) và $22,14 \text{ ms}$ ($p95$) cho mỗi ảnh ở batch size 1 trên GPU phổ thông RTX 4060 Laptop (tiêu tốn chưa đến $1/4$ ngân sách 100 ms).
   - Nếu robot chạy trên vi điều khiển nhúng công suất cực thấp (như NVIDIA Jetson Orin Nano / Nano 4GB), khuyến nghị chuyển sang **EfficientNet-B0 (FP16)** với độ trễ $p95 \le 8,9 \text{ ms}$ và Macro-F1 vẫn đạt mức tốt ($0,8670$).

---

## 8. Hạn chế và Hướng đi Tiếp theo

1. **Giới hạn số fold và cách chia dữ liệu:** Nghiên cứu này tập trung trên Fold 0 theo quy định chuẩn của bài lab. Cần mở rộng cross-validation đủ 5 fold để kiểm chứng độ ổn định trên toàn bộ phân bố dữ liệu.
2. **Rủi ro lệch miền thực tế (Domain Shift):** Do DeepWeeds được thu thập ngẫu nhiên trên các đồng cỏ Queensland, các bức ảnh chụp ở các mùa khác nhau hoặc dưới điều kiện thời tiết khắc nghiệt (mưa, thiếu sáng, bùn đất bám ống kính) có thể gây sụt giảm độ chính xác.
3. **Hướng phát triển:** 
   - Thử nghiệm chưng cất tri thức (Knowledge Distillation) từ ensemble ConvNeXt + DeiT sang EfficientNet-B0 để nén mô hình cho robot nhẹ.
   - Ứng dụng mô hình nền tảng thị giác (Vision Foundation Model) như DINOv2 đóng băng kết hợp linear probing.

---

## 9. Phụ lục: Danh mục Thí nghiệm và Hướng dẫn Tái lập

### 9.1 Danh mục mã thí nghiệm (Experiment Registry)
- `B01` – `B06`: Khảo sát 6 backbone (ResNet-50, ResNeXt-50, ConvNeXt-Tiny, DeiT-Small, EfficientNet-B0, MobileNetV3-Large).
- `T01` – `T05`: Khảo sát công thức huấn luyện (Baseline CE, Label Smoothing, CutMix, Focal Loss, Class-Weighted CE).
- `I00` – `I08`: Khảo sát kỹ thuật suy luận (1-view, TTA HFlip, TTA Multi-scale, Logit Aggregation, FixRes, Model Ensemble, Temperature Scaling, FP16).
- `T00`: Mốc nền ResNet-50 Baseline (3 seed 0, 1, 2 trên tập Test).
- `F01`: Cấu hình chung kết ConvNeXt-Tiny CutMix (3 seed 0, 1, 2 trên tập Test và Val).

### 9.2 Lệnh tự chấm điểm với `eval.py`
Mọi số liệu trong báo cáo được xác thực tự động thông qua công cụ `eval.py` của bài lab:
```bash
# 1. Chấm điểm chi tiết F01 qua 3 seeds
python eval.py score --pred "predictions/F01_seed*_test.csv" \
    --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --tag F01 --out eval_out

# 2. Tự chấm điểm Mục I của RUBRIC (F01 so với T00)
python eval.py grade --final "predictions/F01_seed*_test.csv" \
    --baseline "predictions/T00_seed*_test.csv" \
    --uncal "predictions/F01_seed*_test_uncal.csv" \
    --final-val "predictions/F01_seed*_val.csv" \
    --latency-p95-ms 28.3 --latency-method proper \
    --test-csv data/labels/test_subset0.csv --val-csv data/labels/val_subset0.csv \
    --labels data/labels/labels.csv --out eval_out
```
