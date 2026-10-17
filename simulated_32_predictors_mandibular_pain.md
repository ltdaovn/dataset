# Tập Dữ Liệu Mô Phỏng Dự Báo Nguy Cơ Đau Nặng Sau Phẫu Thuật Nhổ Răng Khôn Hàm Dưới

## 1. Giới thiệu Bài toán (Context)
Phẫu thuật nhổ răng cối thứ ba hàm dưới mọc lệch (răng khôn) là một trong những thủ thuật nha khoa phổ biến nhất. Tuy nhiên, tình trạng đau nặng cấp tính sau phẫu thuật là một thách thức lớn, ảnh hưởng tiêu cực đến chất lượng cuộc sống và tâm lý của bệnh nhân.

Tập dữ liệu này được **mô phỏng dựa trên nghiên cứu y khoa thực tế** (*Zhuang et al., Heliyon 2023*), nhằm mục đích xây dựng các mô hình Khai khoáng Dữ liệu / Học máy (Data Mining & Machine Learning) để dự báo sớm những bệnh nhân có nguy cơ đau nặng (Thang điểm đau $\text{NRS} \ge 7$ trong 24 giờ đầu sau phẫu thuật).

---

## 2. Thông tin Tổng quan về Tập Dữ liệu (Dataset Overview)
* **Số lượng bản ghi (Samples):** 1,000 dòng (tương ứng 1,000 bệnh nhân/ca phẫu thuật).
* **Số lượng đặc trưng (Features):** 32 biến thuộc 7 nhóm yếu tố lâm sàng & tâm lý.
* **Biến mục tiêu (Target variable):** `High_Pain_Risk` (Biến nhị phân 0/1).
* **Tỷ lệ phân bố nhãn:**
  * `0` (Đau nhẹ/trung bình, $\text{NRS} \le 6$): ~81.2%
  * `1` (Nguy cơ đau nặng, $\text{NRS} \ge 7$): ~18.8% (Tập dữ liệu có sự **mất cân bằng nhãn - Imbalanced Data**).

---

## 3. Danh sách Chi tiết 32 Biến Đầu Vào (Attributes / Features)

| STT | Tên Biến (`Feature Name`) | Kiểu Dữ Liệu | Diễn Giải & Giá Trị Khả Thi |
| :---: | :--- | :--- | :--- |
| **I** | **Nhóm Nhân khẩu học (Demographics)** | | |
| 1 | `Age` | Liên tục | Tuổi của bệnh nhân (Năm) |
| 2 | `BMI` | Liên tục | Chỉ số khối cơ thể ($kg/m^2$) |
| 3 | `Sex` | Định danh | Giới tính: `Male`, `Female` |
| **II** | **Nhóm Vệ sinh Răng miệng (Oral Hygiene)** | | |
| 4 | `TB_Freq` | Phân loại | Tần suất đánh răng: `Once_or_less` ($\le 1$ lần/ngày), `Twice_or_more` ($\ge 2$ lần/ngày) |
| 5 | `TB_Time` | Phân loại | Thời gian đánh răng: `Less_2min`, `2-3min`, `More_3min` |
| 6 | `OHI_DI` | Nhị phân | Chỉ số mảng bám OHI-S DI: `0` (Sạch), `1` (Có mảng bám) |
| 7 | `OHI_S_CI` | Nhị phân | Chỉ số cao răng OHI-S CI: `0` (Không cao răng), `1` (Có cao răng) |
| **III** | **Nhóm Dấu hiệu Sinh tồn (Vital Signs)** | | |
| 8 | `Systolic_BP` | Liên tục | Huyết áp tâm thu (mmHg) |
| 9 | `Diastolic_BP` | Liên tục | Huyết áp tâm trương (mmHg) |
| 10 | `Heart_Rate` | Liên tục | Nhịp tim (nhịp/phút) |
| **IV** | **Nhóm Hình ảnh X-quang (Radiographic Factors)** | | |
| 11 | `Winter_Class` | Phân loại | Phân loại thế răng lệch (Winter): `Mesioangular` (Lệch gần), `Vertical` (Thẳng), `Horizontal` (Nằm ngang), `Distoangular` (Lệch xa) |
| 12 | `Pell_Gregory_Vert` | Phân loại | Độ sâu mọc răng (Pell & Gregory): `A`, `B`, `C` |
| 13 | `Pell_Gregory_Horiz` | Phân loại | Khoảng cách đến cành cao hàm dưới: `I`, `II`, `III` |
| 14 | `Root_Number` | Phân loại | Số lượng chân răng: `Single` (1 chân), `Multi` (Nhiều chân) |
| **V** | **Nhóm Yếu tố Tâm lý & Đau (Psychological Factors)** | | |
| 15 | `Patient_Pain_Sens` | Thứ tự | Độ nhạy cảm đau tự báo cáo của bệnh nhân: `Low`, `Middle`, `High` |
| 16 | `Patient_Stress` | Thứ tự | Mức độ căng thẳng của bệnh nhân: `Low`, `Middle`, `High` |
| 17 | `Patient_Fear` | Thứ tự | Mức độ sợ hãi của bệnh nhân: `Low`, `Middle`, `High` |
| 18 | `Preop_NRS` | Thứ tự | Điểm đau trước phẫu thuật: `0`, `1-3`, `4-10` |
| 19 | `Surgeon_Pain_Sens` | Thứ tự | Đánh giá của bác sĩ về độ nhạy cảm đau của bệnh nhân: `Low`, `Middle`, `High` |
| 20 | `Surgeon_Stress` | Thứ tự | Căng thẳng của phẫu thuật viên: `Low`, `Middle`, `High` |
| 21 | `Surgeon_Fear` | Thứ tự | Sợ hãi ca phẫu thuật của phẫu thuật viên: `Low`, `Middle`, `High` |
| **VI** | **Nhóm Yếu tố Phẫu thuật (Operative Factors)** | | |
| 22 | `Op_Duration` | Liên tục | Thời gian phẫu thuật (Phút) |
| 23 | `Ext_Number` | Phân loại | Số lượng răng nhổ cùng lúc: `1` hoặc `2` |
| 24 | `Surgeon_Qual` | Thứ tự | Trình độ bác sĩ: `Primary` (Sơ cấp), `Intermediate` (Trung cấp), `Senior` (Cao cấp/Chuyên gia) |
| 25 | `Flap_Design` | Phân loại | Thiết kế vạt nướu: `No_flap` (Không rạch vạt), `Envelope` (Vạt bao), `Triangular` (Vạt tam giác) |
| 26 | `Bone_Removal` | Phân loại | Có mở xương hay không: `No`, `Yes` |
| 27 | `Elevator` | Phân loại | Loại bẩy răng: `Regular` (Bẩy thường), `Minimally` (Bẩy tối thiểu/xâm lấn tối thiểu) |
| 28 | `Irrigation` | Phân loại | Bơm rửa vết thương: `No`, `Saline` (Nước muối sinh lý) |
| 29 | `Hemostasis` | Phân loại | Cầm máu bằng collagen: `No`, `Collagen` |
| **VII** | **Nhóm Thuốc can thiệp (Medication)** | | |
| 30 | `NSAID` | Phân loại | Loại thuốc giảm đau sử dụng: `No`, `Ibuprofen`, `Loxoprofen` |
| 31 | `Antibiotics` | Phân loại | Sử dụng kháng sinh: `No`, `Yes` |
| 32 | `Dexamethasone` | Phân loại | Sử dụng kháng viêm Dexamethasone: `No`, `Yes` |

---

## 4. Biến Mục tiêu (Target Variable)

* **`High_Pain_Risk`**:
  * `1`: Bệnh nhân bị đau nặng sau phẫu thuật ($\text{NRS} \ge 7$).
  * `0`: Bệnh nhân chỉ bị đau nhẹ/vừa ($\text{NRS} \le 6$).

---

## 5. Gợi ý Đề bài Đồ án / Bài tập Khai khoáng Dữ liệu

Sinh viên có thể sử dụng tập dữ liệu này để thực hiện các nhiệm vụ kỹ thuật sau:

### **Bài 1: Khám phá và Tiền xử lý Dữ liệu (EDA & Preprocessing)**
1. **Mã hóa dữ liệu (Encoding):** Tiến hành Mã hóa một-hot (One-Hot Encoding) cho các biến định danh và Mã hóa thứ tự (Ordinal Encoding) cho các biến có thứ cấp (`Surgeon_Qual`, `Patient_Pain_Sens`...).
2. **Xử lý Mất cân bằng nhãn (Imbalanced Data):** Phân tích ảnh hưởng của hiện tượng mất cân bằng nhãn ($\approx 81:19$) và thử nghiệm các kỹ thuật cân bằng dữ liệu như **SMOTE**, **Random Under-sampling**, hoặc điều chỉnh `class_weight`.

### **Bài 2: Sàng lọc Đặc trưng (Feature Selection)**
1. **Lựa chọn thuộc tính:** Ứng dụng hồi quy phạt **LASSO (L1 regularization)** hoặc **Recursive Feature Elimination (RFE)** để giảm số lượng biến từ 32 xuống các tập biến tối ưu (ví dụ: Top 5 hoặc Top 10 biến quan trọng nhất).
2. **So sánh mối tương quan:** Vẽ ma trận tương quan (Correlation Matrix) hoặc biểu đồ Importance score từ thuật toán Random Forest.

### **Bài 3: Huấn luyện & So sánh Mô hình Phân lớp (Classification)**
Huấn luyện và so sánh ít nhất 3-4 thuật toán khác nhau:
* Decision Tree (C4.5 / CART)
* Random Forest
* XGBoost / LightGBM
* Logistic Regression

### **Bài 4: Đánh giá Mô hình (Model Evaluation)**
Không sử dụng chỉ số *Accuracy* đơn thuần (do dữ liệu lệch nhãn). Sinh viên cần đánh giá mô hình dựa trên các chỉ số:
* **ROC-AUC Score** (Diện tích dưới đường cong ROC)
* **Sensitivity / Recall** (Độ nhạy - Khả năng bắt đúng bệnh nhân bị đau nặng)
* **Precision & F1-Score**

---

## 6. Tham khảo Dữ liệu Gốc
* **Bài báo nghiên cứu gốc:** *Development and validation of machine learning based prediction model for postoperative pain risk after extraction of impacted mandibular third molars*, Heliyon (2023).