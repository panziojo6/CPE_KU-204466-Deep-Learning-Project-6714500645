# CPE_KU-204466 Deep Learning Project  
## การจำแนกป่าไม้โดยใช้ข้อมูล Time Series จากดาวเทียม

### รายละเอียดโครงการ
โปรเจกต์นี้เป็นการทดลองจำแนกพื้นที่ป่าไม้ (Forest) ออกจากพื้นที่อื่น (Other) โดยใช้ข้อมูลอนุกรมเวลาจากดัชนีดาวเทียมรายเดือน (SAVI และ MNDWI) เป็นข้อมูลนำเข้าให้กับโมเดล Deep Learning 3 รูปแบบ ได้แก่  
- LSTM (Baseline)  
- Bidirectional LSTM + Attention  
- Transformer Encoder

---

## 1. ปัญหา (Problem)

การจำแนกพื้นที่ป่าไม้จากข้อมูลดาวเทียมแบบอนุกรมเวลาเป็นโจทย์ที่มีความท้าทาย เนื่องจากโมเดลต้องสามารถเรียนรู้ **ลายเซ็นเชิงฤดูกาล (seasonal signatures)** ของพืชพรรณ เช่น ช่วงที่ค่าดัชนี SAVI สูงสุดในฤดูฝน หรือการเปลี่ยนแปลงของค่าดัชนีน้ำในหน้าแล้ง

นอกจากนี้ยังมีปัญหา Data Imbalance — จำนวนพิกเซลที่เป็นป่าไม้มักน้อยกว่าพื้นที่อื่น จึงต้องใช้วิธี Balanced Sampling เพื่อป้องกันโมเดลเอนเอียงไปทางคลาสที่มีจำนวนมากกว่า

---

## 2. วัตถุประสงค์ (Objectives)

1. สร้างโมเดล Deep Learning สำหรับจำแนกพิกเซลเป็น Forest หรือ Other  
2. เปรียบเทียบประสิทธิภาพของสถาปัตยกรรม 3 รูปแบบ  
   - LSTM  
   - BiLSTM + Attention  
   - Transformer Encoder  
3. วิเคราะห์ผลเชิงประสิทธิภาพ (Accuracy, Precision, Recall, F1, AUC) เพื่อหาสถาปัตยกรรมที่เหมาะสมที่สุดทั้งในเชิงเทคนิคและเชิงปฏิบัติ

---

## 3. ข้อมูลและการเตรียมข้อมูล (Data Description)

| รายการ | รายละเอียด |
|:--|:--|
| Label (y) | จากแผนที่ LULC โดยกำหนดว่า LULC > 6000 → Forest (1), อื่น ๆ → Other (0) |
| Features (X) | ข้อมูลดัชนีดาวเทียมรายเดือน 2 ตัว: SAVI, MNDWI |
| รูปแบบข้อมูล | (จำนวนพิกเซล, จำนวนเดือน, 2) เช่น (N, 12, 2) |
| สมดุลข้อมูล | ใช้ Balanced Sampling ระหว่างคลาสในชุด train |
| ช่วงเวลา | 12 เดือน (อนุกรมเวลารายเดือน 1 ปี) |

---

## 4. วิธีการทดลอง (Methodology)

### 4.1 สถาปัตยกรรมโมเดล (Model Architectures)

#### 1. LSTM Classifier

```mermaid
flowchart LR
    A[Input Sequence\nX ∈ R^(B × T × 2)\n(SAVI, MNDWI)] --> A1[Optional: Normalize/Standardize\nper-channel]
    A1 --> B[LSTM Layer\nhidden=h, layers=L, batch_first=True]
    B --> C[Hidden/Cell States (h_t, c_t)]
    C --> C1[Select Last Hidden State\nh_T ∈ R^(B × h)]
    C1 --> D[Dropout (p)]
    D --> E[Dense (Linear)\nR^(h → 1)]
    E --> F[Sigmoid]
    F --> G[Output: p(Forest)\nThreshold → Forest/Other]
