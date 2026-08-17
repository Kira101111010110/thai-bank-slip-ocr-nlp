# 🧾 Automated Thai Bank Slip OCR & NLP Parser
> ระบบอ่านและสกัดข้อมูลจากสลิปโอนเงินธนาคารไทยอัตโนมัติด้วย Computer Vision (OpenCV), Tesseract OCR และการประมวลผลภาษาธรรมชาติ (NLP)

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Tesseract OCR](https://img.shields.io/badge/Tesseract-OCR-007ACC?style=for-the-badge&logo=google&logoColor=white)

---

## 📌 ภาพรวมโครงการ (Project Overview)
ระบบประมวลผลภาพสลิปการโอนเงินของธนาคารไทย เพื่อดึงข้อมูลสำคัญ เช่น **ชื่อผู้โอน-ผู้รับ, ยอดเงิน, วันที่-เวลา, และรหัสอ้างอิงธุรกรรม** ออกมาในรูปแบบ Structured Data (JSON) โดยใช้เทคนิคการกำหนดและตัดพื้นที่สนใจ (ROI Detection) ร่วมกับการแปลงภาพเป็นข้อความ (OCR) และการสกัดจัดหมวดหมู่ข้อมูลด้วย NLP

---

## ✨ คุณสมบัติเด่น (Features)
* **ROI Extraction:** กำหนดและตัดเฉพาะขอบเขตพื้นที่สำคัญบนสลิป (`ROI.py`) เพื่อเพิ่มความแม่นยำในการอ่านข้อความ
* **Thai & English OCR:** รองรับการอ่านตัวอักษรภาษาไทยและภาษาอังกฤษผ่าน Tesseract OCR Engine (`OCRTesseract.py`)
* **Automated Data Parsing:** ประมวลผลข้อความดิบเพื่อแยก ยอดเงิน, บัญชีผู้โอน/ผู้รับ, วันที่ทำรายการ และจัดเก็บในรูปแบบ JSON (`slip_ocr_processor.py`)
* **Web Interface:** มีหน้าเว็บ UI สำหรับอัปโหลดและแสดงผลลัพธ์แบบ Interactive (`templates/`, `client/`)
* **Testing Dataset:** มีชุดข้อมูลทดสอบโครงสร้างรายจ่ายจำลอง (`fake_expenses_200.json`)

---

## 🛠️ โครงสร้างไฟล์ในโปรเจกต์ (Project Structure)

```text
├── client/                     # Web Frontend / Client Application
├── templates/                  # HTML Templates for UI Rendering
├── ROI.py                      # Region of Interest (ROI) Detection & Cropping
├── OCRTesseract.py             # Tesseract OCR Wrapper & Preprocessing
├── slip_ocr_processor.py       # Main Slip Processing & NLP Extraction Pipeline
├── roi_output.json             # Example of Detected ROI Coordinates
├── fake_expenses_200.json      # Mock Expense Dataset for Testing
├── requirements.txt            # Python Dependencies List
└── README.md                   # Project Documentation
```

🚀 ขั้นตอนการติดตั้งและใช้งาน (Getting Started)
1. ติดตั้ง Tesseract OCR Engine
Windows: ดาวน์โหลดตัวติดตั้ง Tesseract OCR (พร้อมติดตั้งชุดภาษา tha และ eng)

Linux (Ubuntu/Debian):

Bash
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-tha
2. ติดตั้ง Python Dependencies
Bash
git clone [https://github.com/Kira101111010110/thai-bank-slip-ocr-nlp.git](https://github.com/Kira101111010110/thai-bank-slip-ocr-nlp.git)
cd thai-bank-slip-ocr-nlp
pip install -r requirements.txt
3. รันการประมวลผลสลิป
รันสกัดข้อมูลสลิป:

Bash
python slip_ocr_processor.py
ทดสอบตรวจจับ ROI:

Bash
python ROI.py
🔄 กระบวนการทำงานของระบบ (Pipeline Flow)
Input Image: รับภาพสลิปโอนเงิน (JPG / PNG)

Preprocessing & ROI: ปรับคุณภาพภาพ (Grayscale, Thresholding) และตัดเฉพาะโซนข้อความสำคัญ

OCR Engine: อ่านข้อความภาษาไทยและตัวเลขด้วย Tesseract OCR

Regex & NLP Parser: กรองและจัดระเบียบข้อมูลเป็น Key-Value Pair

Output: ส่งออกข้อมูล JSON (Transaction ID, Amount, Sender, Receiver, Timestamp)

Developed as part of Natural Language Processing & Computer Vision Projects.
