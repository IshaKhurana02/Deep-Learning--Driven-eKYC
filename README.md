
# 🔐 Deep Learning-Driven eKYC System

A web application that automates electronic Know Your Customer (eKYC) processes using deep learning. It verifies a person's identity by extracting information from scanned ID documents (Aadhar and PAN) and matching it with a selfie image.

---

## 🚀 Features

- ✅ Identity verification for **Aadhar** and **PAN** cards  
- 📷 Face detection and **face matching** using OpenCV (Haarcascade + pretrained models)  
- 📝 OCR using **EasyOCR** and **PaddleOCR** for high-accuracy text extraction  
- 🧠 Deep Learning-based preprocessing and recognition  
- 🗂️ Stores data securely in **MongoDB**  
- 🌐 Accessible through a **user-friendly web interface**  
- 🔁 Outputs **JSON response** with extracted data and KYC status (`KYC Passed` / `KYC Failed`)

---


---

## 🔧 Tech Stack

| Component       | Tool/Library                     |
|----------------|----------------------------------|
| **OCR**         | EasyOCR, PaddleOCR              |
| **Face Matching** | OpenCV (Haarcascade + pretrained) |
| **Backend**     | Flask                           |
| **Frontend**    | Streamlit                       |
| **Database**    | MongoDB                         |
| **Deployment**  | Localhost                       |

---

## 📥 Input & 📤 Output

### 🔹 Input:
- Scanned image of Aadhar or PAN card
- Selfie image of the user

### 🔹 Output:
- JSON with:
  - Extracted Name, DOB, ID Number
  - KYC status: `KYC Passed` / `KYC Failed`

---


