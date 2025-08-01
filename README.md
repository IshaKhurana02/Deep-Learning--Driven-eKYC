
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

## ⚙️ Working of the Project

The eKYC system follows a multi-stage deep learning pipeline to validate a user's identity using a government-issued ID (Aadhar or PAN) and a selfie image. Below is a step-by-step breakdown of how the system operates:

### 🔹 1. Upload Inputs
- The user uploads:
  - A **scanned image** of their Aadhar or PAN card
  - A **recent selfie** photo

### 🔹 2. Text Extraction using OCR
- The uploaded ID image is processed using **EasyOCR** and **PaddleOCR**
- The system extracts the following key fields:
  - **Name**
  - **Date of Birth**
  - **Gender**
  - **ID Number** (Aadhar or PAN)
- Extracted text is cleaned and validated using **regex** and format rules

### 🔹 3. Face Detection & Matching
- Faces are detected from:
  - The **ID card** using Haarcascade classifier
  - The **selfie** using Haarcascade/OpenCV
- Embedding vectors are computed using **OpenCV’s pretrained face recognizer**
- A **cosine similarity score** is calculated between the two face vectors
- If the score exceeds a predefined threshold (e.g., **0.90**), the faces are considered a match

### 🔹 4. Decision Logic
- The KYC status is determined based on:
  - Successful extraction of all required fields
  - Face match score being above threshold
- Final outcome:
  - ✅ **KYC Passed**: if both OCR and face match succeed
  - ❌ **KYC Failed**: if either OCR or face match fails

### 🔹 5. Result Generation & Storage
- A structured **JSON response** is generated containing:
  - Extracted user data
  - Face match score
  - Final KYC status
- This data is securely stored in **MongoDB** for:
  - Future reference
  - Verification logs
  - Auditing and compliance
