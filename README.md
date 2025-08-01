
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

⚙️ Working of the Project
The eKYC system follows a multi-stage deep learning pipeline to validate a user's identity using a government-issued ID (Aadhar or PAN) and a selfie image. Here’s a step-by-step breakdown:

1. Upload Inputs
The user uploads:

A scanned image of their Aadhar or PAN card

A recent selfie photo

2. Text Extraction using OCR
The uploaded ID image is processed using EasyOCR and PaddleOCR.

The system extracts key fields:

Name

Date of Birth

Gender

ID Number (Aadhar or PAN)

The extracted text is cleaned and validated using regex and format checking.

3. Face Detection & Matching
Faces are detected in both:

ID card photo (using Haarcascade)

Selfie image

A face recognition model (OpenCV with pretrained recognizer) computes embedding vectors for both images.

A cosine similarity score is calculated.

If the score exceeds a certain threshold (e.g., 0.90), it is considered a match.

4. Decision Logic
The system uses the following conditions to decide KYC status:

All required fields successfully extracted

Face match score above threshold

Based on the checks:

KYC Passed: If both text extraction and face match are successful

KYC Failed: If either component fails

5. Result Generation & Storage
A structured JSON response is generated including:

Extracted data

Face match score

KYC status

This data is securely stored in MongoDB for future reference or auditing.
