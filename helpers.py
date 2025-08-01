
from sqlalchemy import text,select
import streamlit as st
import pandas as pd
import face_recognition
from deepface import DeepFace
import numpy as np
import cv2
import os
import logging
import utils
from utils import file_exists, read_yaml
# -------------------------------
import pandas as pd
from datetime import datetime
import json
import re
# ------------------------------
import mysql.connector
import streamlit as st
import pandas as pd
# ------------------------------
import os
import easyocr
import logging
from pymongo import MongoClient
import gridfs


client = MongoClient("mongodb://localhost:27017/") 
db = client["ekyc_database"]
Aadhar_collection=db["aadhar"]
Pan_collection=db["pan"]
fs = gridfs.GridFS(db)

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir,"ekyc_logs.log"), level=logging.INFO, format=logging_str, filemode="a")


config_path = "config.yaml"
config = read_yaml(config_path)

artifacts = config['artifacts']
cascade_path = artifacts['HAARCASCADE_PATH']
output_path = artifacts['INTERMIDEIATE_DIR']
conour_file_name = artifacts['CONTOUR_FILE']

def insert_records(text_info,face_image,collection):
    
    _, buffer = cv2.imencode('.jpg', face_image)
    face_image_bytes = buffer.tobytes()

    image_id = fs.put(face_image_bytes, filename=f"{text_info['ID']}_face.jpg")

    if collection==Aadhar_collection:

        user_doc = {
            "id": text_info['ID'],
            "name": text_info['Name'],
            "Gender": text_info["Gender"],
            "dob": text_info['DOB'],  # Already in YYYY-MM-DD format
            "face_image_id": image_id  # reference to image in GridFS
        }
        
    elif collection==Pan_collection:
        user_doc = {
            "id": text_info['ID'],
            "name": text_info['Name'],
            "father_name": text_info["Father's Name"],
            "dob": text_info['DOB'],  # Already in YYYY-MM-DD format
            "face_image_id": image_id  # reference to image in GridFS
        }
    collection.insert_one(user_doc)

def fetch_records(collection,text_info):
    if isinstance(text_info, dict) and "ID" in text_info:
        user = collection.find_one({"id": text_info["ID"]})
        if user:
            df = pd.DataFrame([user])
            return df
    return pd.DataFrame()


def check_duplicacy(collection,text_info):
    
    # Query for documents with the specified ID
    count = collection.count_documents({"id": text_info})
    
    # Return True if at least one document is found
    return count > 0

def export_users(collection):
    cursor = collection.find()
    users = list(cursor)

    for user in users:
        user.pop('_id', None)  # Remove MongoDB's default _id field for cleaner CSV
        if 'face_image_id' in user:
            user['face_image_id'] = str(user['face_image_id'])  # Convert ObjectId to str

    df = pd.DataFrame(users)
    return df

def detect_and_extract_face(image_path):

    # Read the image
    img = cv2.imread(image_path)

    # Convert the image to grayscale (Haar cascade works better with grayscale images)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load the Haar cascade classifier
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

    # Find the face with the largest area
    max_area = 0
    largest_face = None
    for (x, y, w, h) in faces:
        area = w * h
        if area > max_area:
            max_area = area
            largest_face = (x, y, w, h)

    # Extract the largest face
    if largest_face is not None:
        (x, y, w, h) = largest_face
        # extracted_face = img[y:y+h, x:x+w]
        
        # Increase dimensions by 15%
        new_w = int(w * 1.50)
        new_h = int(h * 1.50)
        
        # Calculate new (x, y) coordinates to keep the center of the face the same
        new_x = max(0, x - int((new_w - w) / 2))
        new_y = max(0, y - int((new_h - h) / 2))

        # Extract the enlarged face
        extracted_face = img[new_y:new_y+new_h, new_x:new_x+new_w]

        # Convert the extracted face to RGB
        # extracted_face_rgb = cv2.cvtColor(extracted_face, cv2.COLOR_BGR2RGB)
 
        current_wd = os.getcwd()
        filename = os.path.join(current_wd, output_path, "extracted_face.jpg")

        if os.path.exists(filename):
            # Remove the existing file
            os.remove(filename)

        cv2.imwrite(filename, extracted_face)
        print(f"Extracted face saved at: {filename}")
        logging.info(f"Image saved successfully: {filename}")
        return filename

        # return extracted_face_rgb
    else:
        return None


def face_recog_face_comparison(image1_path="data\\02_intermediate_data\\extracted_face.jpg", image2_path = "data\\02_intermediate_data\\face_image.jpg"):

    img1_exists = file_exists(image1_path)
    img2_exists = file_exists(image2_path)

    if not(img1_exists and img2_exists):
        print("Check the path for the images provided")
        return False

    image1 = face_recognition.load_image_file(image1_path)
    image2 = face_recognition.load_image_file(image2_path)

    if image1 is not None and image2 is not None:
        face_encodings1 = face_recognition.face_encodings(image1)
        face_encodings2 = face_recognition.face_encodings(image2)

    else:
        print("Image is not loaded properly")
        return False

    # print(face_encodings1)

    # Check if faces are detected in both images
    if len(face_encodings1) == 0 or len(face_encodings2) == 0:
        print("No faces detected in one or both images.")
        return False
    else:
    # Proceed with comparing faces if faces are detected
        matches = face_recognition.compare_faces([face_encodings1[0]], face_encodings2[0],tolerance=0.55)[0]
    # Print the results
    if matches:
        print("Faces are verified")
        return True
    else:
        print("The faces are not similar.")
        return False
    
def deepface_face_comparison(image1_path="data\\02_intermediate_data\\extracted_face.jpg", image2_path = "data\\02_intermediate_data\\face_image.jpg",threshold=0.55):
    
    # data\01_raw_data\bibek_face.jpg
    img1_exists = file_exists(image1_path)
    img2_exists = file_exists(image2_path)

    if not(img1_exists and img2_exists):
        print("Check the path for the images provided")
        return False
    
    verfication = DeepFace.verify(img1_path=image1_path, img2_path=image2_path)
    print(verfication)


    if verfication['distance']<=threshold :
        print("Faces are verified")
        return True
    else:
        print("The faces are not similar.")
        return False

def face_comparison(image1_path, image2_path, model_name = 'deepface'):

    is_verified = False
    if model_name == 'deepface':
        is_verified = deepface_face_comparison(image1_path, image2_path)
    elif model_name ==  'facerecognition':
        is_verified = face_recog_face_comparison(image1_path, image2_path)
    else:
        print("Mention proper model name for face recognition")

    return is_verified

def get_face_embeddings(image_path):

    img_exists = file_exists(image_path)

    if not(img_exists):
        print("Check the path for the images provided")
        return None
    
    embedding_objs = DeepFace.represent(img_path = image_path, model_name = "Facenet")
    embedding = embedding_objs[0]["embedding"]

    if len(embedding) > 0:
        return embedding
    return None


def filter_lines(lines):
    # Convert string input to list if needed
    if isinstance(lines, str):
        lines = [line.strip() for line in lines.split("|") if line.strip()]

    start_index = None
    end_index = None
    filtered_lines = []

    # PAN Card Detection
    for i, line in enumerate(lines):
        if "INCOME TAX DEPARTMENT" in line.upper():
            start_index = i
        if "Signature" in line:
            end_index = i
            break

    if start_index is not None and end_index is not None:
        for line in lines[start_index:end_index + 1]:
            if len(line.strip()) > 2:
                filtered_lines.append(line.strip())
        return filtered_lines

    # Aadhaar Card Fallback: Return all meaningful lines
    for line in lines:
        if len(line.strip()) > 2 and not line.strip().isdigit():
            filtered_lines.append(line.strip())
    
    return filtered_lines


def create_dataframe(texts):
    lines = filter_lines(texts)
    print("="*20)
    print(lines)
    print("="*20)

    data = []
    id_type = ""
    name = ""
    father_name = ""
    dob = ""
    gender = ""
    pan = ""
    aadhaar = ""

    for i, line in enumerate(lines):
        line_lower = line.lower()

        # Detect PAN Card
        if "permanent account number" in line_lower or re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]$', line.strip()):
            id_type = "PAN"
            # Extract PAN number
            for j in range(len(lines)):
                if re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]$', lines[j].strip()):
                    pan = lines[j].strip()
                    break
            # Attempt to extract details assuming typical PAN layout
            if len(lines) > 4:
                name = lines[2].strip()
                father_name = lines[3].strip()
                dob = lines[4].strip()
            break

        # Detect Aadhaar
        if re.search(r'\d{4}\s\d{4}\s\d{4}', line):
            id_type = "Aadhaar"
            aadhaar = re.findall(r'\d{4}\s\d{4}\s\d{4}', line)[0].replace(" ", "")
        
        # Extract gender
        if not gender:
            if "male" in line_lower:
                gender = "Male"
            elif "female" in line_lower:
                gender = "Female"
            elif "transgender" in line_lower:
                gender = "Transgender"

        # Extract DOB or Year of Birth
        if not dob:
            dob_match = re.findall(r'\d{2}/\d{2}/\d{4}', line)
            if dob_match:
                dob = dob_match[0]
            elif "year of birth" in line_lower or "yob" in line_lower:
                yob_match = re.findall(r'\d{4}', line)
                if yob_match:
                    dob = yob_match[0]

        # Try to guess name if not already extracted
        if not name and re.match(r'^[A-Z][a-zA-Z\s]{2,}$', line.strip()) and not any(x in line_lower for x in ["government", "india", "dob", "birth", "male", "female"]):
            name = line.strip()

    if id_type == "PAN":
        data.append({
            "ID": pan,
            "Name": name,
            "Father's Name": father_name,
            "DOB": dob,
            "ID Type": id_type
        })
    elif id_type == "Aadhaar":
        data.append({
            "ID": aadhaar,
            "Name": name,
            "Gender": gender,
            "DOB": dob,
            "ID Type": id_type
        })
    else:
        print("Could not identify ID type or extract sufficient information.")

    df = pd.DataFrame(data)
    return df



 
def read_image(image_path, is_uploaded=False):
    if is_uploaded:
        try:
            # Read image using OpenCV
            image_bytes = image_path.read()
            img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                logging.info("Failed to read image: {}".format(image_path))
                raise Exception("Failed to read image: {}".format(image_path))
            return img
        except Exception as e:
            logging.info(f"Error reading image: {e}")
            print("Error reading image:", e)
            return None
    else:
        try:
            img = cv2.imread(image_path)
            if img is None:
                logging.info("Failed to read image: {}".format(image_path))
                raise Exception("Failed to read image: {}".format(image_path))
            return img
        except Exception as e:
            logging.info(f"Error reading image: {e}")
            print("Error reading image:", e)
            return None

  

def extract_id_card(img):
    """
    Extracts the ID card from an image containing other backgrounds.

    Args:
        img (np.ndarray): The input image.

    Returns:
        np.ndarray: The cropped image containing the ID card, or None if no ID card is detected.
    """

    # Convert image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Noise reduction
    blur = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Select the largest contour (assuming the ID card is the largest object)
    largest_contour = None
    largest_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > largest_area:
            largest_contour = cnt
            largest_area = area

    # If no large contour is found, assume no ID card is present
    if not largest_contour.any():
        return None

    # Get bounding rectangle of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    logging.info(f"contours are found at, {(x, y, w, h)}")
    
    current_wd = os.getcwd()
    filename = os.path.join(current_wd,output_path, conour_file_name)
    contour_id = img[y:y+h, x:x+w]
    is_exists = file_exists(filename)
    if is_exists:
        # Remove the existing file
        os.remove(filename)

    cv2.imwrite(filename, contour_id)

    return contour_id, filename


def save_image(image, filename, path="."):
  """
  Saves an image to a specified path with the given filename.

  Args:
      image (np.ndarray): The image data (NumPy array).
      filename (str): The desired filename for the saved image.
      path (str, optional): The directory path to save the image. Defaults to "." (current directory).
  """

  # Construct the full path
  full_path = os.path.join(path, filename)
  is_exists = file_exists(full_path)
  if is_exists:
        # Remove the existing file
        os.remove(full_path)

  # Save the image using cv2.imwrite
  cv2.imwrite(full_path, image)

  logging.info(f"Image saved successfully: {full_path}")
  return full_path
