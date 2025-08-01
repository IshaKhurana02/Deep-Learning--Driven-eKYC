import cv2
import os
import re
import easyocr
import logging
import streamlit as st
from sqlalchemy import text
from datetime import datetime
import helpers
import time
from pymongo import MongoClient
import gridfs
from paddleocr import PaddleOCR, draw_ocr
ocr = PaddleOCR(use_angle_cls=True, use_gpu=False,lang='en')



def extract_text(image_path,option, confidence_threshold=0.3, languages=['en']):
   
    reader = easyocr.Reader(languages)
    
    try:
        # Read the image and extract text
        if(option=="Aadhar"):
            result = ocr.ocr(image_path,cls=True)
            #st.write(result)
            extracted_text = []
            if result[0]:  # Check if results exist
                for line in result[0]:
                    text = line[1][0]  # Get the text content
                    confidence = line[1][1]  # Get the confidence score
                    if(confidence>0.6):
                        extracted_text.append((text, confidence))
            #st.write(extracted_text)
            return extracted_text
        else: result = reader.readtext(image_path)
        filtered_text = "|"  # Initialize an empty string to store filtered text
        for text in result:
            bounding_box, recognized_text, confidence = text
            if confidence > confidence_threshold:
                filtered_text += recognized_text + "|"  # Append filtered text with newline

        return filtered_text 
    except Exception as e:
        print("An error occurred during text extraction:", e)
        #logging.info(f"An error occurred during text extraction: {e}")
        return ""

def extract_aadhaar_info(ocr_results):
    """
    Extract structured information from PaddleOCR results of an Aadhaar card
    
    Args:
        ocr_results (list): List of [text, confidence] pairs from PaddleOCR
        
    Returns:
        dict: Dictionary containing structured Aadhaar information
    """
    # Initialize the result dictionary
    info = {
        'Name': None,
        'ID': None,
        'DOB': None,
        'Gender': None,
        "ID Type": "Aadhar"
    }
    
    # Store the raw text for reference
    for item in ocr_results:
        text = item[0]
        
    
    # Extract Aadhaar number - typically 12 digits, may be with or without spaces
    aadhaar_pattern = r'\d{6,12}'  # Looking for at least 6 digits together
    for item in ocr_results:
        text = item[0]
        
        # Skip lines that are likely not Aadhaar numbers
        if any(keyword in text.lower() for keyword in ["government", "female", "male", "dob", "/", "issued"]):
            continue
            
        # Find potential Aadhaar numbers
        match = re.search(aadhaar_pattern, text)
        if match and len(match.group()) >= 10:  # Likely an Aadhaar number if 10+ digits
            potential_number = match.group().replace(" ", "")
            # If it's exactly 12 digits, it's almost certainly the Aadhaar number
            if len(potential_number) == 12:
                info['ID'] = potential_number
                break
    
    # Extract name - typically comes after the header and before DOB
    # Names usually don't contain digits or special patterns
    for item in ocr_results:
        text = item[0]
        
        # Skip lines that are clearly not names
        if any(keyword in text.lower() for keyword in ["government", "india", "aadhaar", "dob", "female", "male", "issued", "proof"]):
            continue
            
        # Skip if text contains too many digits or special characters
        if sum(c.isdigit() for c in text) > 1 or sum(not c.isalnum() and not c.isspace() for c in text) > 1:
            continue
            
        # Names are typically 2-3 words
        words = text.split()
        if 1 <= len(words) <= 4 and len(text) > 3:
            info['Name'] = text
            break
    
    # Extract DOB
    dob_pattern = r'(?:DOB|D0B|DO8)[\s:/]*(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})'
    for item in ocr_results:
        text = item[0]
        match = re.search(dob_pattern, text, re.IGNORECASE)
        if match:
            info['DOB'] = match.group(1)
            break
        
        # Alternative pattern if the above doesn't match
        alt_dob_pattern = r'(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})'
        if "/DOB" in text or "/D0B" in text or "/DO8" in text:
            match = re.search(alt_dob_pattern, text)
            if match:
                info['DOB'] = match.group(1)
                break
    
    # Extract gender
    gender_patterns = [
        (r'/\s*FEMALE', 'Female'),
        (r'/\s*MALE', 'Male'),
        (r'FEMALE', 'Female'),
        (r'MALE', 'Male')
    ]
    
    for item in ocr_results:
        text = item[0]
        for pattern, gender in gender_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                info['Gender'] = gender
                break
        if info['Gender']:
            break
    
    # Extract issue date
    
    st.write(info)
    return info


def extract_information(data_string):
    # words = [word.strip() for word in data_string.replace("\n", " ").strip().split() if word.strip()]
    updated_data_string = data_string.replace(".", "")
    words = [word.strip() for word in updated_data_string.split("|") if len(word.strip()) > 2]
    
    
    extracted_info = {
    "ID": "",
    "Name": "",
    "Father's Name": "",
    "DOB": None,
    "ID Type": "PAN"
    }



    try:
        
        dob_index=words.index("Date of Birth") + 1
        extracted_info["DOB"]=words[dob_index]

        if('Name' not in data_string):
            #st.write(words)

            name_index = words.index("GOVT OF INDIA") + 1
            extracted_info["Name"] = words[name_index]

            fathers_name_index = name_index + 1
            extracted_info["Father's Name"] = words[fathers_name_index]

            id_number_index=0
            for i in words:
                if("Permanent Account" in i):
                    id_number_index=words.index(i)+1
            
            extracted_info["ID"] = words[id_number_index]

        else:
            #st.write(words)
            name_index = 6
            extracted_info["Name"] = words[name_index]

            fathers_name_index = name_index + 1
            extracted_info["Father's Name"] = words[fathers_name_index]

            id_number_index = 4
            extracted_info["ID"] = words[id_number_index]

    
    except Exception as e:
        print(f"Parsing error: {e}")
        return None

    return extracted_info

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir,"ekyc_logs.log"), level=logging.INFO, format=logging_str, filemode="a")

# Set wider page layout
def wider_page():
    max_width_str = "max-width: 1200px;"
    st.markdown(
        f"""
        <style>
            .reportview-container .main .block-container{{ {max_width_str} }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    logging.info("Page layout set to wider configuration.")


# Sidebar
def sidebar_section():
    st.sidebar.title("Select ID Card Type")
    option = st.sidebar.selectbox("", ("PAN", "Aadhar"))
    logging.info(f"ID card type selected: {option}")
    return option

# Header
def header_section(option):
    if option == "Aadhar":
        st.title("Registration Using Aadhar Card")
        logging.info("Header set for Aadhar Card registration.")
    elif option == "PAN":
        st.title("Registration Using PAN Card")
        logging.info("Header set for PAN Card registration.")


def main_content(image_file, face_image_file, option, use_webcam=True):
    if image_file is not None:
        image = helpers.read_image(image_file, is_uploaded=True)
        if image is not None:
            image_roi, name = helpers.extract_id_card(image)

            if use_webcam:
                face_image = face_image_file  # already captured
                if face_image is None:
                    st.error("No face captured. Please try again.")
                    return
            else:
                face_image = helpers.read_image(face_image_file, is_uploaded=True)
                if face_image is None:
                    st.error("Face image not uploaded.")
                    return

            face_image_path1 = helpers.save_image(face_image, "face_image.jpg", path="data\\02_intermediate_data")
            face_image_path2 = helpers.detect_and_extract_face(name)
            
            logging.info("Going for face verification")
            is_face_verified = helpers.face_comparison(
                image1_path=face_image_path1,
                image2_path=face_image_path2,
                model_name="facerecognition"
            )

            if is_face_verified:
                extracted_text = extract_text(image_roi,option)

                if(option=="PAN"):
                    text_info = extract_information(extracted_text)
                else:
                    text_info=extract_aadhaar_info(extracted_text)
                st.success("Face verification successful.")
                # st.write(extracted_text)
                # st.write(f"Parsed Info: {text_info}")

                id_type = text_info.get("ID Type", "").lower()
                collection = helpers.Aadhar_collection if id_type == "aadhar" else helpers.Pan_collection

                # Check for duplicate
                if helpers.check_duplicacy(collection, text_info['ID']):
                    st.warning("Duplicate record found. Not inserting again.")
                else:
                    helpers.insert_records(text_info, face_image, collection)
                    st.success("Record inserted into MongoDB.")
            else:
                st.error("Face verification failed. Please try again.")
        else:
            st.warning("Failed to load ID card image.")
    else:
        st.warning("Please upload an ID card.")


def capture_face_from_webcam():
    """
    Opens webcam, detects face, and captures the image when user clicks Capture.
    """
    st.write("Opening webcam. Make sure your face is clearly visible.")

    # Load the Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Unable to access the webcam.")
        logging.error("Webcam access failed.")
        return None

    FRAME_WINDOW = st.image([])  # placeholder for webcam feed

    # Setup session state for capture control
    if 'capture' not in st.session_state:
        st.session_state.capture = False

    # Camera display and capture button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        capture_button = st.button('📸 Capture Image')


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame from the webcam.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)

        FRAME_WINDOW.image(frame_rgb, channels="RGB")

        # If capture button clicked
        if capture_button:
            st.session_state.capture = True

        # After button clicked, capture when a face is detected
        if st.session_state.capture and len(faces) > 0:
            # (Optional) Countdown
            with st.spinner('Capturing image in 3 seconds...'):
                time.sleep(3)

                st.write('3 2 1...')
            
            captured_frame = frame
            st.success("Face Captured Successfully!")
            cap.release()
            cv2.destroyAllWindows()
            return captured_frame

        time.sleep(0.05)

    cap.release()
    cv2.destroyAllWindows()
    return None


def main():
    wider_page()
    option = sidebar_section()
    header_section(option)
    image_file = st.file_uploader("Upload ID Card")

    if image_file is not None:
        st.write("Choose how to take face image:")
        face_option = st.radio("Select method:", ("Upload Face Image", "Capture Face from Webcam"))

        if face_option == "Upload Face Image":
            face_image_file = st.file_uploader("Upload Face Image")
            if face_image_file is not None:
                main_content(image_file, face_image_file, option, use_webcam=False)

        elif face_option == "Capture Face from Webcam":
            if 'captured_face' not in st.session_state:
                st.session_state.captured_face = None
            if 'capturing' not in st.session_state:
                st.session_state.capturing = False

            # If not yet captured and not capturing, show Start Webcam
            if not st.session_state.capturing and st.session_state.captured_face is None:
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    if st.button("Start Webcam"):
                        st.session_state.capturing = True
                        st.rerun()

            # Start capturing via webcam
            if st.session_state.capturing and st.session_state.captured_face is None:
                captured_face = capture_face_from_webcam()
                if captured_face is not None:
                    st.session_state.captured_face = captured_face
                    st.session_state.capturing = False
                    st.rerun()

            # If already captured, show image and verify/retake options
            elif st.session_state.captured_face is not None:
                st.image(
                    cv2.cvtColor(st.session_state.captured_face, cv2.COLOR_BGR2RGB),
                    caption="Captured Face",
                    channels="RGB"
                )

                col1, col2, col3 = st.columns(3)

                with col1:
                    if st.button("Retake"):
                        st.session_state.captured_face = None
                        st.session_state.capturing = False
                        st.rerun

                with col3:
                    if st.button("Verify Face"):
                        main_content(image_file, st.session_state.captured_face, option, use_webcam=True)

    # with st.expander("📤 Export Data"):
    #     st.subheader("Export User Records")

    #     id_type = st.selectbox("Choose ID Type to Export", ["Aadhaar", "PAN"])

    #     if st.button("Download CSV"):
    #         collection = helpers.Aadhar_collection if id_type.lower() == "aadhaar" else helpers.Pan_collection
    #         df = helpers.export_users(collection)

    #         if not df.empty:
    #             csv = df.to_csv(index=False).encode('utf-8')
    #             st.download_button(
    #                 label="Click to Download CSV",
    #                 data=csv,
    #                 file_name=f"{id_type}_records.csv",
    #                 mime="text/csv"
    #             )
    #         else:
    #             st.warning("No records found to export.")


if __name__ == "__main__":
    main()