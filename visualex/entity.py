from flask import session
from . import mysql
from werkzeug.security import check_password_hash
from datetime import datetime
import pytz
import pyttsx3
import base64
from PIL import Image
import io
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from collections import defaultdict
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch

class UserAccount:
    def __init__(self, username=None, password=None, name=None, surname=None, email=None, date_of_birth=None, address=None, membership_tier="basic"):
        self.username = username
        self.password = password
        self.name = name
        self.surname = surname
        self.email = email
        self.date_of_birth = date_of_birth
        self.address = address
        self.membership_tier = membership_tier

    def login(self, username, password):

        session['username'] = username # store the username in the session

        cur = mysql.connection.cursor()

        query = "SELECT password FROM useraccount WHERE username = %s"
        data = (username,)
        cur.execute(query, data)
        account = cur.fetchone()
        check = check_password_hash(account[0], password)

        return check
    
    
    def changePW(self, username, password):
        try:
            cur = mysql.connection.cursor()
            query = "UPDATE useraccount SET password = %s WHERE username = %s"
            data = (password, username)
            cur.execute(query, data)
            mysql.connection.commit()
           
            cur.close()
            return True
        except Exception as e:
            print(f"Error changing Password: {e}")
            return False


    def createUserAcc(self, userAcc):
        try:
           cur = mysql.connection.cursor()

           query = "INSERT INTO useraccount (username, password, name, surname, email, date_of_birth, address, membership_tier) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)" 
           data = (userAcc.username, userAcc.password, userAcc.name, userAcc.surname, userAcc.email, userAcc.date_of_birth, userAcc.address, "basic")
           cur.execute(query, data)
           
           mysql.connection.commit()
           
           cur.close()
           return True
        except Exception as e:
            print(f"Error creating account: {e}")
            return False
        
    def assignMembership(self, username, membership):
        try:
            cur = mysql.connection.cursor()
            query = "UPDATE useraccount SET membership_tier = %s WHERE username = %s"
            data = (membership, username)
            cur.execute(query, data)
            mysql.connection.commit()
           
            cur.close()
            return True
        except Exception as e:
            print(f"Error changing Membership_tier: {e}")
            return False
        
    def checkMembershipExist(self, username):
        try:
            cur = mysql.connection.cursor()

            query = "SELECT membership_tier FROM useraccount WHERE username = %s"
            data = (username,)
            cur.execute(query, data)
            membership_data = cur.fetchone()

            cur.close()

            if membership_data[0] == 'premium':
                return 1
            else:
                return 0
        except Exception as e:
            print(f"Error Checking Membership: {e}")
            return None
    
    def getAllmembership(self):
        try:
            cur = mysql.connection.cursor()
            query = "SELECT username, membership_tier FROM useraccount"
            cur.execute(query)
            users = cur.fetchall()
            cur.close()
            return users  
        except Exception as e:
            print(f"Error retrieving membership info: {e}")
            return None
        
    
    def get_membership_tier_info(self, username):
        try:
            cur = mysql.connection.cursor()
            query = "SELECT membership_tier FROM useraccount WHERE username = %s"
            cur.execute(query, (username,))
            membership_tier = cur.fetchone()[0]
            cur.close()

            if membership_tier == 'basic':
                return {'membership_tier': 'basic', 'monthly_fee': 'Free', 'description': [
                    '1. Generate Descriptive Text', 
                    '2. Text to Speech feature to read text as audio message']}
            elif membership_tier == 'premium':
                return {'membership_tier': 'Premium', 'monthly_fee': '$20.00', 'description': [
                    '1. Generate Descriptive Text',
                    '2. Text to Speech feature to read text as audio message',
                    '3. Selective analysis feature to circle image',
                    '4. Generate descriptive text on the selected part of the image',
                    '5. Generate a story for the image instead of just descriptive text.']}
            else:
                return None
        except Exception as e:
            print(f"Error Retrieving Membership Tier Info: {e}")
            return None
        
    def get_user_info(self, username):
        cur = mysql.connection.cursor()
        query = "SELECT * FROM useraccount WHERE username = %s"
        cur.execute(query, (username,))
        user_data = cur.fetchone()
        mysql.connection.commit()
        cur.close()
        return user_data

    def get_user_info2(self, username):  # to edit membership_tier for admin
        session['selected_user'] = username # store the username in the session
        cur = mysql.connection.cursor()
        query = "SELECT username, name, surname, email, address, membership_tier FROM useraccount WHERE username = %s"
        cur.execute(query, (username,))
        user_data = cur.fetchone()
        mysql.connection.commit()
        cur.close()
        return user_data
    
    def get_all_users(self):
        try:
            cur = mysql.connection.cursor()

            query = "SELECT username FROM useraccount"
            cur.execute(query)
            users_list = []
            for user_name in cur.fetchall():
                username = user_name[0]
                users_list.append(username)

            cur.close()
            return users_list
        except Exception as e:
            print(f"Error getting username list: {e}")

    def search_user(self, username):
        try:
            cur = mysql.connection.cursor()

            query = "SELECT username FROM useraccount where username = %s"
            data = (username,)
            cur.execute(query,data)
           
            result =  cur.fetchone()

            cur.close()
            if result:
                return result[0]
            else:
                return None 
        except Exception as e:
            print(f"Error searching user: {e}")

    def edit_profile(self, oldUsername, newUsername, name, surname, email, dob, address, membership):
        try:
            cur = mysql.connection.cursor()
            query = "UPDATE useraccount SET username = %s, name = %s, surname = %s, email = %s, date_of_birth = %s, address = %s ,membership_tier = %s WHERE username = %s"
            data = (newUsername, name, surname, email, dob, address, membership, oldUsername)
            cur.execute(query, data)
            mysql.connection.commit()
           
            cur.close()
            return True
        except Exception as e:
            print(f"Error changing profile: {e}")
            return False
        
    def edit_profile1(self, oldUsername, newUsername, name, surname, email, dob, address):
        try:
            cur = mysql.connection.cursor()
            query = "UPDATE useraccount SET username = %s, name = %s, surname = %s, email = %s, date_of_birth = %s, address = %s WHERE username = %s"
            data = (newUsername, name, surname, email, dob, address, oldUsername)
            cur.execute(query, data)
            mysql.connection.commit()
           
            cur.close()
            return True
        except Exception as e:
            print(f"Error changing profile: {e}")
            return False
            
    def delete_account(self, username):
        try:
            cur = mysql.connection.cursor()
            delete_query = "DELETE FROM useraccount WHERE username = %s"
            delete_query1 = "DELETE FROM feedback WHERE username = %s"
            delete_query2 = "DELETE FROM transaction WHERE username = %s"
            delete_query3 = "DELETE FROM history WHERE username = %s"
            cur.execute(delete_query, (username,))
            cur.execute(delete_query1, (username,))
            cur.execute(delete_query2, (username,))
            cur.execute(delete_query3, (username,))
            mysql.connection.commit()
            cur.close()
            return True
        except Exception as e:
            print(f"Error deleting account: {e}")
            return False



class FeedbackForum:
    def __init__(self, feedback_id=None, username=None, content=None, feedback_date=None):
        self.feedback_id = feedback_id
        self.username = username
        self.content = content
        self.feedback_date = feedback_date

    def submitfeedback(self, username, content):
        try:
            cur = mysql.connection.cursor()

            query = "INSERT INTO feedback (username, f_content, feedback_date) VALUES (%s, %s, %s)"
            data = (username, content, datetime.now())
            cur.execute(query, data)

            mysql.connection.commit()

            cur.close()
            return True
        except Exception as e:
            print(f"Error saving feedback: {e}")
            return False

    def get_all_feedback(self):
        try:
            cur = mysql.connection.cursor()

            query = "SELECT * FROM feedback"
            cur.execute(query)
            feedback_list = []
            for feedback_data in cur.fetchall():
                feedback = FeedbackForum(feedback_id=feedback_data[0], username=feedback_data[1], content=feedback_data[2], feedback_date=feedback_data[3])
                feedback_list.append(feedback)

            cur.close()
            return feedback_list
        except Exception as e:
            print(f"Error getting feedback list: {e}")
    
    def time_difference(self, feedback_date):
        #Calculate the time difference between the feedback date and the current time.
        current_time = datetime.now()
        time_diff = current_time - feedback_date
        
        # Extract days, hours, and minutes
        days = time_diff.days
        hours = time_diff.seconds // 3600
        minutes = (time_diff.seconds // 60) % 60
        
        # Format the time difference
        if days == 0:
            if hours == 0:
                if minutes < 2:
                    return "just now"
                else:
                    return f"{minutes} minutes ago"
            elif hours == 1:
                return "1 hour ago"
            else:
                return f"{hours} hours ago"
        elif days == 1:
            return "1 day ago"
        elif days < 30:
            return f"{days} days ago"
        elif days < 365:
            months = days // 30
            return f"{months} months ago"
        else:
            years = days // 365
            return f"{years} years ago"

class Transactions:
    def __init__(self, transaction_id=None, username=None, payment_timestamp=None, charges=None):
        self.transaction_id = transaction_id
        self.username = username
        self.payment_timestamp = payment_timestamp
        self.charges = charges

    def make_payment(self, username, charges):
        try:

            current_timestamp = datetime.utcnow()

            # Define the timezone GMT+8
            gmt8_timezone = pytz.timezone('Asia/Singapore')

            # Localize the current timestamp to GMT+8 timezone
            payment_timestamp = pytz.utc.localize(current_timestamp).astimezone(gmt8_timezone)

            cur = mysql.connection.cursor()

            query = "INSERT INTO transaction (username, payment_timestamp, charges) VALUES (%s, %s, %s)"
            data = (username, payment_timestamp, charges)
            cur.execute(query, data)

            mysql.connection.commit()

            cur.close()
            payment_timestamp = str(payment_timestamp)
            parts = payment_timestamp.split(".")
            fisrt_part = parts[0]
            return fisrt_part
        except Exception as e:
            print(f"Error saving transaction: {e}")
            return False
        
    def get_invoice(self, payment_timestamp):
        try:

            cur = mysql.connection.cursor()

            query = "SELECT * FROM transaction WHERE payment_timestamp = %s"
            data = (payment_timestamp,)
            cur.execute(query, data)

            display = cur.fetchone()

            cur.close()
            return display
        except Exception as e:
            print(f"Error saving transaction: {e}")
            return False
        
class HistoryLogs:
    def __init__(self, history_id=None, username=None, h_date=None, result_id=None):
        self.history_id = history_id
        self.username = username
        self.h_date = h_date
        self.result_id = result_id
    
    def get_history_logs_with_predictions(self, username):
        try:
            cur = mysql.connection.cursor()

            query = """
            SELECT h.*, p.image_id, p.predicted_label, p.laplacian_score, i.image_data
            FROM history h
            JOIN prediction_results p ON h.result_id = p.result_id
            JOIN image_metadata i ON p.image_id = i.image_id
            WHERE h.username = %s
            """
            cur.execute(query, (username,))
            history_logs_with_predictions_and_images = []
            for row in cur.fetchall():
                history_log = HistoryLogs(history_id=row[0], username=row[1], h_date=row[2], result_id=row[3])
                image_blob = row[7]  # Assuming image_blob is the column storing BLOB data
                # Convert the binary image data to Base64
                image_data_base64 = base64.b64encode(image_blob).decode('utf-8')
                prediction_info = {
                    'image_id': row[4],
                    'predicted_label': row[5],
                    'laplacian_score': row[6],
                    'image_data': image_data_base64
                }
                history_logs_with_predictions_and_images.append((history_log, prediction_info))

            cur.close()
            return history_logs_with_predictions_and_images
        except Exception as e:
            print(f"Error getting history logs with predictions and images: {e}")
            return []
            
    def store_predictedResults(self, username, image_id, predicted_label, laplacian_score):
        try:
            cur = mysql.connection.cursor()

            query = "INSERT INTO prediction_results (image_id, predicted_label, laplacian_score) VALUES (%s, %s, %s)"
            data = (image_id, predicted_label, laplacian_score)
            cur.execute(query, data)
            cur.execute("SELECT LAST_INSERT_ID()")
            result_id = cur.fetchone()[0]
            print("result_id" + str(result_id))
            query1 = "INSERT INTO history (username, h_date, result_id) VALUES (%s, DATE(NOW()), %s)"
            data1 = (username, result_id)
            cur.execute(query1, data1)

            mysql.connection.commit()

            cur.close()
        
            return True
        except Exception as e:
            print(f"Error saving transaction: {e}")
            return False

class ImageData:
    def __init__(self, image_id=None, image_data=None):
        self.image_id = image_id
        self.image_data = image_data
    
    def store_image(self, image_path):
        try:
            # Connect to MySQL database
            cur = mysql.connection.cursor()

            # Read image files and insert image data into the database
            with open(image_path, 'rb') as file:
                image_data = file.read()
                # Insert image data into image_metadata table
                cur.execute("INSERT INTO image_metadata (image_data) VALUES (%s)", (image_data,))
            
            # Get the last inserted image_id
            cur.execute("SELECT LAST_INSERT_ID()")
            image_id = cur.fetchone()[0]

            # Commit changes and close connection
            mysql.connection.commit()
            cur.close()
            print("Image data inserted successfully.")
            return image_id
        except Exception as e:
            print(f"Failed to insert image data: {e}")
        
    #for object detection   
    def objectDetection(self, image_id):
        labels = open('visualex/coco.names').read().strip().split('\n')
        # Defining paths to the weights and configuration file with model of Neural Network
        weights_path = 'visualex/yolov3.weights'
        configuration_path = 'visualex/yolov3.cfg'
        # Setting minimum probability to eliminate weak predictions
        probability_minimum = 0.5
        # Setting threshold for non maximum suppression
        threshold = 0.3
        network = cv2.dnn.readNetFromDarknet(configuration_path, weights_path)
        # Getting names of all layers
        layers_names_all = network.getLayerNames()  # list of layers' names
        # Getting only output layers' names that we need from YOLO algorithm
        output_layer_indices = network.getUnconnectedOutLayers()
        layers_names_output = [layers_names_all[i - 1] for i in output_layer_indices]  # list of layers' names
        # Establish a connection to MySQL
        cur = mysql.connection.cursor()
        # Debug: Print image_id for troubleshooting
        #print(f"Fetching image data for image_id: {image_id}")
        # Retrieve image data from the database
        cur.execute("SELECT image_data FROM image_metadata WHERE image_id = %s", (image_id,))
        row = cur.fetchone()
        if row is None or not row[0]:  # Check if row is None or image_data is empty
            return "Image data not found or empty in the database"
        
        image_data = row[0]  
        # # Convert binary data to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        # Decode numpy array as an image using OpenCV
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # Getting image shape
        image_input_shape = img.shape
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        # Slicing blob and transposing to make channels come at the end
        blob_to_show = blob[0, :, :, :].transpose(1, 2, 0)
        # Calculating at the same time, needed time for forward pass
        network.setInput(blob)  # setting blob as input to the network
        start = time.time()
        output_from_network = network.forward(layers_names_output)
        end = time.time()
        # In this way we can keep specific colour the same for every class
        np.random.seed(42)
        # randint(low, high=None, size=None, dtype='l')
        colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
        # Preparing lists for detected bounding boxes, obtained confidences and class's number
        bounding_boxes = []
        confidences = []
        class_numbers = []
        # Getting spacial dimension of input image
        h, w = image_input_shape[:2]  # Slicing from tuple only first two elements
        for result in output_from_network:
            # Going through all detections from current output layer
            for detection in result:
                # Getting class for current object
                scores = detection[5:]
                class_current = np.argmax(scores)

                # Getting confidence (probability) for current object
                confidence_current = scores[class_current]

                # Eliminating weak predictions by minimum probability
                if confidence_current > probability_minimum:
                    # Scaling bounding box coordinates to the initial image size
                    # YOLO data format keeps center of detected box and its width and height
                    # That is why we can just elementwise multiply them to the width and height of the image
                    box_current = detection[0:4] * np.array([w, h, w, h])

                    # From current box with YOLO format getting top left corner coordinates
                    # that are x_min and y_min
                    x_center, y_center, box_width, box_height = box_current.astype('int')
                    x_min = int(x_center - (box_width / 2))
                    y_min = int(y_center - (box_height / 2))

                    # Adding results into prepared lists
                    bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                    confidences.append(float(confidence_current))
                    class_numbers.append(class_current)
        results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)
        # Initialize a dictionary to count detected objects
        object_counts = defaultdict(int)
        # Count detected objects
        for line in results:
            label = labels[int(class_numbers[line])]
            object_counts[label] += 1

        # Generate the summary sentence
        if object_counts:
            # Construct a list of strings for each object count
            object_strings = [f"{count} {label}" if count == 1 else f"{count} {label}s" for label, count in object_counts.items()]
            # Join the object strings into a single sentence
            sentence = "In the scene, various entities were detected, including " + ", ".join(object_strings[:-1])
            sentence += f", and {object_strings[-1]}."
            return sentence
        else:
            print("No objects were detected.")

    def caption_gen(self, image_id):
        # Load the VisionEncoderDecoderModel
        model = VisionEncoderDecoderModel.from_pretrained("visualex/model_folder")
        # Load the ViTFeatureExtractor
        feature_extractor = ViTFeatureExtractor.from_pretrained("visualex/feature_extractor_folder")
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained("visualex/tokenizer_folder")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        max_length = 16
        num_beams = 4
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
        images = []
        # Establish a connection to MySQL
        cur = mysql.connection.cursor()
        # Debug: Print image_id for troubleshooting
        #print(f"Fetching image data for image_id: {image_id}")
        # Retrieve image data from the database
        cur.execute("SELECT image_data FROM image_metadata WHERE image_id = %s", (image_id,))
        row = cur.fetchone()
        if row is None or not row[0]:  # Check if row is None or image_data is empty
            return "Image data not found or empty in the database"
        # Get image data from the database
        image_data = row[0]
        # Convert the image data to PIL Image object
        image = Image.open(io.BytesIO(image_data))
        # Convert image to RGB mode if necessary
        if image.mode != "RGB":
            image = image.convert(mode="RGB")
        # Append the image to the images list
        images.append(image)

        pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)

        output_ids = model.generate(pixel_values, **gen_kwargs)

        preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds

    def generateText(self, image_id):
        text1 = self.objectDetection(image_id)
        text2 = self.caption_gen(image_id)
        # Ensure text1 and text2 are strings
        text2 = ' '.join(text2) if isinstance(text2, list) else text2
        text = text1 + " This image shows, " + text2 + "."
        return text
    
    def autoSelectObjects(self, image_id):
        labels = open('visualex/coco.names').read().strip().split('\n')
        weights_path = 'visualex/yolov3.weights'
        configuration_path = 'visualex/yolov3.cfg'
        probability_minimum = 0.5
        threshold = 0.3
        
        network = cv2.dnn.readNetFromDarknet(configuration_path, weights_path)
        layers_names_all = network.getLayerNames()
        output_layer_indices = network.getUnconnectedOutLayers()
        layers_names_output = [layers_names_all[i - 1] for i in output_layer_indices]
        
        cur = mysql.connection.cursor()
        cur.execute("SELECT image_data FROM image_metadata WHERE image_id = %s", (image_id,))
        row = cur.fetchone()
        if row is None or not row[0]:
            return "Image data not found or empty in the database"
        
        image_data = row[0]
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image_input_shape = img.shape
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        network.setInput(blob)
        output_from_network = network.forward(layers_names_output)
        
        np.random.seed(42)
        colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
        
        bounding_boxes = []
        confidences = []
        class_numbers = []
        h, w = image_input_shape[:2]
        
        for result in output_from_network:
            for detection in result:
                scores = detection[5:]
                class_current = np.argmax(scores)
                confidence_current = scores[class_current]
                if confidence_current > probability_minimum:
                    box_current = detection[0:4] * np.array([w, h, w, h])
                    x_center, y_center, box_width, box_height = box_current.astype('int')
                    x_min = int(x_center - (box_width / 2))
                    y_min = int(y_center - (box_height / 2))

                    bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                    confidences.append(float(confidence_current))
                    class_numbers.append(class_current)
        
        results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)
        object_counts = defaultdict(int)
        cropped_images = []
        for line in results:
            if len(cropped_images) >= 6:
                break
            label = labels[int(class_numbers[line])]  # Fix the index error here
            object_counts[label] += 1
            x_min, y_min, box_width, box_height = bounding_boxes[line]  # Fix the index error here
            cropped_img = img[y_min:y_min+box_height, x_min:x_min+box_width]
            # Encode the cropped image to base64
            _, buffer = cv2.imencode('.png', cropped_img)
            cropped_img_base64 = base64.b64encode(buffer).decode()
            cropped_images.append((cropped_img_base64, label))


        return cropped_images

class PredictionResults:
    def __init__(self, result_id=None, model_id=None, image_id=None, predicted_label=None, confidence_score=None, timestamp=None):
        self.result_id = result_id
        self.model_id = model_id
        self.image_id = image_id
        self.predicted_label = predicted_label
        self.confidence_score = confidence_score
        self.timestamp = timestamp

    def generate_audio_from_text(self, text, output_file):
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)  # Speed of speech
            engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"Error generating audio: {e}")
            return False

import cv2
import numpy

class Blur_Detection:

    @staticmethod
    def fix_image_size(image: numpy.array, expected_pixels: float = 2E6):
        ratio = numpy.sqrt(expected_pixels / (image.shape[0] * image.shape[1]))
        return cv2.resize(image, (0, 0), fx=ratio, fy=ratio)

    @staticmethod
    def estimate_blur(image: numpy.array, threshold: int = 100):
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blur_map = cv2.Laplacian(image, cv2.CV_64F)
        score = numpy.var(blur_map)
        return blur_map, score, bool(score < threshold)

    @staticmethod
    def pretty_blur_map(blur_map: numpy.array, sigma: int = 5, min_abs: float = 0.5):
        abs_image = numpy.abs(blur_map).astype(numpy.float32)
        abs_image[abs_image < min_abs] = min_abs

        abs_image = numpy.log(abs_image)
        cv2.blur(abs_image, (sigma, sigma))
        return cv2.medianBlur(abs_image, sigma)
