from flask import Blueprint, render_template, request, flash, redirect, url_for, session, jsonify
from . import mysql

import stripe

from werkzeug.security import generate_password_hash

from visualex import controller
from visualex import entity

import urllib.request
import os
from werkzeug.utils import secure_filename

import cv2

boundary = Blueprint('boundary', __name__)  # Blueprints means it has roots inside a bunch of URLs defined

stripe.api_key = "sk_test_51Ou5FZRqVdY5zwenlu9FnQVHaDupGLqxqjC6J6eyBXR09ZccqROeV85QcLOCWb8wFtQYMT4P3FlaIGOOmxJHHFCa00K7pXYYUU"

YOUR_DOMAIN = "http://localhost:5000"

# AUTHENTICATION
@boundary.route('/', methods=['GET', 'POST'])
def login():                                     
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        loginController = controller.LoginController()
        user = loginController.userLogin(username, password)
        #print(user.username)
        if (user):
            return redirect(url_for('boundary.home'))
        else:
            flash('Wrong password or username', category='error')

    return render_template("login.html", boolean=True)

@boundary.route('/home', methods=['GET', 'POST'])
def home():
    username = session.get('username')
    return render_template("homepage.html", user_name = username)

@boundary.route('/forgotpw', methods=['GET', 'POST'])
def forgotpw():
    if request.method == 'POST':
        username = request.form.get('username')
        password1 = request.form.get('newPassword')
        password2 = request.form.get('cfmPassword')

        if password1 != password2:
            flash('Password don\'t match.', category='error')
        elif len(password1) < 7:
            flash('Password must be at least 7 characters.', category='error')
        else:
            changePasswordController = controller.ChangePasswordController()
            password1 = generate_password_hash(password1, method='pbkdf2')
            result = changePasswordController.changePW(username,password1)
            if (result):
                flash('Password Changed!', category='success')
            else:
                flash('Cannot change Password!', category='error')

    return render_template("forgotpw.html")

@boundary.route('/logout')
def logout():
    return"<p>Logout</p>"

@boundary.route('/sign-up', methods=['GET', 'POST'])
def sign_up():
    if request.method == 'POST':
        email = request.form.get('email')
        username = request.form.get('username')
        name = request.form.get('name')
        surname = request.form.get('surname')
        date_of_birth = request.form.get('date_of_birth')
        address = request.form.get('address')
        password1 = request.form.get('password1')
        password2 = request.form.get('password2')

        if len(email) < 4:
            flash('Email must be greater than 3 characters.', category='error')
        elif len(name) < 2:
            flash('Name must be greater than 1 character.', category='error')
        elif password1 != password2:
            flash('Password don\'t match.', category='error')
        elif len(password1) < 7:
            flash('Password must be at least 7 characters.', category='error')
        else:
            createAccountController = controller.CreateUserAccController()
            password1 = generate_password_hash(password1, method='pbkdf2')
            userAcc = entity.UserAccount(username, password1, name, surname, email, date_of_birth, address)
            result = createAccountController.createUserAccount(userAcc)
            if (result):
                flash('Account created!', category='success')
            else:
                flash('Cannot Create Account!', category='error')

    return render_template("sign_up.html")

@boundary.route('/about-us')
def aboutus():
    username = session.get('username')
    return render_template("AboutUs.html", user_name = username)

@boundary.route('/membersubscription')
def membersubscription():
    username = session.get('username')
    return render_template('membersubscription.html', user_name = username)


@boundary.route('/paymentSuccess')     # payment page
def handle_payment_success():
    username = session.get('username')
    getInvoiceController = controller.GetInvoiceController()
    makePaymentController = controller.MakePaymentController()
    assignMembershipController = controller.AssignMembershipController()
    charges = 20.00
    membership = "premium"
    payment = makePaymentController.makePayment(username, charges)
    display = getInvoiceController.viewDisplay(payment)
    assignMembershipController.assign_membership(username, membership)
    return render_template("paymentSuccess.html", user_name = username, display = display, tier = membership.capitalize())


@boundary.route('/create-payment-session', methods=['POST'])  # process payment
def create_checkout_session():
    try:

        username = session.get('username')

        checkMembershipController = controller.CheckMembershipController()

        check_membership = checkMembershipController.checkMembershipExist(username)

        if 'premium-membership' in request.form:

            if(check_membership == 1):
                 flash('Already has premium membership!', category='error')
                 return render_template("membersubscription.html", user_name = username)
            else:
                checkout_session = stripe.checkout.Session.create(
                    line_items = [
                        {
                            'price' : 'price_1Ou8paRqVdY5zwen2Qyhu1Ii',
                            'quantity':1
                        }
                    ],
                    mode="subscription",
                    success_url=url_for('boundary.handle_payment_success', _external=True),
                    cancel_url=YOUR_DOMAIN + "/membersubscription"
                )
    except Exception as e:
        return str(e)
    
    return redirect(checkout_session.url, code=303)


@boundary.route('/userfb', methods=['GET'])
def user_feedback():
    username = session.get('username')
    feedback_controller = controller.ViewFeedbackController()
    feedback_list = feedback_controller.viewFeedback()
    
    replyController = controller.ViewRepliesController()
    reply_list = replyController.getReplies()
    print(reply_list)
    
    return render_template("feedbackUserPage.html", feedback_list=feedback_list, user_name = username, reply_list=reply_list)


@boundary.route('/submitfb', methods=['GET', 'POST'])
def submit_feedback():
    username = session.get('username')
    if request.method == 'POST':
        feedback_content = request.form['feedbackText']
        submitFBController = controller.UploadFeedbackController()
        submitFB = submitFBController.uploadFeedback(username, feedback_content)
        # uploading test
        if submitFB:
            flash('Feedback submitted!', category='success')
            # Redirect back to the user feedback page after submitting feedback
            return redirect(url_for('boundary.user_feedback'))
        else:
            flash('Feedback Submission Invalid!', category='error')
                
    return render_template("feedbackSubmitPage.html", user_name = username)



@boundary.route('/adminfb', methods=['GET'])
def admin_feedback():
    username = session.get('username')
    feedback_controller = controller.ViewFeedbackController()
    feedback_list = feedback_controller.viewFeedback()
    return render_template("feedbackAdminPage.html", feedback_list=feedback_list, user_name=username)



# Upload Image Function
UPLOAD_FOLDER = 'visualex/static/uploads/'
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
 
@boundary.route('/uploadImage')
def upload():
    username = session.get('username')
    return render_template('uploadImage.html', user_name = username)
 
@boundary.route('/uploadImage', methods=['POST'])
def upload_image():
    username = session.get('username')
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(UPLOAD_FOLDER, filename))
        static_folder = os.path.abspath('visualex')
        relative_filepath = r"static/uploads/" + filename
        image_path = os.path.join(static_folder, relative_filepath)
        image = image_path
        image = cv2.imread(str(image))
        results = []

        blur_map,score, blurry = entity.Blur_Detection.estimate_blur(image, threshold=250.0)
        results.append({'Laplacian score': score, 'blurry': blurry})
        for item in results:
                # Construct a formatted string for the dictionary
                formatted_string = ', '.join(f"{key}: {value}" for key, value in item.items())
        if blurry:
            flash(formatted_string, category='error')
            flash('Image is blurry please upload another Image',category='error')
            image_path = os.path.join(UPLOAD_FOLDER, filename)
            os.remove(image_path)
            return redirect(request.url)
        else:
            image_controller = controller.StoreImagesController()
            image_path = os.path.join(UPLOAD_FOLDER, filename)
            image_id = image_controller.store_img(image_path)
            session['image_id'] = image_id
            session['laplacian_score'] = score # store the laplacian score in the session
            session['filename'] = filename
            flash(formatted_string)
            print('upload_image filename: ' + filename)
            flash('Image successfully uploaded')
            return render_template('uploadImage.html', filename=filename, image_id=image_id, user_name=username)
    else:
        flash('Allowed image types are - png and jpeg only')
        return redirect(request.url)
 
@boundary.route('/display/<filename>')  #display image on uploadImage.html
def display_image(filename):
    print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
 
@boundary.route('/history', methods=['GET', 'POST'])
def history_logs():
    username = session.get('username')
    history_logs_controller = controller.ViewHistoryController()
    history_logs = history_logs_controller.viewHistory(username)
    return render_template('history.html', history_logs=history_logs, user_name=username)

@boundary.route('/viewmembershiptier')
def view_membership_tier():
    username = session.get('username')
    membership_controller = controller.MembershipController()
    membership_tier = membership_controller.get_membership_tier_info(username)
    return render_template('viewmembershiptier.html', membership_tier=membership_tier, user_name=username)

@boundary.route('/accountDetail', methods=['GET', 'POST'])
def display_profile():
    username = session.get('username')
    if username:
        display= controller.DisplayController()
        user = display.get_user_info2(username)
        if user:
            username ,name, surname, email, date_of_birth, address, membershipTier = user
            return render_template("accountdetail.html", username=username,name=name, surname=surname, email=email, date_of_birth=date_of_birth, address=address, membershipTier = membershipTier, user_name = username)
        else:
            flash('User not found', category='error')
            return redirect(url_for('boundary.login'))
    else:
        flash('User not logged in', category='error')
        return redirect(url_for('boundary.login'))

# Admin View All users details Main page   
@boundary.route('/viewAllUsers')
def viewAllUsers():
    username = session.get('username')
    getAllUserController = controller.GetAllUsersController()
    users_list = getAllUserController.get_all_users()
    return render_template('viewAllUsersAccount.html', user_name = username, users_list = users_list, search_exist = False)

# If admin clicks view button
@boundary.route('/viewUserDetails', methods=['POST'])
def viewUserDetails():
    username = session.get('username')
    selected_user = request.form['selectedUsername']
    display= controller.DisplayController()
    user = display.get_user_info2(selected_user)
    if user:
        selected_user,name, surname, email,date_of_birth, address, membershipTier = user
        return render_template("accountdetail.html", username=selected_user,name=name, surname=surname, email=email,date_of_birth=date_of_birth, address=address, user_name = username, membershipTier = membershipTier)


# If admin clicks search
@boundary.route('/viewSearchedUserDetails', methods=['POST'])
def viewSearchedUserDetails():
    username = session.get('username')
    inputUsername = request.form.get('inputUsername')
    searchUserController = controller.SearchUserController()
    searchedUsername = searchUserController.search_user(inputUsername)
    if searchedUsername is None:
        flash('Username does not exist!', category='error')
        getAllUserController = controller.GetAllUsersController()
        users_list = getAllUserController.get_all_users()
        return render_template('viewAllUsersAccount.html', user_name = username, users_list = users_list, search_exist = False)
    else:
        return render_template('viewAllUsersAccount.html', user_name = username, users = searchedUsername ,search_exist = True)

@boundary.route('/deleteAccount', methods=['GET', 'POST'])
def delete_account():
        username = session.get('username')
        selected_user = session.get('selected_user')
        if username == "admin":
            userController = controller.DeleteController()
            result = userController.delete_profile(selected_user)
            if result :
                # Account deleted successfully
                # You might want to clear the session and provide a confirmation message
                return redirect(url_for('boundary.login'))
            else:
                # Account deletion failed (username not found, database error, etc.)
                flash('Failed to delete your account. Please try again later.', category='error')
                return redirect(url_for('boundary.accountDetail'))
        else:
            userController = controller.DeleteController()
            result = userController.delete_profile(username)
            if result :
                # Account deleted successfully
                # You might want to clear the session and provide a confirmation message
                return redirect(url_for('boundary.login'))
            else:
                # Account deletion failed (username not found, database error, etc.)
                flash('Failed to delete your account. Please try again later.', category='error')
        return redirect(url_for('boundary.accounDetail'))

@boundary.route('/generateaudio', methods=['GET', 'POST'])
def generate_audio():
    if request.method == 'POST':
        text = request.form.get('text')
        username = session.get('username')
        image_id = request.form.get('image_id')
        prediction_result = session.get('prediction_result')
        filename = session.get('filename')
        output_file = 'visualex/static/audio.mp3'  # Output file path for generated audio
        text_to_audio_controller = controller.TextToAudioController()
        success = text_to_audio_controller.generate_audio_from_text(text, output_file)
        if success:
            flash('Audio generated successfully!', category='success')  # Flash success message
    return render_template("uploadImage.html", text=text, user_name=username, image_id=image_id, prediction_result=prediction_result, filename=filename)

@boundary.route('/generatestoryaudio', methods=['GET', 'POST'])
def generate_storyaudio():
    if request.method == 'POST':
        text = request.form.get('text')
        username = session.get('username')
        image_id = request.form.get('image_id')
        prediction_result = session.get('prediction_result')
        filename = session.get('filename')
        output_file = 'visualex/static/storyaudio.mp3'  # Output file path for generated audio
        text_to_audio_controller = controller.TextToAudioController()
        success = text_to_audio_controller.generate_story_audio_from_text(text, output_file)
        if success:
            flash('Audio generated successfully!', category='success')  # Flash success message
    return render_template("generateStory.html", user_name=username, image_id=image_id, prediction_result=prediction_result, filename=filename)
    
@boundary.route('/assignmembership', methods=['GET', 'POST'])
def assign_membership():
    # Retrieve username and membership tier
    membership_controller = controller.MembershipController()
    users = membership_controller.getAllmembership()

    if request.method == 'POST':
        username = request.form['username']
        membership_tier = request.form['membership_tier']

        assignmembershipController = controller.AssignMembershipController()
        result = assignmembershipController.assign_membership(username, membership_tier)
        print(result)
        if result: 
            flash('Membership Assigned Successfully!', category='success') 
        else: 
            flash('Failed to assign membership', category='error')

    return render_template("assignmembership.html", users=users)

@boundary.route('/generateText', methods=['POST'])
def generateText():
    username = session.get('username')
    image_id = request.form.get('image_id')
    laplacian_score = session.get('laplacian_score')
    filename = session.get('filename')
    print(filename)
    prediction_controller = controller.GenerateTextController()
    prediction_result = prediction_controller.generate_text(image_id)
    session['prediction_result'] = prediction_result
    storePredictedResultsController = controller.storePredictedResultsController() # store entry in prediction_results table
    storePredictedResultsController.store_PredictedResults(username, image_id,prediction_result,laplacian_score)
    # Do something with the image_id, such as storing it in a database
    return render_template('uploadImage.html', user_name=username, image_id=image_id, prediction_result=prediction_result, filename=filename)

@boundary.route('/editProfile', methods=['GET', 'POST'])
def editProfile():
    username = session.get('username')
    selected_user = session.get('selected_user')
    display= controller.DisplayController()

    
    if username == 'admin':
        user = display.get_user_info2(selected_user)
        selected_user,name, surname, email, date_of_birth, address, membershipTier = user
        if request.method == 'POST':
            email1 = request.form.get('email')
            username1 = request.form.get('username')
            name1 = request.form.get('name')
            surname1 = request.form.get('surname')
            date_of_birth1 = request.form.get('date_of_birth')
            address1 = request.form.get('address')
            membershipTier1 = request.form.get('membershipTier')
            editProfileController = controller.EditProfileController()
            editProfile = editProfileController.edit_profile(selected_user, username1, name1, surname1, email1, date_of_birth1, address1, membershipTier1)
            if editProfile:
                flash('Profile updated successfully')
            else:
                flash('Cannot update profile')
        return render_template("editProfile.html", username=selected_user,name=name, surname=surname, email=email, address=address, user_name = username, membershipTier = membershipTier, dob=date_of_birth)
    else:
        user = display.get_user_info3(username)
        username,name, surname, email, date_of_birth, address= user
        if request.method == 'POST':
            email1 = request.form.get('email')
            username1 = request.form.get('username')
            name1 = request.form.get('name')
            surname1 = request.form.get('surname')
            date_of_birth1 = request.form.get('date_of_birth')
            address1 = request.form.get('address')
            membershipTier1 = request.form.get('membershipTier')
            editProfileController = controller.EditProfileController()
            editProfile = editProfileController.edit_profile1(username, username1, name1, surname1, email1, date_of_birth1, address1)
            if editProfile:
                flash('Profile updated successfully')
            else:
                flash('Cannot update profile')
        return render_template("editProfile.html", username=selected_user,name=name, surname=surname, email=email, address=address, user_name = username, dob=date_of_birth)

# Admin clicks on view all Logs
@boundary.route('/viewAllLogs')
def viewAllLogs():
    username = session.get('username')
    getAllUserController = controller.GetAllUsersController()
    users_list = getAllUserController.get_all_users()
    return render_template('viewAllHistoryLogs.html', user_name = username, users_list = users_list, search_exist = False)

# If admin clicks search
@boundary.route('/viewSearchedUserLogs', methods=['POST'])
def viewSearchedUserLogs():
    username = session.get('username')
    inputUsername = request.form.get('inputUsername')
    searchUserController = controller.SearchUserController()
    searchedUsername = searchUserController.search_user(inputUsername)
    if searchedUsername is None:
        flash('Username does not exist!', category='error')
        getAllUserController = controller.GetAllUsersController()
        users_list = getAllUserController.get_all_users()
        return render_template('viewAllHistoryLogs.html', user_name = username, users_list = users_list, search_exist = False)
    else:
        return render_template('viewAllHistoryLogs.html', user_name = username, users = searchedUsername ,search_exist = True)
    
# If admin clicks view button (viewAllHistoryLogs)
@boundary.route('/viewUserLogs', methods=['POST'])
def viewUserLogs():
    username = session.get('username')
    selected_user = request.form['selectedUsername']
    history_logs_controller = controller.ViewHistoryController()
    history_logs = history_logs_controller.viewHistory(selected_user)
    return render_template('history.html', history_logs=history_logs, user_name=username)

@boundary.route('/cancel-membership', methods=['POST'])
def cancel_membership():
    username = session.get('username')
    assignMembershipController = controller.AssignMembershipController()
    membership = "basic"
    assignMembershipController.assign_membership(username, membership)
    membership_controller = controller.MembershipController()
    membership_tier = membership_controller.get_membership_tier_info(username)
    flash('Membership Cancelled Successfully!')
    return render_template('viewmembershiptier.html', user_name = username, membership_tier = membership_tier)

@boundary.route('/autoSelectImages', methods=['POST'])
def auto_select_objects():
    username = session.get('username')
    image_id = session.get('image_id')
    filename = session.get('filename')
    membershipController = controller.MembershipController()
    membership = membershipController.getUserMembership(username)
    
    # Check if membership is premium
    if membership != 'premium':
        message = "Membership not premium. Please subscribe to use this feature"
        return render_template('uploadImage.html', message=message, user_name=username, image_id=image_id, filename=filename)
    
    autoSelectController = controller.AutoSelectObjectsController()
    cropped_images = autoSelectController.auto_select_objects(image_id)
    return render_template('generateStory.html', cropped_images=cropped_images, user_name=username)

@boundary.route('/generate_story', methods=['POST'])
def generate_story():
    order = request.form['order']  # Get the order array sent from the client
    username = session.get('username')
    image_id = session.get('image_id')
    autoSelectController = controller.AutoSelectObjectsController()
    cropped_images = autoSelectController.auto_select_objects(image_id)
    # For debugging purposes (can ignore it)
    print("Boundary: ")
    print(type(order))
    print(order)
    list_order = order.split(",")
    # For debugging purposes (can ignore it)
    print(type(list_order))
    print(list_order)
    #story_result = "hi it's working!"
    # Process the order array as needed
    story_controller = controller.StoryTellingController()
    story_result = story_controller.story_teller(list_order)
    session['story_result'] = story_result
    #print(story_result)
    return render_template('generateStory.html', user_name=username, story=story_result, cropped_images=cropped_images)

@boundary.route('/imageGeneration')
def image_gen():
    username = session.get('username')
    return render_template('imagesGeneration.html', user_name = username)

@boundary.route('/imagesGeneration', methods = ['POST'])
def generate_image():
    images = []
    prompt = request.form['prompt']
    username = session.get('username')
    print("have reach boundary")
    membershipController = controller.MembershipController()
    membership = membershipController.getUserMembership(username)
    
    # Check if membership is premium
    if membership != 'premium':
        message = "Membership not premium. Please subscribe to use this feature"
        return render_template('imagesGeneration.html', message=message, user_name=username)
    
    images_controller = controller.imagesGenerationController()
    images_result = images_controller.imagesGenerator(prompt)
    if len(images_result)>0:
        for img in images_result:
            images.append(img['url'])

    return render_template("imagesGeneration.html",user_name = username, prompt=prompt, images = images)

@boundary.route('/replyfeedback', methods=['POST'])
def reply_feedback():
    # Get data from the form
    feedback_id = request.form['feedback_id']
    reply_content = request.form['reply_content']
    
    replyFBController = controller.ReplyToFeedbackController()
    reply = replyFBController.replyToFeedback(feedback_id, reply_content)
    
    if reply:
        flash('Reply sent successfully', 'success')
    else:
        flash('Failed to send reply', 'error')
    
    username = session.get('username')          
    feedback_controller = controller.ViewFeedbackController()
    feedback_list = feedback_controller.viewFeedback()
    return render_template("feedbackAdminPage.html", feedback_list=feedback_list, user_name=username)
