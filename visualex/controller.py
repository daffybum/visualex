from visualex import entity

# User Account related Controllers
class LoginController:
    def __init__(self):
        self.userAccount = entity.UserAccount()

    def userLogin(self, username, password):
        return self.userAccount.login(username, password)

class CreateUserAccController:
    def __init__(self):
        self.userAccount = entity.UserAccount()
    def createUserAccount(self, userAcc):
        return self.userAccount.createUserAcc(userAcc)
    
class GetAllEmailsController:
    def __init__(self):
        self.email_list = entity.UserAccount()
    def getEmails(self):
        return self.email_list.get_all_emails()
    
class ChangePasswordController:
    def __init__(self):
        self.userAccount = entity.UserAccount()
    def changePW(self, username, password):
        return self.userAccount.changePW(username, password)
    

class UploadFeedbackController:
    def __init__(self):
        self.feedback_forum = entity.FeedbackForum()
    def uploadFeedback(self, username, feedback):
        return self.feedback_forum.submitfeedback(username, feedback)
    
class ViewFeedbackController:
    def __init__(self):
        self.feedback_list = entity.FeedbackForum()
    def viewFeedback(self):
        return self.feedback_list.get_all_feedback()
    
class CheckMembershipController:
    def __init__(self):
        self.userAccount = entity.UserAccount()
    def checkMembershipExist(self, username):
        return self.userAccount.checkMembershipExist(username)
    
# Transaction Related Controller
class MakePaymentController:
    def __init__(self):
        self.transactions = entity.Transactions()
    def makePayment(self, username, charges):
        return self.transactions.make_payment(username, charges)
    
class GetInvoiceController:
    def __init__(self):
        self.display = entity.Transactions()
    def viewDisplay(self,payment_timestamp):
        return self.display.get_invoice(payment_timestamp)
    
class ViewHistoryController:
    def __init__(self):
        self.history_log = entity.HistoryLogs()
    def viewHistory(self, username):
        return self.history_log.get_history_logs_with_predictions(username)

class MembershipController:
    def __init__(self):
        self.view_membership_tier = entity.UserAccount()

    def get_membership_tier_info(self, username):
        return self.view_membership_tier.get_membership_tier_info(username)
    
    def getAllmembership(self):
        return self.view_membership_tier.getAllmembership()

    def getUserMembership(self, username):
        return self.view_membership_tier.get_membership_tier(username)

class AssignMembershipController:
    def __init__(self):
        self.userAccount = entity.UserAccount()

    def assign_membership(self, username, membership_tier):
        return self.userAccount.assignMembership(username, membership_tier)

class DisplayController:
    def __init__(self):
        self.userAccount = entity.UserAccount()

    def get_user_info(self, username):
        return self.userAccount.get_user_info(username)
    def get_user_info2(self, username):
        return self.userAccount.get_user_info2(username)
    def get_user_info3(self, username):
        return self.userAccount.get_user_info3(username)
    
    
class GetAllUsersController:
    def __init__(self):
        self.userAccount = entity.UserAccount()
    
    def get_all_users(self):
        return self.userAccount.get_all_users()
    
class SearchUserController:
    def __init__(self):
        self.userAccount = entity.UserAccount()

    def search_user(self,username):
        return self.userAccount.search_user(username)

class DeleteController:
    def __init__(self):
        self.userAccount = entity.UserAccount()

    def delete_profile(self, username):
        return self.userAccount.delete_account(username, )
    

class StoreImagesController:
    def __init__(self):
        self.storeImage = entity.ImageData()
    
    def store_img(self, image_path):
        return self.storeImage.store_image(image_path)
    
class TextToAudioController:
    def __init__(self):
        self.audio = entity.PredictionResults()

    def generate_audio_from_text(self, text, output_file):
        return self.audio.generate_audio_from_text(text, output_file)

    def generate_story_audio_from_text(self, text, output_file):
        return self.audio.generate_story_audio_from_text(text, output_file)

class GenerateTextController:
    def __init__(self):
        self.genText = entity.ImageData()
        
    def generate_text(self, image_id):
        return self.genText.generateText(image_id)

class EditProfileController:
    def __init__(self):
        self.userAccount = entity.UserAccount()
    def edit_profile(self, oldUsername,  name, surname, email, dob, address, membership):
        return self.userAccount.edit_profile(oldUsername, name, surname, email, dob, address, membership)
    def edit_profile1(self, oldUsername, name, surname, email, dob, address):
        return self.userAccount.edit_profile1(oldUsername, name, surname, email, dob, address)

class storePredictedResultsController:
    def __init__(self):
        self.historylogs = entity.HistoryLogs()

    def store_PredictedResults(self, username, image_id, predicted_label, laplacian_score):
        return self.historylogs.store_predictedResults(username, image_id, predicted_label, laplacian_score)
    
class AutoSelectObjectsController:
    def __init__(self):
        self.objectImages = entity.ImageData()
        
    def auto_select_objects(self, image_id):
        return self.objectImages.autoSelectObjects(image_id)

class StoryTellingController:
    def __init__(self):
        self.story = entity.ImageData()
        
    def story_teller(self, object_list):
        return self.story.storyTelling(object_list)

class imagesGenerationController:
    def __init__(self):
        self.images = entity.ImageData()
    def imagesGenerator(self, prompt):
        return self.images.imagesGeneration(prompt)
    
class ReplyToFeedbackController:
    def __init__(self):
        self.reply = entity.FeedbackForum()
        
    def replyToFeedback(self, feedback_id, reply):
        return self.reply.reply_feedback(feedback_id, reply)
    
class ViewRepliesController:
    def __init__(self):
        self.reply_list = entity.FeedbackForum()
        
    def getReplies(self):
        return self.reply_list.get_replies()
    
class VisionDescriptionController:
    def __init__(self):
        self.vision = entity.ImageData()
        
    def vision_description(self, image_id):
        return self.vision.visionDescription(image_id)
