import praw

class Bot:

    reddit_obj = None

    def __init__(self):
        #credentials are stored in ini file
        self.reddit_obj = praw.Reddit("bot1")
    #Returns image URL, submission object, and submission URL
    def get_image_and_post(self):
        cat_sub = self.reddit_obj.subreddit("cat")

        #Get new submissions
        for submission in cat_sub.new(limit= 10):
            #Make sure not to comment on mourning posts out of respect
            if("died" not in submission.title or "dead" not in submission.title 
               or "passed away" not in submission.title or "pass away" not in submission.title):
                #checks if each submission has images
                if hasattr(submission, 'media_metadata'):
                    for key in submission.media_metadata:
                        for image in submission.media_metadata[key]['p']:
                            #make sure images are large enough to be processed
                            if(image['y'] >= 640 and image['x'] >= 640):
                                return image['u'], submission, submission.url
    
    def make_comment(self, post_object, breed):
        post_object.reply("Hello!\n According to my model your cat is a " + breed + ".\n Am I right?\n")

    def __str__(self):
        return "Username: " + self.username