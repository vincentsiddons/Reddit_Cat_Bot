from Model import Model
from Bot import Bot
import time

reddit_bot = Bot()

#dummy values, we aren't training
model = Model(1, 1)

while True:
    input = reddit_bot.get_image_and_post()
    breed_name = model.get_breed(input[0])
    reddit_bot.make_comment(input[1], breed_name)
    #Wait every 15 minutes before selecting a post to comment on
    time.sleep(901)





