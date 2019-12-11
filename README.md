# PokerFace
This is the repository for UW-Amazon DeepLens Hackathon.<br>
Team member: Shih-Hao Yeh, Shou-En Hsiao, Wei-Ning Liang

Problem Statement
---
We noticed that some skillful poker players will tend to fool their opponents when playing. We want to apply deep learning to extract the features from the opponents' face and use the features to help identify the winning rate for the opponents. By doing so, we can decide whether we should call or fold. 

In our setting, we choose Texas Hold'em (heads-up) as our target poker game.

Solution
---
We think that the face of a player can show some sign even the player tries to hide it. However, we also think that using only the face is not enough for predicting the winning rate. Therefore, we consider also the winning rate of ourselves (by calculating the probability with our cards and those on the table) and use 1 minus the winning rate as the expectation of opponent's winning rate. By weighting between the pre-predict winning rate from the face model and the expectation winning rate we calculate, we can get the final predict winning rate for the opponent.

Approach
---
First, the lambda function on deeplens will keep detecting face and predict the pre-predict winning rate with two model. One is the face detection model and the other one is the model that takes in the cropped face and output the pre-predict winning rate. After getting the pre-predict winning rate, we will store the value onto S3 and wait for another lambda function being triggered to collect this value.

The other lambda function will be triggered when we publish the expected winning rate. This serve as a starting sign that tells deeplens that we want to calculate the current winning rate for the opponent. The lambda function will retrieve the value from S3 and use the expected winning rate to calculate a weighted final prediction and then publish the result.

Challenge
---
Since playing poker game is a continuous motion, it's hard to use only the face image extract from one frame to predict the winning rate. The message from a player's face should be a sequence of motion. Therefore, during our training, we found that the training loss has dropped, but the validation loss tends to vibrate around 50%, meaning that the predicted result is very unreliable.

If we have more time, we are thinking of using a sequence of frame instead of only one frame as the input for the pre-predict model.
