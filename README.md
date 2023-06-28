# Sentiment-Analysis-of-live-Youtube-comments-using-NLP
Summary: Sentiment analysis can be used to analyze the opinions and attitudes expressed in the text data related to the Dalai Lama controversy. A controversy had erupted recently after an incident in which the Dalai Lama kissed a young boy during a public function. The incident had led to an international furor with allegations of child abuse against the Dalai Lama. I wanted to explore how different cultures react to the controversy.

Data collection: In order to generate original dataset, I decided to scrape the YouTube comments. Three videos have been chosen which have been uploaded by spokesperson from Ladakh, China and India. This will help in attracting comments from the various cultures which will in turn help us analyze the sentiments of the public. The links to the videos are:
Sonam Wangchuk: https://www.youtube.com/watch?v=YipZ7day1rw
South China Morning Post: https://youtu.be/M3UuGKRmakI 
India Today: https://www.youtube.com/watch?v=J1xSK3mPZ9k

Data preprocessing: I am preprocessing the collected data by removing any irrelevant information such as stopwords, special characters, punctuations, HTML tags and any character which differs from English language character. I have also performed stemming to reduce the number of unique words in the comments data. The comments have been converted into lower-case characters. The libraries used for data preprocessing include TextBlob, metrics,langdetect ,nltk, Stopwords, word_tokenize, string. We are segregating the data into positive and negative comments to measure the degree to which the text expresses a positive or negative sentiment.

Model training: I have trained the machine learning model such as Logistic Regression and Naive Bayes using the extracted features and corresponding sentiment labels. 

Model evaluation: The performance of the trained model is evaluated with the help of appropriate metrics such as accuracy, precision, recall, and F1 score. A classification report for both Logistic Regression and Multinomial Naïve Baye’s models have been incorporated in the code.

We can see that the accuracy for Logistic Regression(86%) is higher than that for Multinomial Naïve Bayes’ model(53%). To improve the accuracy, I have preprocessed the data thoroughly, which helped me achieve slightly higher accuracy.

This project can help us understand the overall sentiment of the public towards the controversy and identify any patterns or trends.

Note: When you run the function for scraping comments from YouTube videos, a Chrome/web browser icon will pop up in the taskbar, please open it as soon as possible to avoid running into an error, you just need to click on the icon and the comments will be scraped automatically and it will be closed automatically. In case you run into an error, please increase the wait time in the code, ‘wait = WebDriverWait(driver, 100)’.  Please install the library ‘langdetect’ to avoid running into an error. 
Command: pip install langdetect
