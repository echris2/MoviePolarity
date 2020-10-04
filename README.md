# Movie Polarity
Project using various supervised learning algorithms to predict whether or not a movie is Controversial based on its MovieLens ratings. See R markdown file or knitted pdf version for full write up of analysis. Data has been collected from Kaggle's Movies Database (https://www.kaggle.com/rounakbanik/the-movies-dataset)

## Instructions to recreate analysis

1.	Go to https://www.kaggle.com/rounakbanik/the-movies-dataset 

2.	Download the keywords, links, movies_metadata, and rating CSV files, and add them under the Data folder

3.	Run the RunMe.py file
    - If it is your first time running the file, you will need to load the database [Enter ‘Y’ when at the first prompt in the script]
  
    - In order to train the Word2Vec model, you must download the pre-endcoded word vectors from the Google News corpus here: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
      - This is a large file and downloading will take some time
      - Once download is complete, add the file to the top level folder (same directory as 'RunMe.py') and enter 'N' at the prompt. The model will be saved as preTrainGoog.model, and you can enter 'Y' at the promots in subsequent runs

4. 	Once python script has finished its run, should take about 7 minutes (with database loaded and Word2Vec model trained), csv file, “TheMovieDatabase_keywords.csv”, should be located in the Data Folder. 

    - To generate specific Word Clouds, use the defined functions: cloudCluster(), with a keyword cluster number as an argument, or cloudCluster_byId(), with a movie ID as an argument.

5.	Run the R Markdown script in its entirety, or by chunks sequentially

