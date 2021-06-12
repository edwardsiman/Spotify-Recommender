## Spotify Recommendation System with Content and Collaborative Recommendation

This project is a recommendation system unsupervised machine learning application to provide recommendation by grouping similar songs based on its attribute such as artists, genres, tempos, loudness, etc. This project is inspired from the medium article of 'Machine learning and recommender systems using your own Spotify data'  (https://towardsdatascience.com/machine-learning-and-recommender-systems-using-your-own-spotify-data-4918d80632e3) with some modifications in the machine learning algorithm on content and collaborative recommendation

The dataset for this project is taken from the Spotify Open API provided in the Spotify Developer Page. The code to access this dataset can be seen in spotify_data.py which leverages the function from call_function.py

The machine learning model applied to this project is K-Nearest Neighbor model where a cluster center is placed by selecting the n closest neighbors for collaborative recommendation. The model for content recommendation is leveraging TF-IDF and cosine similarities. 
The required package in this machine learning models are:
- Pandas (v 1.0.4)
- Numpy (v 1.17.2)
- Matplotlib (v 3.1.1)
- Scikit-Learn (v 0.21.3)
- Scipy (v 1.6.2)
- Spotipy (v 2.18.0)
- Yaml (v 0.2.5)

Thanks, kindly provide any feedback or suggestion regarding the project