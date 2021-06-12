import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import spotipy
import yaml
import logging
import argparse
from spotipy.oauth2 import SpotifyOAuth
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

class Recommender:
    def __init__(self, songs, recommender_type, n_songs):
        #Read the data of top artists and tracks alongside the saved playlist
        self.df_top_artist = pd.read_pickle('spotify/top_artists.pkl')
        self.df_top_tracks = pd.read_pickle('spotify/top_tracks.pkl')
        self.df_playlist_tracks = pd.read_pickle('spotify/playlist_tracks.pkl')
        self.df_recommendation = pd.read_pickle('spotify/recommendation_tracks.pkl')
        self.songs = songs
        self.recommender_type = recommender_type
        self.n_songs = n_songs
        self.main(self.recommender_type)

    def preprocessing(self):
        #Standardize the popularity rating to between 0-1
        self.df_top_tracks['popularity'] = self.df_top_tracks['popularity'] / 100
        self.df_playlist_tracks['popularity'] = self.df_playlist_tracks['popularity'] / 100

        #Remove duplicate songs
        self.df_playlist_tracks = self.df_playlist_tracks.drop_duplicates(subset = 'name', keep = 'first')

        #Drop the unwanted column
        self.df_top_artist = self.df_top_artist.drop(['uri','type'], axis = 1)
        self.df_top_tracks = self.df_top_tracks.drop(['type','is_local','album_artist_id','album_artist_name','album_tracks','album_type','album_genres','uri','track_href','analysis_url'], axis = 1)
        self.df_playlist_tracks = self.df_playlist_tracks.drop(['type','is_local','album_artist_id','album_artist_name','album_tracks','album_type','album_genres','playlist_id','playlist_name','playlist_tracks','added_at','added_by','uri','track_href','analysis_url'], axis = 1)
        #Combine the top_tracks and playlist_tracks together by using union and check for duplicates
        self.df_tracks = pd.concat([self.df_top_tracks, self.df_playlist_tracks], axis = 0)
        self.df_tracks = self.df_tracks.drop_duplicates(subset = 'name', keep = 'first')

        #Fill the na value in the data with 0
        self.df_tracks = self.df_tracks.fillna(0)

        #Reset the index for the combined dataset
        self.df_tracks = self.df_tracks.reset_index()
    
    def collaborative_recommendation(self):
        #Pick the numerical column from the dataset
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        self.df_tracks_numeric = self.df_tracks.select_dtypes(include=numerics)
        self.df_tracks_numeric = self.df_tracks_numeric.drop(['index','duration_ms','disc_number','track_number'], axis=1)

        #Normalize the values inside dataframe and turn into csr matrix
        standardization = MinMaxScaler()
        x_val = standardization.fit_transform(self.df_tracks_numeric.values)
        self.df_tracks_numeric = pd.DataFrame(x_val)
        self.df_tracks_numeric = csr_matrix(self.df_tracks_numeric.values)

        #Train the data using K Nearest Neigbors
        self.knn = NearestNeighbors(n_neighbors = 15, metric = 'cosine', algorithm = 'auto')
        self.knn.fit(self.df_tracks_numeric)

    #Get the recommendation from some of the songs using collaborative recommendation
    def recommend_songs_collaborative(self,song_name,n_songs):
        songs_list = self.df_tracks[self.df_tracks['name'].str.contains(song_name)]
        if len(songs_list) > 0:
            songs_idx = songs_list.iloc[0]['id']
            songs_idx = self.df_tracks[self.df_tracks['id'] == songs_idx].index[0]
            distances,indices = self.knn.kneighbors(self.df_tracks_numeric[songs_idx],n_neighbors=n_songs+1)    
            self.rec_song_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[1:]
            self.recommend_list = []
            for val in self.rec_song_indices:
                song_idx = self.df_tracks.iloc[val[0]]['id']
                idx = self.df_tracks[self.df_tracks['id'] == song_idx].index
                self.recommend_list.append({'Title':self.df_tracks.iloc[idx]['name'].values[0],'Distance':val[1]})
            df = pd.DataFrame(self.recommend_list,index=range(1,n_songs+1))
            return df
        else:
            return "No songs found. Please check your input"

    def content_recommendation(self):
        #Preprocess the data further before putting it into TF-IDF Vectorizer

        #Combine the various genres under one string
        self.df_tracks['genres_combine'] = self.df_tracks['genres'].apply(lambda x: ' '.join(x))
        #Lower the string and combine the name for album and artists
        self.df_tracks['artist_name'] = self.df_tracks['artist_name'].apply(lambda x: x.lower())
        self.df_tracks['artist_name'] = self.df_tracks['artist_name'].apply(lambda x: ''.join(x.split()))
        self.df_tracks['album_name'] = self.df_tracks['album_name'].apply(lambda x: x.lower())
        self.df_tracks['album_name'] = self.df_tracks['album_name'].apply(lambda x: ''.join(x.split()))
        #Combine the required information under one column
        self.df_tracks['overview'] = self.df_tracks['genres_combine'] + ' ' + self.df_tracks['artist_name'] + ' ' + self.df_tracks['album_name']
        self.df_content_recommendation = self.df_tracks[['name','overview']]

        #Create and categorize the words using TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(analyzer='word',ngram_range=(1,2),min_df=0.003, max_df=0.5, stop_words='english', max_features=5000)
        track_matrix = self.vectorizer.fit_transform(self.df_tracks['overview'])
        #Perform further analysis using cosine similarity to calculate the distance and similar value between songs
        self.cos_sim = cosine_similarity(track_matrix, track_matrix)
        #Generate a new DataFrame to obtain the track title for recommendation later
        self.track_title = self.df_tracks['name']

    #Get the recommendation from some of the songs using content recommendation
    def recommend_songs_content(self,song_name,n_songs):
        if len(song_name) > 0:
            songs_idx = self.df_tracks.index[self.df_tracks['name'].str.contains(song_name)]
            similarity_scores = list(enumerate(self.cos_sim[songs_idx[0]]))
            similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
            similarity_scores = similarity_scores[:n_songs]
            songs_indices = [song[0] for song in similarity_scores if song[0] != songs_idx[0]]
            songs_title = self.track_title.iloc[songs_indices]
            songs_title = songs_title.reset_index(drop=True)
            songs_scores = pd.Series([song[1] for song in similarity_scores if song[0] != songs_idx[0]])
            frame = {'Title': songs_title, 'Score': songs_scores}
            songs = pd.DataFrame(frame)
            return songs
        else:
            return "No songs found. Please check your input"

    #Create a recommendation by combining content recommendation and collaborative recommendation
    def recommend_songs_hybrid(self, song_name, n, content_weight=2, collaborative_weight=1):
        #Content Recommendation
        songs_idx = self.df_tracks.index[self.df_tracks['name'].str.contains(song_name)]
        similarity_scores = list(enumerate(self.cos_sim[songs_idx[0]]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        songs_indices = [song[0] for song in similarity_scores if song[0] != songs_idx[0]]
        songs_title = self.track_title.iloc[songs_indices]
        songs_title = songs_title.reset_index(drop=True)
        songs_scores = pd.Series([song[1] for song in similarity_scores if song[0] != songs_idx[0]])
        frame = {'Title': songs_title, 'Score': songs_scores}
        self.songs = pd.DataFrame(frame)
    
        #Collaborative Recommendation
        n_songs = len(self.df_tracks)
        songs_list = self.df_tracks[self.df_tracks['name'].str.contains(song_name)]
        songs_id = songs_list.iloc[0]['id']
        songs_id = self.df_tracks[self.df_tracks['id'] == songs_id].index[0]
        distances,indices = self.knn.kneighbors(self.df_tracks_numeric[songs_id],n_neighbors=n_songs)    
        self.rec_song_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[1:]
        self.recommend_list = []
        for val in self.rec_song_indices:
            song_idx = self.df_tracks.iloc[val[0]]['id']
            idx = self.df_tracks[self.df_tracks['id'] == song_idx].index
            self.recommend_list.append({'Title':self.df_tracks.iloc[idx]['name'].values[0],'Distance':val[1]})
        self.df = pd.DataFrame(self.recommend_list,index=range(1,n_songs))
    
        #Merge both the dataframe to calculate the overall distance
        self.df_recommended = pd.merge(self.songs, self.df, how='inner', left_on='Title', right_on='Title')
    
        #Standardize the value of distance as the lower score indicate the better tracks in distance
        self.df_recommended['Distance'] = 1 / self.df_recommended['Distance']
        max_score = self.df_recommended['Distance'].max()
        self.df_recommended['Distance'] = self.df_recommended['Distance'] / max_score
    
        #Calculate the combined score from respective weight of recommendation
        self.df_recommended['Combined_Score'] = self.df_recommended['Score'] * content_weight + self.df_recommended['Distance'] * collaborative_weight
        self.df_recommended = self.df_recommended.sort_values(by='Combined_Score', ascending=False)
        self.df_recommended = self.df_recommended.drop(['Score','Distance'], axis=1)
        return self.df_recommended[:n]
    
    #Create a playlist with a list of songs (maximum 20)
    def create_playlist(self,songs, recommender_type, n_songs):
        if len(songs) > 20:
            return 'Please choose a maximum of 20 songs'

        self.df_new_playlist = pd.DataFrame(columns = ['Title','Score'])
        for song in songs:
            if recommender_type == "Content":
                self.df_song = self.recommend_songs_content(song, n_songs)
            elif recommender_type == "Collaborative":
                self.df_song = self.recommend_songs_collaborative(song, n_songs)
            elif recommender_type == "Hybrid":
                self.df_song = self.recommend_songs_hybrid(song, n_songs)
            self.df_new_playlist = pd.concat([self.df_new_playlist, self.df_song], axis = 0)
    
        #Combine the new playlist full of recommendation with original playlist track
        self.df_new_playlist = pd.merge(self.df_new_playlist, self.df_tracks, how='left', left_on=['Title'], right_on=['name'])
        #Access the Spotify Client ID and URI
        with open('spotify/spotify.yml') as spotify:
            spotify_details = yaml.safe_load(spotify)
        #Define the scope 
        scope = 'playlist-modify-private'

        #Access the Spotify API
        self.sp = spotipy.Spotify(auth_manager = SpotifyOAuth(client_id = spotify_details['client_id'],client_secret = spotify_details['client_secret'],redirect_uri = spotify_details['redirect_url'],scope = scope))
        # Create a new playlist for tracks to add - you may also add these tracks to your source playlist and proceed
        new_playlist = self.sp.user_playlist_create(user = spotify_details['user'], name = "Machine Learning Recommendation System",public = False, collaborative = False, description = "Created with automated own recommender systems")
        #Add the tracks into the playlist
        for id in self.df_new_playlist['id']:
            self.sp.user_playlist_add_tracks(user = spotify_details['user'], playlist_id = new_playlist['id'], tracks = [id])

    def main(self, recommender_type):
        try:
            logging.info("Processing the data")
            self.preprocessing()
            logging.info("Finished processing the data")
        except:
            print("Cannot preprocess the data")
        
        if recommender_type == "Content":
            logging.info("Processing the recommender type")
            self.content_recommendation()
            logging.info("Finished processing the recommender")
        elif recommender_type == "Collaborative":
            logging.info("Processing the recommender type")
            self.collaborative_recommendation()
            logging.info("Finished processing the recommender")
        elif recommender_type == "Hybrid":
            logging.info("Processing the recommender type")
            self.content_recommendation()
            self.collaborative_recommendation()
            logging.info("Finished processing the recommender")
        else:
            print("Please write the correct recommender type")

        try:
            logging.info("Creating the spotify playlist")
            self.create_playlist(self.songs,recommender_type,self.n_songs)
            logging.info("Finished creating the spotify playlist")
        except:
            print("Playlist cannot be created")
       
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--number', '-n', type=int, help='Input the number of songs needed')
    parser.add_argument('--type', '-t', help='Select the type of recommende you want to select')
    parser.add_argument('--songs', '-s', nargs='+', default=['Careless Whisper', 'Sunflower'], help='Select the songs as the benchmark')
    args = parser.parse_args()
    Recommender(args.songs,args.type,args.number)