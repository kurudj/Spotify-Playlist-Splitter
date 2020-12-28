import spotipy
import time
from IPython.core.display import clear_output
from spotipy import SpotifyClientCredentials, util
import pandas as pd
import time
import numpy as np
import csv
import matplotlib.pyplot as plt
import numpy as np
import requests
import json
from xml.dom import minidom
import xml.etree.ElementTree as ET
from datetime import date

def getPlaylistTracksIDs(user, playlist_id):
    ids = []
    playlist = sp.user_playlist(user, playlist_id)
    for item in playlist['tracks']['items']:
        track = item['track']
        ids.append(track['id'])
    return ids

def getTrackFeatures(id):
    meta = sp.track(id)
    features = sp.audio_features(id)
    name = meta['name']
    album = meta['album']['name']
    artist = meta['album']['artists'][0]['name']
    release_date = meta['album']['release_date']
    length = meta['duration_ms']
    popularity = meta['popularity']
    acousticness = features[0]['acousticness']
    danceability = features[0]['danceability']
    energy = features[0]['energy']
    instrumentalness = features[0]['instrumentalness']
    liveness = features[0]['liveness']
    loudness = features[0]['loudness']
    speechiness = features[0]['speechiness']
    tempo = features[0]['tempo']
    time_signature = features[0]['time_signature']
    track = [name, album, artist, acousticness, danceability, energy, instrumentalness, liveness, loudness, speechiness, tempo]
    return track

def distance(a, b): # compute euclidean distance between two points
    return np.linalg.norm(a - b)

def vector(song): # transform array of a song into a numpy array feature vector
    #return np.array((float(song[2]),float(song[3]),float(song[4]),float(song[5]),float(song[6]),float(song[7]),float(song[8]),float(song[9]))) #with loudness
    return np.array((float(song[2]),float(song[3]),float(song[4]),float(song[5]),float(song[6]),float(song[8]),float(song[9])))

def cluster_centers(songs, numClusters): #cluster centers - the first 10 songs of the playlist
    clusterCenters = []
    for i in range(numClusters):
        clusterCenters.append(songs[i+1])
    return clusterCenters

def whichCluster(song, clusterCenters, firstSong): # find the cluster number for which the curr song has the min distance towards
    minDist = distance(vector(song), vector(firstSong))
    index = 0
    clusterNum = 0
    for centroid in clusterCenters:
        if distance(vector(song), vector(centroid))<minDist:
            minDist = distance(vector(song), vector(centroid))
            clusterNum = index
        index+=1
    return clusterNum

def kmeans(songs, numClusters):
    clusterCenters = []
    clusters = []
    finalClusters = []
    clusterCenters = cluster_centers(songs, numClusters)
    for centroid in clusterCenters:
        clusters.append([int(centroid[0])])
    firstSong = songs[0]
    for song in songs:
        clusters[whichCluster(song, clusterCenters, firstSong)].append(int(song[0]))
    for cluster in clusters:
        setCluster = set(cluster)
        finalClusters.append(list(setCluster))
    return finalClusters

def xml(artist, song): # get xml file from Last.Fm and write it into feed.xml
    URL = "http://ws.audioscrobbler.com/2.0/?method=track.getInfo&api_key=082817e1c54aae082664212000a0f5c5&artist=" + artist + "&track=" + song
    response = requests.get(URL)
    with open('feed.xml', 'wb') as file:
        file.write(response.content)

def genreParsing(artist, song): # return the genre from the .xml file
    xml(artist, song)
    tree = ET.parse("feed.xml")
    root = tree.getroot()
    return root.find("./track/toptags/tag/name").text

def databaseParsing(filename): # parsing the database of music genres
    database = []
    with open(filename) as f:
        for line in f:
            database.append(line)
    database = [x.rstrip('\n') for x in database]
    return database

def genresConstruct(songs): # constructing an array of genres present in the playlist
    newSongs = []
    for row in songs:
        songPlusGenre = []
        try:
            genre = genreParsing(row[3], row[1])
            if genre not in genreDatabase:
                continue
            if genre not in genresRecognized:
                genresRecognized.append(genre)
            songPlusGenre.append(row[0])
            songPlusGenre.append(genre)
            newSongs.append(songPlusGenre)
        except:
            continue
    return genresRecognized, newSongs

def arrOfNewPlaylists(genresRecognized):
    newPlaylistArr = []
    playlist = []
    for i in range(len(genresRecognized)):
        playlist.append(genresRecognized[i])
        newPlaylistArr.append(playlist)
        playlist = []
    return newPlaylistArr

def addingSongsToNewPlaylists(newSongs, newPlaylistArr):
    for row in newSongs:
        genre = row[1]
        for playlist in newPlaylistArr:
            if playlist[0]==genre:
                playlist.append(row[0])
    return newPlaylistArr # adding songs to the playlists of the songs' genres

def genreMerging(newPlaylistArr): # complete
    return newPlaylistArr

def shortenPlaylists(newPlaylistArr): # getting rid of playlists that are too short
    finalPlaylistArr = []
    for playlist in newPlaylistArr:
        if len(playlist)>=6:
            finalPlaylistArr.append(playlist)
    return finalPlaylistArr

def songIndexToID(newPlaylistArr, df): # transform the songs' indexes in the playlist array to their spotify IDs
    for playlist in newPlaylistArr:
        for index in range(1, len(playlist)):
            songArtist = indexToSongArtist(playlist[index], df)
            playlist[index] = findSongID(songArtist)
    return newPlaylistArr

def indexToSongArtist(index, df): # fetch song's index and get a tuple from the df to get artist and song name
    rowData = df.iloc[int(index), :]
    return (rowData[0], rowData[2])

def findSongID(songArtist): # fetch song's ID from Spotify API from a tuple of artist and song name
    results = sp.search(q='artist:' + songArtist[1] + ' track:' + songArtist[0], type='track')
    id = 0
    for result in results['tracks']['items']:
        id = result['id']
    return id

def createPlaylist(newPlaylistArr, userID, userSecret, oldPlaylistName):  # builds new playlists
    id = []
    for playlist in newPlaylistArr:
        sp.user_playlist_create(userID, name = playlist[0], public = True, collaborative = False, description=playlist[0] + " playlist from your ''" + oldPlaylistName + "'' playlist.")
        id = str(sp.user_playlists(userID)['items'][0]['uri'])
        print(id)
        sp.playlist_add_items(id, playlist[1:])

genreDatabase = databaseParsing("/Users/kurudj/Desktop/SCHOOL/PROJECTS/SpotifyPlaylistClustering/GenreDatabase.txt")
client_id = '46c1cfc8247e4267b8f5516f9a61affc'
client_secret = '704027695cb84a95b86351d933d63acd'
#client_credentials_manager = SpotifyClientCredentials(client_id, client_secret)
token = util.prompt_for_user_token(
    username='92y2ill9uonu0cqagd25rvztt',
    scope='playlist-modify-public',
    client_id=client_id,
    client_secret=client_secret,
    redirect_uri="http://localhost:8000/callback/"
)
if token:
    sp = spotipy.Spotify(auth=token)
#sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
ids  = getPlaylistTracksIDs('92y2ill9uonu0cqagd25rvztt', '0sh6lRs8uetLG2pnwIZcmW')
tracks = []
for i in range(len(ids)):
  track = getTrackFeatures(ids[i])
  tracks.append(track)
df = pd.DataFrame(tracks, columns = ['name', 'album', 'artist', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo'])
df2 = df
#GENRE API
df.drop(df.iloc[:, 3:15], inplace = True, axis = 1)
df.to_csv("/Users/kurudj/Desktop/SCHOOL/PROJECTS/untitled folder/spotify.csv", sep = ',')
songs = []
index = 0
with open('/Users/kurudj/Desktop/SCHOOL/PROJECTS/untitled folder/spotify.csv','rt')as f:
  data = csv.reader(f)
  for row in data:
      if index==0:
          index+=1
          continue
      songs.append(row)
      index+=1
genresRecognized = []
newSongs = []
genresRecognized, newSongs = genresConstruct(songs)
newPlaylistArr = []
newPlaylistArr = arrOfNewPlaylists(genresRecognized)
newPlaylistArr = addingSongsToNewPlaylists(newSongs, newPlaylistArr)
#genremerging -> complete
newPlaylistArr = shortenPlaylists(newPlaylistArr)
#for i in newPlaylistArr:
#    print(i)
newPlaylistArr = songIndexToID(newPlaylistArr, df2)
#for i in newPlaylistArr:
#    print(i)
createPlaylist(newPlaylistArr, "92y2ill9uonu0cqagd25rvztt", client_secret, "all time favs")














# K-MEANS CLUSTERING
"""
df.drop(df.iloc[:, 1:3], inplace = True, axis = 1)
df.to_csv("/Users/kurudj/Downloads/spotify.csv", sep = ',')
songs = []
index = 0
with open('/Users/kurudj/Downloads/spotify.csv','rt')as f:
  data = csv.reader(f)
  for row in data:
      if index==0:
          index+=1
          continue
      songs.append(row)
      index+=1
print(len)
numClusters = 10
clusters = kmeans(songs, numClusters)
index = 0
for cluster in clusters:
    playlist = []
    for songIndex in cluster:
        playlist.append(songs[songIndex])
    df = pd.DataFrame(playlist)
    df.to_csv("/Users/kurudj/Desktop/SCHOOL/PROJECTS/SpotifyPlaylistClustering/Playlist CSVs/spotify" + str(index) + ".csv", sep = ',')
    index+=1
"""
#8 clusters 9 coordinates
#cluster_centers = [(0, 0, 0, 0.2, 0, 0.1, -20, 0.01, 80), (0.125, 0.125, 0.125, 0.3, 0.2, 0.2, -17, 0.025, 95), (0.25, 0.25, 0.25, 0.4, 0.4, 0.3, -14, 0.04, 105), (0.325, 0.325, 0.325, 0.5, 0.6, 0.4, -11, 0.06, 115),
#                   (0.475,0.475, 0.475, 0.6, 0.8, 0.5, -7, 0.07, 125), (0.6, 0.6, 0.6, 0.7, 1, 0.6, -5, 0.08, 135), (0.725, 0.725, 0.725, 0.8, 1.2, 0.7, -3, 0.09, 145), (0.85, 0.85, 0.85, 0.9, 1.5, 0.8, 0, 0.1, 155)]
