import streamlit as st
import pandas as pd
from numpy import load
from scipy.sparse import load_npz
from content_based_filtering import content_recommendation
from hybrid_recommendations import HybridRecommenderSystem
from collaborative_filtering import collaborative_recommendation  # your function

# ------------------------
# Load all data
# ------------------------
songs_data = pd.read_csv("data/cleaned_data.csv")
filtered_data = pd.read_csv("data/collab_filtered_data.csv")
transformed_data = load_npz("data/transformed_data.npz")
transformed_hybrid_data = load_npz("data/transformed_hybrid_data.npz")
interaction_matrix = load_npz("data/interaction_matrix.npz")
track_ids = load("data/track_ids.npy", allow_pickle=True)

# ------------------------
# Streamlit UI
# ------------------------
st.title("Welcome to the Spotify Song Recommender!")
st.write("### Enter a song and choose a recommendation type 🎵🎧")

# User inputs
song_name = st.text_input("Enter a song name:")
artist_name = st.text_input("Enter the artist name:")
k = st.selectbox("How many recommendations do you want?", [5, 10, 15, 20], index=1)

# Recommendation type
rec_type = st.selectbox("Select Recommendation Type:",
                        ["Content-Based", "Collaborative", "Hybrid"])

# Diversity slider only for Hybrid
if rec_type == "Hybrid":
    diversity = st.slider("Diversity in Recommendations (Hybrid)",
                          min_value=1, max_value=9, value=5, step=1)

# ------------------------
# Handle button click
# ------------------------
if st.button("Get Recommendations"):
    if not song_name or not artist_name:
        st.warning("Please enter both a song name and artist name.")
    else:
        song_name_lower = song_name.lower()
        artist_name_lower = artist_name.lower()

        # Existence checks
        song_exists = ((songs_data["name"].str.lower() == song_name_lower) &
                       (songs_data["artist"].str.lower() == artist_name_lower)).any()
        filtered_song_exists = ((filtered_data["name"].str.lower() == song_name_lower) &
                                (filtered_data["artist"].str.lower() == artist_name_lower)).any()

        recommendations = None

        # ------------------------
        # Content-Based
        # ------------------------
        if rec_type == "Content-Based":
            if song_exists:
                st.write(f"Recommendations for **{song_name.title()}** by **{artist_name.title()}** (Content-Based)")
                recommendations = content_recommendation(song_name=song_name_lower,
                                                         artist_name=artist_name_lower,
                                                         songs_data=songs_data,
                                                         transformed_data=transformed_data,
                                                         k=k)
            else:
                st.warning(f"Song not found for Content-Based Filtering.")

        # ------------------------
        # Collaborative Filtering
        # ------------------------
        elif rec_type == "Collaborative":
            if filtered_song_exists:
                st.write(f"Recommendations for **{song_name.title()}** by **{artist_name.title()}** (Collaborative)")
                recommendations = collaborative_recommendation(song_name=song_name_lower,
                                                                artist_name=artist_name_lower,
                                                                track_ids=track_ids,
                                                                songs_data=filtered_data,
                                                                interaction_matrix=interaction_matrix,
                                                                k=k)
            else:
                st.warning(f"Song not found for Collaborative Filtering.")

        # ------------------------
        # Hybrid
        # ------------------------
        elif rec_type == "Hybrid":
            if filtered_song_exists:
                st.write(f"Recommendations for **{song_name.title()}** by **{artist_name.title()}** (Hybrid)")
                weight_content_based = 1 - (diversity / 10)
                hybrid_recommender = HybridRecommenderSystem(number_of_recommendations=k,
                                                             weight_content_based=weight_content_based)
                recommendations = hybrid_recommender.give_recommendations(song_name=song_name_lower,
                                                                          artist_name=artist_name_lower,
                                                                          songs_data=filtered_data,
                                                                          transformed_matrix=transformed_hybrid_data,
                                                                          track_ids=track_ids,
                                                                          interaction_matrix=interaction_matrix)
            else:
                st.warning(f"Song not found for Hybrid Filtering.")

        # ------------------------
        # Display recommendations
        # ------------------------
        if recommendations is not None and not recommendations.empty:
            for ind, rec in recommendations.iterrows():
                rec_song = rec["name"].title()
                rec_artist = rec["artist"].title()
                spotify_url = rec.get("spotify_preview_url", None)

                if ind == 0:
                    st.markdown("## Currently Playing")
                elif ind == 1:
                    st.markdown("### Next Up 🎵")
                else:
                    st.markdown(f"#### {ind}. {rec_song} by {rec_artist}")

                st.markdown(f"**{rec_song}** by **{rec_artist}**")
                if spotify_url:
                    st.audio(spotify_url)
                st.write("---")

    