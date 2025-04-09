import streamlit as st
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import requests
import random
import time

class WordPredictionGame:
    def __init__(self, sentence_length):
        self.mistral_api_key = st.secrets.get("MISTRAL_API_KEY", "")
        if not self.mistral_api_key:
            st.error("Mistral API key not found. Please set it in the Streamlit secrets.")
            st.stop()

        if 'embedding_model' not in st.session_state:
            try:
                st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                st.info("Sentence Transformer model loaded.")
            except Exception as e:
                st.error(f"Error loading embedding model: {e}")
                st.warning("Falling back to basic embedding.")
                st.session_state.embedding_model = None

        self.embedding_model = st.session_state.embedding_model
        self.sentence_length = sentence_length
        self.reset_game()

    def reset_game(self):
        self.initial_sentence = self._generate_initial_sentence(self.sentence_length)
        self.llm_starts = self._get_llm_starts(self.sentence_length)
        self.user_predictions = []
        self.llm_predictions = []
        self.cumulative_distance = 0
        self.game_over = False

    def _get_llm_starts(self, length):
        if length == 7:
            return 3
        elif length == 10:
            return 4
        elif length == 13:
            return 5
        return 3

    def _generate_initial_sentence(self, length):
        prompt = f"Generate a random and diverse sentence of exactly {length} words."
        max_retries = 3
        delay_seconds = 2

        for attempt in range(max_retries):
            try:
                url = "https://api.mistral.ai/v1/chat/completions"
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.mistral_api_key}"}
                payload = {"model": "mistral-small-latest", "messages": [{"role": "user", "content": prompt}]}
                response = requests.post(url, json=payload, headers=headers)
                if response.status_code == 200:
                    sentence = response.json()["choices"][0]["message"]["content"].strip().split()
                    if len(sentence) == length:
                        return sentence
                    else:
                        st.warning(f"Attempt {attempt + 1}: Generated {len(sentence)} words, expected {length}.")
                elif response.status_code == 429:
                    st.error(f"API rate limit hit (attempt {attempt + 1}).")
                    time.sleep(delay_seconds * (attempt + 1))
                else:
                    st.error(f"API error (attempt {attempt + 1}): {response.status_code}")
            except requests.exceptions.RequestException as e:
                st.error(f"Network error (attempt {attempt + 1}): {e}")
                time.sleep(delay_seconds * (attempt + 1))
            if attempt < max_retries - 1:
                time.sleep(delay_seconds)

        st.error(f"Failed to generate a {length}-word sentence after {max_retries} attempts.")
        return ["The", "quick", "brown"][:length]

    def get_word_embedding(self, word):
        if self.embedding_model:
            try:
                return self.embedding_model.encode(word, convert_to_tensor=True).cpu().numpy()
            except Exception as e:
                st.error(f"Error embedding '{word}': {e}")
                return np.zeros(50)
        else:
            return np.zeros(50)

    def calculate_distance(self, user_word, llm_word):
        if self.embedding_model is None:
            return 1 if user_word.lower() != llm_word.lower() else 0
        user_embedding = self.get_word_embedding(user_word)
        llm_embedding = self.get_word_embedding(llm_word)
        return np.linalg.norm(user_embedding - llm_embedding)

    def get_llm_prediction(self, context):
        prompt = f"Given '{' '.join(context)}', predict the next single word."
        try:
            url = "https://api.mistral.ai/v1/chat/completions"
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.mistral_api_key}"}
            payload = {"model": "mistral-small-latest", "messages": [{"role": "user", "content": prompt}]}
            response = requests.post(url, json=payload, headers=headers)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"].strip()
            else:
                st.error(f"LLM prediction API error: {response.status_code}")
                return ""
        except requests.exceptions.RequestException as e:
            st.error(f"LLM prediction network error: {e}")
            return ""

    def play_round(self, user_word):
        if self.game_over:
            st.warning("Game over. Start new game.")
            return None, None

        if not user_word or not user_word.strip():
            st.error("Enter a valid word.")
            return None, None

        if len(self.user_predictions) >= (self.sentence_length - self.llm_starts):
            st.info("Sentence prediction complete!")
            self.game_over = True
            return None, None

        context = self.initial_sentence[:self.llm_starts + len(self.user_predictions) + len(self.llm_predictions)]
        llm_word = self.get_llm_prediction(context)
        if llm_word:
            distance = self.calculate_distance(user_word, llm_word)
            self.cumulative_distance += distance
            self.user_predictions.append(user_word)
            self.llm_predictions.append(llm_word)
            return distance, llm_word
        return None, None

def main():
    st.title("Word Prediction Challenge 🎲")
    st.write("Try to complete the sentence after the AI!")

    sentence_length = st.sidebar.slider("Sentence Length", 7, 13, 7, 3, help="Choose sentence length.")

    if 'game' not in st.session_state or st.session_state.get('sentence_length') != sentence_length:
        with st.spinner("Starting new game..."):
            try:
                st.session_state.game = WordPredictionGame(sentence_length)
                st.session_state['sentence_length'] = sentence_length
            except Exception as e:
                st.error(f"Error initializing game: {e}")
                st.stop()

    game = st.session_state.game
    llm_starts = game.llm_starts
    remaining_words = sentence_length - len(game.user_predictions) - len(game.llm_predictions)

    st.write("### Initial Sentence:")
    st.write(" ".join(game.initial_sentence[:llm_starts] + ["_"] * (sentence_length - llm_starts)))

    prediction_data = {
        "Your Predictions": [" ".join(game.initial_sentence[:llm_starts])] + game.user_predictions + ["_"] * (remaining_words if remaining_words > 0 else 0),
        "AI Predictions": [" ".join(game.initial_sentence[:llm_starts])] + game.llm_predictions + ["_"] * (remaining_words if remaining_words > 0 else 0),
    }

    predictions_df = pd.DataFrame(prediction_data).iloc[[0] + list(range(1, len(prediction_data["Your Predictions"])))].T
    predictions_df.columns = ["Initial Words"] + [f"Prediction {i+1}" for i in range(remaining_words + len(game.user_predictions))]
    st.table(predictions_df)

    st.write(f"Cumulative Embedding Distance: {game.cumulative_distance:.4f}")

    user_word = st.text_input("Your next prediction:")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Predict") and not game.game_over:
            if user_word:
                with st.spinner("Getting AI prediction..."):
                    distance, llm_word = game.play_round(user_word)
                    if distance is not None:
                        st.experimental_rerun()
            elif not user_word and not game.game_over:
                st.warning("Please enter a word.")
        elif game.game_over:
            st.info("Game over. Start a new game.")

    with col2:
        if st.button("New Game"):
            st.session_state.pop('game', None)
            st.experimental_rerun()

if __name__ == "__main__":
    main()
