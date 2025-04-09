import streamlit as st
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import requests
import random  # For more diverse sentence generation

class WordPredictionGame:
    def __init__(self, sentence_length):
        self.mistral_api_key = st.secrets.get("MISTRAL_API_KEY", "")
        if not self.mistral_api_key:
            st.error("Mistral API key not found. Please set it in the Streamlit secrets.")
            st.stop()

        # Initialize embedding model (check if it's already in session state)
        if 'embedding_model' not in st.session_state:
            try:
                st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                st.info("Sentence Transformer model loaded.")
            except Exception as e:
                st.error(f"Error loading Sentence Transformer model: {str(e)}")
                st.warning("Falling back to a very basic character-based embedding. Semantic similarity will not be captured.")
                st.session_state.embedding_model = None

        self.embedding_model = st.session_state.embedding_model
        self.sentence_length = sentence_length
        self.reset_game()

    def reset_game(self):
        self.initial_sentence = self._generate_initial_sentence(self.sentence_length)
        self.current_sentence_display = self.initial_sentence[:self._get_llm_starts(self.sentence_length)].copy()
        self.user_predictions = []
        self.llm_predictions = []
        self.cumulative_distance = 0
        self.game_over = False

    def _get_llm_starts(self, length):
        """Determines how many initial words the LLM provides based on sentence length."""
        if length == 7:
            return 3
        elif length == 10:
            return 4
        elif length == 13:
            return 5
        return 3  # Default

    def _generate_initial_sentence(self, length):
        """Generates a random initial sentence of the specified length using Mistral."""
        num_initial_words = self._get_llm_starts(length)
        prompt = f"Generate a random and diverse sentence of exactly {length} words."

        try:
            url = "https://api.mistral.ai/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.mistral_api_key}"
            }
            payload = {
                "model": "mistral-small-latest",
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }

            response = requests.post(url, json=payload, headers=headers)

            if response.status_code == 200:
                result = response.json()
                sentence = result["choices"][0]["message"]["content"].strip().split()
                if len(sentence) == length:
                    return sentence
                else:
                    st.warning(f"Generated sentence has {len(sentence)} words, expected {length}. Trying again.")
                    return self._generate_initial_sentence(length) # Recursive call for correct length
            else:
                st.error(f"Error generating initial sentence: API returned status code {response.status_code}")
                return ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "."][:length] # Fallback

        except Exception as e:
            st.error(f"Error generating initial sentence: {str(e)}")
            return ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "."][:length] # Fallback

    def get_word_embedding(self, word):
        if self.embedding_model:
            try:
                return self.embedding_model.encode(word, convert_to_tensor=True).cpu().numpy()
            except Exception as e:
                st.error(f"Error getting embedding for '{word}': {str(e)}")
                st.warning("Falling back to basic embedding for this word.")
                embedding = np.zeros(50)
                for i, char in enumerate(word.lower()):
                    pos = ord(char) - ord('a')
                    if 0 <= pos < 26:
                        embedding[pos % len(embedding)] += 1
                return embedding / (np.linalg.norm(embedding) + 1e-8)
        else:
            st.info("Using basic character-based embedding.")
            embedding = np.zeros(50)
            for i, char in enumerate(word.lower()):
                pos = ord(char) - ord('a')
                if 0 <= pos < 26:
                    embedding[pos % len(embedding)] += 1
            return embedding / (np.linalg.norm(embedding) + 1e-8)

    def calculate_distance(self, user_word, llm_word):
        if self.embedding_model is None:
            st.error("Embedding model is None in calculate_distance!")
            return 1 if user_word.lower() != llm_word.lower() else 0
        user_embedding = self.get_word_embedding(user_word)
        llm_embedding = self.get_word_embedding(llm_word)
        return np.linalg.norm(user_embedding - llm_embedding)

    def get_llm_prediction(self, context, temperature):
        prompt = f"Given the context '{' '.join(context)}', predict the next most likely word. Return ONLY the word."

        try:
            url = "https://api.mistral.ai/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.mistral_api_key}"
            }
            payload = {
                "model": "mistral-small-latest",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature
            }

            response = requests.post(url, json=payload, headers=headers)

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            else:
                st.error(f"Error getting LLM prediction: API returned status code {response.status_code}")
                return "the"

        except Exception as e:
            st.error(f"Error getting LLM prediction: {str(e)}")
            return "the"

    def play_round(self, user_word, temperature):
        if self.game_over:
            st.warning("Game is already over. Please start a new game.")
            return None, None

        if not user_word or not user_word.strip():
            st.error("Please enter a valid word.")
            return None, None

        llm_starts = self._get_llm_starts(self.sentence_length)
        if len(self.user_predictions) + len(self.llm_predictions) >= (self.sentence_length - llm_starts):
            st.info("Sentence prediction complete!")
            self.game_over = True
            return None, None

        context = self.initial_sentence[:llm_starts + len(self.user_predictions) + len(self.llm_predictions)]
        llm_word = self.get_llm_prediction(context, temperature)
        distance = self.calculate_distance(user_word, llm_word)
        self.cumulative_distance += distance

        self.user_predictions.append(user_word)
        self.llm_predictions.append(llm_word)
        self.current_sentence_display.append(user_word)
        self.current_sentence_display.append(llm_word)

        if len(self.current_sentence_display) == self.sentence_length:
            self.game_over = True
            st.success("Sentence prediction complete!")

        return distance, llm_word

def main():
    st.title("Word Prediction Challenge 🎲")
    st.write("Predict the next words in a sentence!")

    # Sentence length selection slider
    sentence_length = st.sidebar.slider(
        "Sentence Length",
        min_value=7,
        max_value=13,
        value=7,
        step=3,
        help="Choose the desired length of the sentence (7, 10, or 13 words)."
    )

    # Initialize game based on selected sentence length
    if 'game' not in st.session_state or st.session_state.get('sentence_length') != sentence_length:
        with st.spinner("Initializing game..."):
            try:
                st.session_state.game = WordPredictionGame(sentence_length)
                st.session_state['sentence_length'] = sentence_length
            except Exception as e:
                st.error(f"Failed to initialize game: {str(e)}")
                st.stop()

    game = st.session_state.game

    # Game configuration
    st.sidebar.header("Game Settings")
    temperature = st.sidebar.slider("LLM Creativity", 0.0, 1.0, 0.5, 0.1, format="%.1f",
                                    help="Control how creative the AI is with its predictions.",
                                    key="temperature_slider")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Creativity Levels:**")
    st.sidebar.markdown("0.0: **Sane** (More predictable)")
    st.sidebar.markdown("1.0: **Wacky!** (Highly creative)")

    # Display initial context
    st.write("### Current Sentence:")
    st.write(" ".join(game.current_sentence_display))

    # User input
    user_word = st.text_input("Enter your predicted word:")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Submit Word"):
            if user_word and not game.game_over:
                with st.spinner("Getting AI prediction..."):
                    result = game.play_round(user_word, temperature)
                    if result:
                        distance, llm_word = result
                        st.write(f"LLM's word: {llm_word}")
                        st.write(f"Distance between embeddings: {distance:.4f}")
                        st.write(f"Cumulative Distance: {game.cumulative_distance:.4f}")
                        st.experimental_rerun() # Rerun to update the displayed sentence
            elif game.game_over:
                st.info("Game over. Start a new game.")

    with col2:
        if st.button("New Game"):
            with st.spinner("Starting new game..."):
                st.session_state.pop('game', None) # Force re-initialization
                st.experimental_rerun()

    # Prediction History - Horizontal Table
    if game.user_predictions:
        st.header("Prediction History")
        history_data = {
            "Your Prediction": game.user_predictions,
            "AI Prediction": game.llm_predictions,
        }
        history_df = pd.DataFrame(history_data).T
        st.table(history_df)

    # Addressing LLM Repetition
    st.subheader("Regarding LLM Predictions:")
    st.info("The language model's predictions can sometimes be repetitive. Adjusting the 'LLM Creativity' slider might introduce more variety.")

if __name__ == "__main__":
    main()
