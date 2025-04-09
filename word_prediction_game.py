import streamlit as st
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import requests

class WordPredictionGame:
    def __init__(self):
        # Initialize Mistral API key
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

        # Game state variables
        self.reset_game()

    def reset_game(self):
        try:
            # Generate initial sentence from Mistral using direct API call
            url = "https://api.mistral.ai/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.mistral_api_key}"
            }
            payload = {
                "model": "mistral-small-latest",  # Use the latest version
                "messages": [
                    {"role": "user", "content": "Generate a random sentence that is interesting but not too complex."}
                ]
            }

            response = requests.post(url, json=payload, headers=headers)

            if response.status_code == 200:
                result = response.json()
                self.initial_sentence = result["choices"][0]["message"]["content"].strip().split()
                self.current_sentence = self.initial_sentence.copy()
            else:
                st.error(f"Error generating initial sentence: API returned status code {response.status_code}")
                # Fallback to a default sentence
                default_sentence = "The curious cat watched the birds outside."
                st.warning(f"Using default sentence: {default_sentence}")
                self.initial_sentence = default_sentence.split()
                self.current_sentence = self.initial_sentence.copy()

        except Exception as e:
            st.error(f"Error generating initial sentence: {str(e)}")
            # Fallback to a default sentence
            default_sentence = "The curious cat watched the birds outside."
            st.warning(f"Using default sentence: {default_sentence}")
            self.initial_sentence = default_sentence.split()
            self.current_sentence = self.initial_sentence.copy()

        self.user_predictions = []
        self.llm_predictions = []
        self.cumulative_distance = 0
        self.game_over = False

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
                return embedding / (np.linalg.norm(embedding) + 1e-8)  # Normalize
        else:
            # Use the simple character-based embedding if the model failed to load
            embedding = np.zeros(50)
            for i, char in enumerate(word.lower()):
                pos = ord(char) - ord('a')
                if 0 <= pos < 26:
                    embedding[pos % len(embedding)] += 1
            return embedding / (np.linalg.norm(embedding) + 1e-8)  # Normalize

    def calculate_distance(self, user_word, llm_word):
        user_embedding = self.get_word_embedding(user_word)
        llm_embedding = self.get_word_embedding(llm_word)
        return np.linalg.norm(user_embedding - llm_embedding)

    def get_llm_prediction(self, context, temperature):
        prompt = f"Given the context '{' '.join(context)}', predict the next most likely word. Return ONLY the word."

        try:
            # Make direct API call to Mistral
            url = "https://api.mistral.ai/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.mistral_api_key}"
            }
            payload = {
                "model": "mistral-small-latest",  # Use the latest version
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
                # Return a fallback word
                return "the"

        except Exception as e:
            st.error(f"Error getting LLM prediction: {str(e)}")
            # Return a fallback word
            return "the"

    def play_round(self, user_word, temperature):
        # Check if game is already over
        if self.game_over:
            st.warning("Game is already over. Please start a new game.")
            return None, None

        # Validate user input
        if not user_word or not user_word.strip():
            st.error("Please enter a valid word.")
            return None, None

        # Get LLM prediction
        llm_word = self.get_llm_prediction(self.current_sentence, temperature)

        # Calculate distance
        distance = self.calculate_distance(user_word, llm_word)
        self.cumulative_distance += distance

        # Update game state
        self.current_sentence.append(user_word)
        self.current_sentence.append(llm_word)
        self.user_predictions.append(user_word)
        self.llm_predictions.append(llm_word)

        # Check for game end (full stop)
        if '.' in llm_word or '.' in user_word:
            self.game_over = True
            st.success("Game Over!")

        return distance, llm_word

def main():
    st.title("Word Prediction Challenge 🎲")
    st.write("Play a word prediction game with an AI!")

    # Option to choose initial sentence source
    initial_sentence_option = st.radio(
        "Choose the source for the initial sentence:",
        ("Generate a random sentence (AI)", "Enter your own sentence"),
        index=0,  # Default to AI generation
        help="Select whether the AI should generate the first sentence or if you'd like to provide it."
    )

    user_provided_sentence = ""
    if initial_sentence_option == "Enter your own sentence":
        user_provided_sentence = st.text_input("Enter your starting sentence:",
                                               placeholder="e.g., The old house stood on a hill.",
                                               key="user_initial_sentence")

    # Check if Mistral API key is set
    if not st.secrets.get("MISTRAL_API_KEY"):
        st.error("Mistral API key not found. Please add it to your Streamlit secrets.")
        st.write("To set up your API key locally:")
        st.code("""
        # In .streamlit/secrets.toml
        MISTRAL_API_KEY = "your_api_key_here"
        """)
        st.stop()

    # Initialize game if not already initialized
    if 'game' not in st.session_state or st.session_state.get('initial_sentence_option') != initial_sentence_option or (initial_sentence_option == "Enter your own sentence" and st.session_state.get('user_provided_sentence') != user_provided_sentence):
        with st.spinner("Initializing game..."):
            try:
                st.session_state.game = WordPredictionGame(initial_sentence_option, user_provided_sentence)
                st.session_state['initial_sentence_option'] = initial_sentence_option
                st.session_state['user_provided_sentence'] = user_provided_sentence
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
    st.write(" ".join(game.current_sentence))

    # User input
    user_word = st.text_input("Enter your predicted word:")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Submit Word"):
            if user_word:
                with st.spinner("Getting AI prediction..."):
                    result = game.play_round(user_word, temperature)
                    if result:
                        distance, llm_word = result
                        st.write(f"LLM's word: {llm_word}")
                        st.write(f"Distance between embeddings: {distance:.4f}")
                        st.write(f"Cumulative Distance: {game.cumulative_distance:.4f}")

    with col2:
        if st.button("New Game"):
            with st.spinner("Starting new game..."):
                game.reset_game()
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
    st.subheader("Regarding Repetitive Predictions:")
    st.info("The language model sometimes repeats words, especially at lower creativity levels or when it lacks strong contextual cues. Increasing the 'LLM Creativity' slider might introduce more varied predictions. However, very high creativity can lead to less coherent sentences. It's a balance!")

# Modified WordPredictionGame class to accept the initial sentence option
class WordPredictionGame:
    def __init__(self, initial_sentence_option, user_provided_sentence):
        self.mistral_api_key = st.secrets.get("MISTRAL_API_KEY", "")
        if not self.mistral_api_key:
            st.error("Mistral API key not found. Please set it in the Streamlit secrets.")
            st.stop()

        self.initial_sentence_option = initial_sentence_option
        self.user_provided_sentence = user_provided_sentence
        self.reset_game()

    def reset_game(self):
        if self.initial_sentence_option == "Enter your own sentence" and self.user_provided_sentence:
            self.initial_sentence = self.user_provided_sentence.strip().split()
            self.current_sentence = self.initial_sentence.copy()
        else:
            try:
                # Generate initial sentence from Mistral using direct API call
                url = "https://api.mistral.ai/v1/chat/completions"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.mistral_api_key}"
                }
                payload = {
                    "model": "mistral-small-latest",  # Use the latest version
                    "messages": [
                        {"role": "user", "content": "Generate a random sentence that is interesting but not too complex."}
                    ]
                }

                response = requests.post(url, json=payload, headers=headers)

                if response.status_code == 200:
                    result = response.json()
                    self.initial_sentence = result["choices"][0]["message"]["content"].strip().split()
                    self.current_sentence = self.initial_sentence.copy()
                else:
                    st.error(f"Error generating initial sentence: API returned status code {response.status_code}")
                    # Fallback to a default sentence
                    default_sentence = "The curious cat watched the birds outside."
                    st.warning(f"Using default sentence: {default_sentence}")
                    self.initial_sentence = default_sentence.split()
                    self.current_sentence = self.initial_sentence.copy()

            except Exception as e:
                st.error(f"Error generating initial sentence: {str(e)}")
                # Fallback to a default sentence
                default_sentence = "The curious cat watched the birds outside."
                st.warning(f"Using default sentence: {default_sentence}")
                self.initial_sentence = default_sentence.split()
                self.current_sentence = self.initial_sentence.copy()

        self.user_predictions = []
        self.llm_predictions = []
        self.cumulative_distance = 0
        self.game_over = False

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
                return embedding / (np.linalg.norm(embedding) + 1e-8)  # Normalize
        else:
            # Use the simple character-based embedding if the model failed to load
            embedding = np.zeros(50)
            for i, char in enumerate(word.lower()):
                pos = ord(char) - ord('a')
                if 0 <= pos < 26:
                    embedding[pos % len(embedding)] += 1
            return embedding / (np.linalg.norm(embedding) + 1e-8)  # Normalize

    def calculate_distance(self, user_word, llm_word):
        user_embedding = self.get_word_embedding(user_word)
        llm_embedding = self.get_word_embedding(llm_word)
        return np.linalg.norm(user_embedding - llm_embedding)

    def get_llm_prediction(self, context, temperature):
        prompt = f"Given the context '{' '.join(context)}', predict the next most likely word. Return ONLY the word."

        try:
            # Make direct API call to Mistral
            url = "https://api.mistral.ai/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.mistral_api_key}"
            }
            payload = {
                "model": "mistral-small-latest",  # Use the latest version
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
                # Return a fallback word
                return "the"

        except Exception as e:
            st.error(f"Error getting LLM prediction: {str(e)}")
            # Return a fallback word
            return "the"

    def play_round(self, user_word, temperature):
        # Check if game is already over
        if self.game_over:
            st.warning("Game is already over. Please start a new game.")
            return None, None

        # Validate user input
        if not user_word or not user_word.strip():
            st.error("Please enter a valid word.")
            return None, None

        # Get LLM prediction
        llm_word = self.get_llm_prediction(self.current_sentence, temperature)

        # Calculate distance
        distance = self.calculate_distance(user_word, llm_word)
        self.cumulative_distance += distance

        # Update game state
        self.current_sentence.append(user_word)
        self.current_sentence.append(llm_word)
        self.user_predictions.append(user_word)
        self.llm_predictions.append(llm_word)

        # Check for game end (full stop)
        if '.' in llm_word or '.' in user_word:
            self.game_over = True
            st.success("Game Over!")

        return distance, llm_word

if __name__ == "__main__":
    main()
