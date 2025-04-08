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
            
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
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
    # Very simple fallback embedding (not recommended for production)
    # This creates a basic embedding based on character counts
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
    if 'game' not in st.session_state:
        with st.spinner("Initializing game..."):
            try:
                st.session_state.game = WordPredictionGame()
            except Exception as e:
                st.error(f"Failed to initialize game: {str(e)}")
                st.stop()

    game = st.session_state.game

    # Game configuration
    st.sidebar.header("Game Settings")
    temperature = st.sidebar.slider("LLM Creativity", 0.0, 1.0, 0.5, 0.1)
    
    # Display initial context
    st.write("### Initial Sentence:")
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

    # Prediction History
    if game.user_predictions:
        st.header("Prediction History")
        history_df = pd.DataFrame({
            'User Predictions': game.user_predictions,
            'LLM Predictions': game.llm_predictions
        })
        st.dataframe(history_df)

# Alternative Client Library Implementation
# This is kept for reference but not used in the main code
class MistralClientLibrary:
    def __init__(self, api_key):
        try:
            # Import here to avoid errors if not used
            from mistralai.client import MistralClient
            from mistralai.models.chat_completion import ChatMessage
            
            self.client = MistralClient(api_key=api_key)
            self.ChatMessage = ChatMessage
            self.use_library = True
        except Exception as e:
            st.warning(f"Failed to initialize Mistral client library: {str(e)}")
            st.warning("Falling back to direct API calls.")
            self.use_library = False
    
    def chat(self, model, messages, temperature=0.7):
        if self.use_library:
            try:
                # Convert to ChatMessage format
                chat_messages = [self.ChatMessage(role=msg["role"], content=msg["content"]) for msg in messages]
                return self.client.chat(model=model, messages=chat_messages, temperature=temperature)
            except Exception as e:
                st.warning(f"Error using Mistral client library: {str(e)}. Falling back to direct API calls.")
                self.use_library = False
        
        # Fallback to direct API call
        if not self.use_library:
            url = "https://api.mistral.ai/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature
            }
            
            response = requests.post(url, json=payload, headers=headers)
            
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"API Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    main()
