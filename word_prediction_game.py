import streamlit as st
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

class WordPredictionGame:
    def __init__(self):
        # Initialize Mistral client and embedding model
        self.mistral_api_key = st.secrets.get("MISTRAL_API_KEY")
        self.mistral_client = MistralClient(api_key=self.mistral_api_key)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Game state variables
        self.reset_game()

    def reset_game(self):
        # Generate initial sentence from Mistral
        initial_response = self.mistral_client.chat(
            model="mistral-small-latest",
            messages=[
                ChatMessage(role="user", content="Generate a random sentence that is interesting but not too complex.")
            ]
        )
        self.initial_sentence = initial_response.choices[0].message.content.split()
        self.current_sentence = self.initial_sentence.copy()
        self.user_predictions = []
        self.llm_predictions = []
        self.cumulative_distance = 0
        self.game_over = False

    def get_word_embedding(self, word):
        return self.embedding_model.encode([word])[0]

    def calculate_distance(self, user_word, llm_word):
        user_embedding = self.get_word_embedding(user_word)
        llm_embedding = self.get_word_embedding(llm_word)
        return np.linalg.norm(user_embedding - llm_embedding)

    def get_llm_prediction(self, context, temperature):
        prompt = f"Given the context '{' '.join(context)}', predict the next most likely word. Return ONLY the word."
        
        response = self.mistral_client.chat(
            model="mistral-small-latest",
            messages=[
                ChatMessage(role="user", content=prompt)
            ],
            temperature=temperature
        )
        
        return response.choices[0].message.content.strip()

    def play_round(self, user_word, temperature):
        # Check if game is already over
        if self.game_over:
            st.warning("Game is already over. Please start a new game.")
            return

        # Validate user input
        if not user_word or not user_word.strip():
            st.error("Please enter a valid word.")
            return

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
    
    # Initialize game if not already initialized
    if 'game' not in st.session_state:
        st.session_state.game = WordPredictionGame()

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
                distance, llm_word = game.play_round(user_word, temperature)
                st.write(f"LLM's word: {llm_word}")
                st.write(f"Distance between embeddings: {distance:.4f}")
                st.write(f"Cumulative Distance: {game.cumulative_distance:.4f}")

    with col2:
        if st.button("New Game"):
            game.reset_game()
            st.experimental_rerun()

    # Prediction History
    st.header("Prediction History")
    history_df = pd.DataFrame({
        'User Predictions': game.user_predictions,
        'LLM Predictions': game.llm_predictions
    })
    st.dataframe(history_df)

if __name__ == "__main__":
    main()
