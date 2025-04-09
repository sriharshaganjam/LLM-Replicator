import streamlit as st
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import requests
import random
import time

class WordPredictionGame:
    def __init__(self):
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
        self.reset_game()

    def reset_game(self):
        self.initial_sentence = self._generate_initial_sentence()
        self.num_words = len(self.initial_sentence)
        self.llm_starts = min(3, self.num_words)  # Display first 3 words
        self.user_predictions = []
        self.llm_predictions = []
        self.cumulative_distance = 0
        self.game_over = False

    def _generate_initial_sentence(self):
        # List of different prompt templates to get more variety
        prompt_templates = [
            "Generate a short interesting sentence about {topic} with around {length} words. The sentence must be grammatically correct and meaningful.",
            "Write a clear, complete sentence about {topic} using around {length} words. Make it compelling and varied.",
            "Create a {length}-word sentence about {topic}. Ensure it's grammatically correct and interesting."
        ]
        
        topics = [
            "nature", "technology", "history", "food", "travel", 
            "science", "art", "sports", "music", "literature",
            "ocean", "mountains", "cities", "animals", "weather"
        ]
        
        # Generate a random sentence length between 5 and 10 words
        length = random.randint(5, 10)
        topic = random.choice(topics)
        prompt = random.choice(prompt_templates).format(topic=topic, length=length)
        
        max_retries = 3
        delay_seconds = 2

        for attempt in range(max_retries):
            try:
                url = "https://api.mistral.ai/v1/chat/completions"
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.mistral_api_key}"}
                payload = {
                    "model": "mistral-small-latest", 
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,  # Add some randomness but not too much
                    "max_tokens": 30
                }
                response = requests.post(url, json=payload, headers=headers)
                if response.status_code == 200:
                    sentence_text = response.json()["choices"][0]["message"]["content"].strip()
                    # Clean up the sentence - remove periods, quotation marks, etc.
                    sentence_text = sentence_text.replace('"', '').replace("'", "").strip(".")
                    sentence = sentence_text.split()
                    
                    # Verify we have a good sentence of appropriate length
                    if 4 <= len(sentence) <= 10 and len(set(sentence)) == len(sentence):  # No repeated words
                        return sentence
                    
                elif response.status_code == 429:
                    st.error(f"API rate limit hit.")
                    time.sleep(delay_seconds * (attempt + 1))
                else:
                    st.error(f"API error: {response.status_code}")
            except requests.exceptions.RequestException as e:
                st.error(f"Network error: {e}")
                time.sleep(delay_seconds * (attempt + 1))
            if attempt < max_retries - 1:
                time.sleep(delay_seconds)

        # Fallback sentences if API fails
        fallback_sentences = [
            "The quick brown fox jumps over".split(),
            "Scientists discovered a new species yesterday".split(),
            "Ancient civilizations built impressive stone monuments".split(),
            "Technology continues to evolve at unprecedented".split(),
            "The mountain climbers reached the summit".split()
        ]
        return random.choice(fallback_sentences)

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
        # Check if context already has words and use a more specific prompt
        if context:
            context_text = ' '.join(context)
            prompt = f"Complete this sentence by adding one word after '{context_text}'. Provide exactly ONE word that would naturally follow in this context. The word should make grammatical sense and help form a coherent sentence."
        else:
            prompt = "Provide a single word to start a sentence."
            
        try:
            url = "https://api.mistral.ai/v1/chat/completions"
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.mistral_api_key}"}
            payload = {
                "model": "mistral-small-latest", 
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.4,  # Lower temperature for more predictable responses
                "max_tokens": 5  # Limit tokens to avoid getting more than one word
            }
            response = requests.post(url, json=payload, headers=headers)
            if response.status_code == 200:
                # Extract just the first word from the response
                prediction = response.json()["choices"][0]["message"]["content"].strip()
                prediction = prediction.split()[0].strip(".,!?;:")  # Get first word and remove punctuation
                
                # Check if the predicted word is already in the context to avoid repetition
                if prediction.lower() in [w.lower() for w in context]:
                    # If there's repetition, ask for an alternative
                    alt_prompt = f"Provide a different single word (not '{prediction}') that could follow '{context_text}' in a sentence."
                    alt_payload = {
                        "model": "mistral-small-latest", 
                        "messages": [{"role": "user", "content": alt_prompt}],
                        "temperature": 0.7,  # Higher temperature for more variety
                        "max_tokens": 5
                    }
                    alt_response = requests.post(url, json=alt_payload, headers=headers)
                    if alt_response.status_code == 200:
                        prediction = alt_response.json()["choices"][0]["message"]["content"].strip().split()[0].strip(".,!?;:")
                
                return prediction
            else:
                st.error(f"LLM prediction API error: {response.status_code}")
                return ""
        except requests.exceptions.RequestException as e:
            st.error(f"LLM prediction network error: {e}")
            return ""
        except IndexError:
            # If there's an issue parsing the response
            return "the"  # Safe fallback

    def play_round(self, user_word):
        if self.game_over:
            st.warning("Game over. Start new game.")
            return None, None

        if not user_word or not user_word.strip():
            st.error("Enter a valid word.")
            return None, None

        if len(self.user_predictions) >= (len(self.initial_sentence) - self.llm_starts):
            self.game_over = True
            st.info("Sentence prediction complete!")
            return None, None

        context = self.initial_sentence[:self.llm_starts] + self.user_predictions
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

    if 'game' not in st.session_state:
        with st.spinner("Starting new game..."):
            try:
                st.session_state.game = WordPredictionGame()
            except Exception as e:
                st.error(f"Error initializing game: {e}")
                st.stop()

    game = st.session_state.game
    llm_starts = game.llm_starts
    num_words = game.num_words
    remaining_words = num_words - llm_starts - len(game.user_predictions)

    st.write(f"The AI has generated a sentence of **{num_words}** words. Guess the remaining **{num_words - llm_starts}** words after the first **{llm_starts}**.")

    st.write("### Initial Words:")
    st.write(" ".join(game.initial_sentence[:llm_starts] + ["_"] * (num_words - llm_starts)))

    user_sentence_display = " ".join(game.initial_sentence[:llm_starts] + game.user_predictions + ["_"] * (num_words - llm_starts - len(game.user_predictions)))
    llm_sentence_display = " ".join(game.initial_sentence[:llm_starts] + game.llm_predictions + ["_"] * (num_words - llm_starts - len(game.llm_predictions)))

    st.write("### Your Sentence:")
    st.write(user_sentence_display)
    st.write("### AI's Sentence:")
    st.write(llm_sentence_display)

    st.write(f"Cumulative Embedding Distance: {game.cumulative_distance:.4f}")

    user_word = st.text_input("Your next word:", max_chars=20, help="Enter one word at a time.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Predict") and not game.game_over:
            if user_word:
                user_words = user_word.strip().split()
                if len(user_words) == 1:
                    with st.spinner("Getting AI prediction..."):
                        distance, llm_word = game.play_round(user_word)
                        if distance is not None:
                            st.experimental_rerun()
                else:
                    st.error("Please enter only one word for your prediction.")
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
