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
        self.word_distances = []  # Store individual word distances
        self.sentence_distance = None  # To store full sentence distance
        self.game_over = False
        # Store the original complete sentence for better LLM prediction
        self.original_complete_sentence = self.initial_sentence.copy()

    def _generate_initial_sentence(self):
        # List of different prompt templates to get more variety
        prompt_templates = [
            "Generate a coherent, complete sentence about {topic} with exactly {length} words. The sentence must be grammatically correct and meaningful. Do not include quotation marks or periods.",
            "Write a clear, complete sentence about {topic} using exactly {length} words. Make it compelling and varied. Avoid special characters, quotes or periods.",
            "Create a straightforward {length}-word sentence about {topic}. Ensure it's grammatically correct and interesting. No quotes or periods."
        ]
        
        topics = [
            "nature", "technology", "history", "food", "travel", 
            "science", "art", "sports", "music", "literature",
            "ocean", "mountains", "cities", "animals", "weather",
            "education", "health", "economy", "architecture", "space"
        ]
        
        # Generate a random sentence length between 5 and 10 words
        length = random.randint(5, 8)
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
                    "temperature": 0.5,
                    "max_tokens": 30
                }
                response = requests.post(url, json=payload, headers=headers)
                if response.status_code == 200:
                    sentence_text = response.json()["choices"][0]["message"]["content"].strip()
                    # Clean up the sentence - remove periods, quotation marks, etc.
                    sentence_text = sentence_text.replace('"', '').replace("'", "").strip(".!?")
                    sentence = sentence_text.split()
                    
                    # Verify we have a good sentence of appropriate length
                    if 4 <= len(sentence) <= 10 and len(set(sentence)) >= len(sentence) * 0.8:  # Allow some repeats like "the", "a"
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

    def get_sentence_embedding(self, sentence):
        if self.embedding_model:
            try:
                return self.embedding_model.encode(sentence, convert_to_tensor=True).cpu().numpy()
            except Exception as e:
                st.error(f"Error embedding sentence: {e}")
                return np.zeros(384)  # Default embedding size
        else:
            return np.zeros(384)

    def calculate_distance(self, user_word, llm_word):
        if self.embedding_model is None:
            return 1 if user_word.lower() != llm_word.lower() else 0
        user_embedding = self.get_word_embedding(user_word)
        llm_embedding = self.get_word_embedding(llm_word)
        return np.linalg.norm(user_embedding - llm_embedding)

    def calculate_sentence_distance(self):
        """Calculate embedding distance between complete sentences"""
        user_sentence = " ".join(self.initial_sentence[:self.llm_starts] + self.user_predictions)
        llm_sentence = " ".join(self.initial_sentence[:self.llm_starts] + self.llm_predictions)
        
        user_embedding = self.get_sentence_embedding(user_sentence)
        llm_embedding = self.get_sentence_embedding(llm_sentence)
        
        return np.linalg.norm(user_embedding - llm_embedding)

    def get_llm_prediction(self, context):
        # Instead of asking the LLM to predict the next word,
        # we'll use a stronger systematic approach to ensure coherence
        
        # Method 1: Use a "cloze" task with the context
        context_text = ' '.join(context)
        position = len(context)
        
        if position < len(self.initial_sentence):
            # We know the actual next word from initial generation
            return self.initial_sentence[position]
        
        # Method 2: Ask LLM for a very specific next word
        prompt = f"""Given this sentence beginning: "{context_text}"

Please predict ONLY the SINGLE NEXT WORD that would make the most grammatical and semantic sense to continue this sentence.
You must return EXACTLY ONE WORD with no punctuation, quotation marks, or explanation.

Next word:"""
            
        try:
            url = "https://api.mistral.ai/v1/chat/completions"
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.mistral_api_key}"}
            payload = {
                "model": "mistral-small-latest", 
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,  # Very low temperature for predictability
                "max_tokens": 3  # Strict limit on tokens
            }
            response = requests.post(url, json=payload, headers=headers)
            
            if response.status_code == 200:
                # Extract just one word, processing carefully
                prediction_text = response.json()["choices"][0]["message"]["content"].strip()
                words = prediction_text.split()
                if words:
                    prediction = words[0].strip(".,!?;:\"'()[]{}").lower()
                    
                    # Check if it's a duplicate of any previous word
                    if prediction.lower() in [w.lower() for w in context[-3:]]:
                        # Try again with explicit instruction to avoid that word
                        avoid_words = [w.lower() for w in context[-3:]]
                        alt_prompt = f"""Given this sentence beginning: "{context_text}"

Please predict the SINGLE NEXT WORD that would make grammatical and semantic sense.
DO NOT use any of these words: {', '.join(avoid_words)}
Return EXACTLY ONE WORD with no punctuation or explanation.

Next word:"""
                        
                        alt_payload = {
                            "model": "mistral-small-latest", 
                            "messages": [{"role": "user", "content": alt_prompt}],
                            "temperature": 0.5,
                            "max_tokens": 3
                        }
                        
                        alt_response = requests.post(url, json=alt_payload, headers=headers)
                        if alt_response.status_code == 200:
                            alt_text = alt_response.json()["choices"][0]["message"]["content"].strip()
                            alt_words = alt_text.split()
                            if alt_words:
                                prediction = alt_words[0].strip(".,!?;:\"'()[]{}").lower()
                    
                    # Make sure we have a real word
                    if len(prediction) >= 2:  # Avoid single letters except maybe "a" or "I"
                        return prediction
                    elif len(prediction) == 1 and prediction in ['a', 'i']:
                        return prediction
                
                # Fallback to basic words that often work
                return self._get_fallback_word(context)
            else:
                st.error(f"LLM prediction API error: {response.status_code}")
                return self._get_fallback_word(context)
        except Exception as e:
            st.error(f"Error in LLM prediction: {e}")
            return self._get_fallback_word(context)

    def _get_fallback_word(self, context):
        # Common words that often work well in most sentences
        fallbacks = ["the", "to", "of", "in", "and", "with", "for", "that", "by", "as"]
        
        # Try to pick one that hasn't been used recently
        for word in fallbacks:
            if word not in [w.lower() for w in context[-3:]]:
                return word
        
        return random.choice(fallbacks)

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
            self.user_predictions.append(user_word)
            self.llm_predictions.append(llm_word)
            self.word_distances.append(distance)
            
            # Calculate sentence distance if this was the last word
            if len(self.user_predictions) >= (len(self.initial_sentence) - self.llm_starts):
                self.sentence_distance = self.calculate_sentence_distance()
                self.game_over = True
            
            return distance, llm_word
        return None, None

def create_embedding_legend():
    """Create a legend explaining embedding distances"""
    st.markdown("### Understanding Embedding Distances")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Word-Level Distance")
        st.markdown("""
        - **0.0 - 0.1**: Words are nearly identical in meaning
        - **0.11 - 0.4**: Words are very similar semantically
        - **0.41 - 0.99**: Words share related concepts or have some semantic relationship
        - **> 1.0**: Words have little or no semantic similarity
        """)
    
    with col2:
        st.markdown("#### Sentence-Level Distance")
        st.markdown("""
        - **0.0 - 0.4**: Sentences convey nearly identical meaning
        - **0.41 - 0.6**: Sentences share common topics/themes
        - **0.61 - 0.90**: Sentences have some topical overlap
        - **> 0.91**: Sentences discuss different topics/concepts
        """)
    
    st.markdown("""
    *Note: Lower distances indicate greater semantic similarity between your predictions and the AI's predictions.*
    """)

def main():
    st.title("Word Prediction Challenge 🎲")
    st.write("Learn to think like an LLM one word at a time, and see if your words match with the LLM's predictions")

    # Initialize session state for tracking input
    if 'input_key' not in st.session_state:
        st.session_state.input_key = 0

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

    st.write(f"Our LLM has generated a sentence of **{num_words}** words. Guess the remaining **{num_words - llm_starts}** words after the first **{llm_starts}**.")

    # Create sentence displays
    initial_sentence_display = " ".join(game.initial_sentence[:llm_starts] + ["_"] * (num_words - llm_starts))
    user_sentence_display = " ".join(game.initial_sentence[:llm_starts] + game.user_predictions + ["_"] * (num_words - llm_starts - len(game.user_predictions)))
    llm_sentence_display = " ".join(game.initial_sentence[:llm_starts] + game.llm_predictions + ["_"] * (num_words - llm_starts - len(game.llm_predictions)))

    # Display Initial Words directly inside blue box
    st.markdown(f"""
    <div style="background-color:#3366cc; padding:10px; border-radius:5px; margin-bottom:10px;">
        <h3 style="margin-top:0; color:#ffffff;">Initial Words:</h3>
        <p style="font-size:16px; color:#ffffff;">
            {initial_sentence_display}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Display Your Sentence directly inside green box
    st.markdown(f"""
    <div style="background-color:#1e7d32; padding:10px; border-radius:5px; margin-bottom:10px;">
        <h3 style="margin-top:0; color:#ffffff;">Your Sentence:</h3>
        <p style="font-size:16px; color:#ffffff;">
            {user_sentence_display}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Display AI's Sentence directly inside purple box
    st.markdown(f"""
    <div style="background-color:#6a1b9a; padding:10px; border-radius:5px; margin-bottom:15px;">
        <h3 style="margin-top:0; color:#ffffff;">AI's Sentence:</h3>
        <p style="font-size:16px; color:#ffffff;">
            {llm_sentence_display}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display sentence-level distance if game is over
    if game.game_over and game.sentence_distance is not None:
        st.markdown(f"""
        <div style="background-color:#c62828; padding:15px; border-radius:5px; margin:15px 0; text-align:center;">
            <h2 style="margin:0; font-size:24px; color:#ffffff;">Sentence Embedding Distance: {game.sentence_distance:.4f}</h2>
        </div>
        """, unsafe_allow_html=True)

    # Use the key to force text_input to reset
    user_word = st.text_input("Your next word:", max_chars=20, key=f"word_input_{st.session_state.input_key}", help="Enter one word at a time.")

    def reset_input():
        # Increment the key to force the text input to reset
        st.session_state.input_key += 1

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Predict") and not game.game_over:
            if user_word:
                user_words = user_word.strip().split()
                if len(user_words) == 1:
                    with st.spinner("Getting AI prediction..."):
                        distance, llm_word = game.play_round(user_word)
                        if distance is not None:
                            # Reset the input field before rerunning
                            reset_input()
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
            reset_input()  # Also reset input when starting a new game
            st.experimental_rerun()
    
    # Display word-level distances table
    if game.word_distances:
        st.markdown("### Word Prediction Results")
        
        # Create table with word-level distances
        data = {
            "Position": list(range(1, len(game.user_predictions) + 1)),
            "Your Word": game.user_predictions,
            "AI's Word": game.llm_predictions,
            "Embedding Distance": [f"{dist:.4f}" for dist in game.word_distances]
        }
        df = pd.DataFrame(data)
        st.table(df)
    
    # Add legend explaining embedding distances
    create_embedding_legend()

if __name__ == "__main__":
    main()
