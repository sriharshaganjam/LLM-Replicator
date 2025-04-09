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
        user_preds = ["You"] + game.user_predictions
        llm_preds = ["AI"] + game.llm_predictions

        # Ensure both lists have the same length for the table
        max_len = max(len(user_preds), len(llm_preds))
        user_preds.extend([""] * (max_len - len(user_preds)))
        llm_preds.extend([""] * (max_len - len(llm_preds)))

        history_data = {"": list(range(max_len)), "Your Prediction": user_preds, "AI Prediction": llm_preds}
        history_df = pd.DataFrame(history_data).set_index("")
        st.table(history_df)

    # Addressing LLM Repetition
    st.subheader("Regarding Repetitive Predictions:")
    st.info("The language model sometimes repeats words, especially at lower creativity levels or when it lacks strong contextual cues. Increasing the 'LLM Creativity' slider might introduce more varied predictions. However, very high creativity can lead to less coherent sentences. It's a balance!")

# (The WordPredictionGame class remains the same as the previous version)

if __name__ == "__main__":
    main()
