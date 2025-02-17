import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Roman Urdu Poetry Generator",
    page_icon="📝",
    layout="wide"
)

# Model definition
class GRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embed = self.embedding(x)
        output, hidden = self.gru(embed, hidden)
        output = self.dropout(output)
        prediction = self.fc(output)
        return prediction, hidden

# Utility functions
def create_word_vocab(texts, poet_names):
    all_words = []
    for text in texts:
        words = []
        for line in text.split('\n'):
            words.extend(line.split())
            words.append('\n')
        all_words.extend(words[:-1])

    for poet in poet_names:
        all_words.extend(poet.split())

    unique_words = sorted(list(set(all_words)))
    special_tokens = ['<pad>', '<unk>', '<poet>', '</poet>']
    vocab = special_tokens + unique_words

    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}

    return word2idx, idx2word, len(vocab)

@st.cache_resource
def load_model_and_vocab():
    # Load the dataset to recreate vocabulary
    df = pd.read_csv('filtered_ghazals.csv')
    poetry_texts = df['Poetry Text'].tolist()
    poets = df['Poet'].unique().tolist()
    word2idx, idx2word, vocab_size = create_word_vocab(poetry_texts, poets)

    # Load the model configuration
    checkpoint = torch.load('poetry_generator.pth', map_location=torch.device('cpu'))
    config = checkpoint['config']

    # Initialize the model
    model = GRU(
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )

    # Load the trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, word2idx, idx2word, poets

def generate_poetry(model, word2idx, idx2word, poet_name, seed_text=None, max_length=100, temperature=0.8, repetition_penalty=1.2):
    # ... existing function code from final.py ...
    model.eval()

    # Start with poet context
    context = f"<poet> {poet_name} </poet>"
    poet_context_length = len(context.split())

    if seed_text:
        context += " " + seed_text

    # Convert to tokens
    tokens = [word2idx.get(word, word2idx['<unk>']) for word in context.split()]
    initial_input = torch.LongTensor(tokens).unsqueeze(0)

    generated = tokens
    with torch.no_grad():
        hidden = None
        curr_input = initial_input

        for _ in range(max_length):
            output, hidden = model(curr_input, hidden)
            next_token_logits = output[0, -1, :] / temperature

            # Apply repetition penalty
            for idx in set(generated):
                next_token_logits[idx] /= repetition_penalty

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if next_token.item() == word2idx['<pad>']:
                break

            generated.append(next_token.item())
            curr_input = next_token.view(1, 1)

    # Convert indices back to words
    result = []
    for idx in generated[poet_context_length:]:
        word = idx2word[idx]
        if word == '\n':
            result.append('\n')
        elif word not in ['<poet>', '</poet>']:
            result.append(word)

    generated_text = ' '.join(result).replace(' \n', '\n')
    return generated_text.strip()

def create_word_frequency_plot(text):
    words = text.split()
    word_freq = Counter(words)
    most_common = dict(word_freq.most_common(10))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=list(most_common.values()), y=list(most_common.keys()))
    plt.title("Most Common Words")
    return fig

def create_word_cloud(text):
    wordcloud = WordCloud(
        width=800, height=400,
        background_color='white',
        min_font_size=10
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

def create_visual_poetry(text):
    lines = text.split('\n')
    cols = st.columns(len(lines))
    colors = sns.color_palette("husl", len(lines)).as_hex()
    
    for i, (line, col) in enumerate(zip(lines, cols)):
        col.markdown(
            f"<div style='padding: 10px; border-radius: 5px; background-color: {colors[i]}20;'>"
            f"{line}</div>", 
            unsafe_allow_html=True
        )

def main():
    st.title("Urdu Poetry Generator")
    st.write("Generate Urdu poetry in the style of famous poets!")

    # Load model and vocabulary
    with st.spinner("Loading model... Please wait."):
        model, word2idx, idx2word, poets = load_model_and_vocab()

    # Sidebar controls
    st.sidebar.header("Generation Settings")
    
    # Filter out "Poet" and replace "Unknown" with "Generic" in display list
    display_poets = [p if p != "Unknown" else "Generic" for p in poets if p != "Poet"]
    
    # Select poet
    selected_display_poet = st.sidebar.selectbox("Select Poet", display_poets)
    
    # Map "Generic" back to "Unknown" for model
    selected_poet = "Unknown" if selected_display_poet == "Generic" else selected_display_poet
    
    # Input seed text
    seed_text = st.sidebar.text_input("Seed Text (Optional)", value="dil")
    
    # Replace radio button with slider for creativity
    creativity_levels = {
        0: {"name": "Conservative", "temp": 0.3},
        1: {"name": "Balanced", "temp": 0.7},
        2: {"name": "Creative", "temp": 1.0},
        3: {"name": "Experimental", "temp": 1.2}
    }
    
    creativity_index = st.sidebar.slider(
        "Creativity Level",
        min_value=0,
        max_value=3,
        value=1,
        help="Controls how creative/varied the generated poetry will be"
    )
    
    # Display selected creativity level name
    st.sidebar.write(f"Selected: {creativity_levels[creativity_index]['name']}")
    
    # Map selected creativity to temperature
    temperature = creativity_levels[creativity_index]['temp']
    
    max_length = st.sidebar.slider("Maximum Length", 50, 200, 100, 10,
                                 help="Maximum number of words to generate")

    # Generate button
    if st.sidebar.button("Generate Poetry"):
        with st.spinner("Generating poetry..."):
            generated_text = generate_poetry(
                model=model,
                word2idx=word2idx,
                idx2word=idx2word,
                poet_name=selected_poet,
                seed_text=seed_text,
                temperature=temperature,  # Using mapped temperature value
                repetition_penalty=1.2,
                max_length=max_length
            )
            
            # Create tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["Poetry", "Visual Layout", "Word Frequency", "Word Cloud"])
            
            with tab1:
                st.subheader(f"Generated Poetry in the style of {selected_display_poet}")
                st.markdown(f"```urdu\n{generated_text}\n```")
                
                st.download_button(
                    label="Download Poetry",
                    data=generated_text,
                    file_name="generated_poetry.txt",
                    mime="text/plain"
                )
            
            with tab2:
                st.subheader("Visual Poetry Layout")
                create_visual_poetry(generated_text)
            
            with tab3:
                st.subheader("Word Frequency Analysis")
                freq_fig = create_word_frequency_plot(generated_text)
                st.pyplot(freq_fig)
            
            with tab4:
                st.subheader("Word Cloud Visualization")
                cloud_fig = create_word_cloud(generated_text)
                st.pyplot(cloud_fig)

if __name__ == "__main__":
    main()
