# GRU-Based Roman Urdu Ghazal Generator

A deep learning project that generates Roman Urdu poetry (ghazals) using Gated Recurrent Units (GRU). The system is capable of generating poetry in the style of various famous Urdu poets.

## Live Demo
Try the live application here: [Urdu Ghazal Generator](https://urdughazalgenerator.streamlit.app/)

## Blog
Read about the technical details and implementation in this blog post: [Roman Urdu Poetry Text Generation using GRU based on Poet style](https://medium.com/@basil451287/roman-Urdu-poetry-text-generation-using-GRU-based-on-poet-style-23a5b4fbfccb)

## Features

- Generates Roman Urdu poetry using GRU neural network
- Interactive web interface built with Streamlit
- Multiple creativity levels for poetry generation
- Support for different famous poets' styles
- Visual analytics including word clouds and frequency analysis
- Optional seed text input for guided generation
- Downloadable generated poetry

## Tech Stack

- Python 3.x
- PyTorch
- Streamlit
- Pandas
- Matplotlib
- Seaborn
- WordCloud

## Installation

1. Clone the repository:
```bash
git clone https://github.com/saadsohail05/GRU-Based-Roman-Urdu-Ghazal-Generation.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

2. Access the web interface through your browser (typically http://localhost:8501)

3. Select your desired settings:
   - Choose a poet
   - Set creativity level
   - Enter seed text (optional)
   - Adjust maximum length
   
4. Click "Generate Poetry" to create new ghazals

## Model Architecture

The poetry generator uses a GRU (Gated Recurrent Unit) neural network with:
- Word embeddings
- Multiple GRU layers
- Dropout for regularization
- Linear layer for output
- Poet-specific conditioning using special tokens

### Conditioning Mechanism

The model implements style-based generation through a conditioning mechanism:
- Each input sequence is prefixed with poet-specific tokens (`<poet> poet_name </poet>`)
- This conditioning allows the model to learn and generate poetry in the distinctive style of each poet
- The mechanism ensures stylistic consistency while maintaining the poet's unique characteristics
- Style transfer is possible by changing the poet token while keeping the same seed text

## Data Collection
The poetry data was collected through web scraping from [Rekhta.org](https://rekhta.org), a comprehensive online repository of Urdu poetry. The scraping process and data preprocessing pipeline will be detailed soon in a separate documentation.

**Coming Soon:**
- Detailed scraping methodology
- Data cleaning process
- Dataset statistics
- Code for the scraping pipeline

## Dataset

The model is trained on a curated dataset of Roman Urdu ghazals from various renowned poets including:
- Mirza Ghalib
- Dagh Dehlvi
- Shakeel Badayuni
- Ahmad Faraz
- Hasrat Mohani
- Faiz Ahmad Faiz
- Jaun Eliya
- Bashir Badr
- Shahryar
- Ahmad Mushtaq
- Generic

## Visualization Features

- Visual Poetry Layout
- Word Frequency Analysis
- Word Cloud Generation
- Interactive Poetry Display

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
