# GRU-Based Roman Urdu Ghazal Generator

A deep learning project that generates Roman Urdu poetry (ghazals) using Gated Recurrent Units (GRU). The system is capable of generating poetry in the style of various famous Urdu poets.

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
git clone https://github.com/yourusername/GRU-Based-Roman-Urdu-Ghazal-Generation.git
cd GRU-Based-Roman-Urdu-Ghazal-Generation
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
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

## Dataset

The model is trained on a curated dataset of Roman Urdu ghazals from various renowned poets including:
- Mirza Ghalib
- Faiz Ahmad Faiz
- Ahmad Faraz
- And others

## Visualization Features

- Visual Poetry Layout
- Word Frequency Analysis
- Word Cloud Generation
- Interactive Poetry Display

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to all the poets whose works were used in training
- Special thanks to contributors and dataset providers

## Contact

For queries and suggestions, please open an issue on GitHub.

---
