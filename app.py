import streamlit as st
import torch
import torch.nn as nn
import json
import math
from PIL import Image
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.let_it_rain import rain
from streamlit_extras.colored_header import colored_header

# Load vocabulary
with open("vocabulary.json", "r") as f:
    vocab = json.load(f)

st.set_page_config(page_title="C++ & Pseudocode Translator", page_icon="🌍", layout="wide")
st.sidebar.success("✅ Vocabulary loaded with {} tokens".format(len(vocab)))

# Custom Styling
st.markdown(
    """
    <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            padding: 10px;
            border-radius: 10px;
        }
        .stTextArea textarea {
            font-size: 16px;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Transformer Configuration
class Config:
    vocab_size = 12006  # Adjust based on vocabulary.json
    max_length = 100
    embed_dim = 256
    num_heads = 8
    num_layers = 2
    feedforward_dim = 512
    dropout = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

# Transformer Model
class Seq2SeqTransformer(nn.Module):
    def __init__(self, config):
        super(Seq2SeqTransformer, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.positional_encoding = PositionalEncoding(config.embed_dim, config.max_length)
        self.transformer = nn.Transformer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            num_encoder_layers=config.num_layers,
            num_decoder_layers=config.num_layers,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout
        )
        self.fc_out = nn.Linear(config.embed_dim, config.vocab_size)

    def forward(self, src, tgt):
        src_emb = self.embedding(src) * math.sqrt(config.embed_dim)
        tgt_emb = self.embedding(tgt) * math.sqrt(config.embed_dim)
        src_emb = self.positional_encoding(src_emb)
        tgt_emb = self.positional_encoding(tgt_emb)
        out = self.transformer(src_emb.permute(1, 0, 2), tgt_emb.permute(1, 0, 2))
        out = self.fc_out(out.permute(1, 0, 2))
        return out

# Load Models
@st.cache_resource
def load_model(path):
    model = Seq2SeqTransformer(config).to(config.device)
    model.load_state_dict(torch.load(path, map_location=config.device))
    model.eval()
    return model

cpp_to_pseudo_model = load_model("cpp_to_pseudo_epoch_1.pth")
pseudo_to_cpp_model = load_model("transformer_epoch_1.pth")

st.sidebar.success("✅ Models loaded successfully!")

# Translation Function

def translate(model, input_tokens, vocab, device, max_length=50):
    model.eval()
    input_ids = [vocab.get(token, vocab["<unk>"]) for token in input_tokens]
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    output_ids = [vocab["<start>"]]
    translations = []
    
    for _ in range(max_length):
        output_tensor = torch.tensor(output_ids, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            predictions = model(input_tensor, output_tensor)
        next_token_id = predictions.argmax(dim=-1)[:, -1].item()
        output_ids.append(next_token_id)
        if next_token_id == vocab["<end>"]:
            break
        id_to_token = {idx: token for token, idx in vocab.items()}
        translations.append(id_to_token.get(next_token_id, "<unk>"))
    
    return translations

# Streamlit UI
st.title("🌍 C++ & Pseudocode Translator")
mode = st.radio("🔄 Select Translation Mode", ("C++ → Pseudocode", "Pseudocode → C++"))
user_input = st.text_area("✍️ Enter code:")

if st.button("🚀 Translate"):
    tokens = user_input.strip().split()
    if mode == "C++ → Pseudocode":
        translated_code = translate(cpp_to_pseudo_model, tokens, vocab, config.device)
    else:
        translated_code = translate(pseudo_to_cpp_model, tokens, vocab, config.device)
    
    st.subheader("📜 Line-by-Line Translation:")
    for line in translated_code:
        st.write(f"👉 {line}")
