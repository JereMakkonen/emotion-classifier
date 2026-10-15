import os
import pandas as pd
import numpy as np
import nltk
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
import gensim.downloader as api

class CommentData(Dataset):
    def __init__(self, path: str, embedder, num_classes=28):
        df = pd.read_csv(path, sep="\t", header=None, names=["text", "labels", "id"])
        self.embeddings_path = f"{path.split(".")[0]}-{embedder.model_name}.pt"
        
        if os.path.exists(self.embeddings_path):
            self.embeddings = torch.load(self.embeddings_path)
            print(f"Loaded {len(self)} cached embeddings.")
        else:
            print("Encoding comments...")
            embeddings = embedder.encode(df["text"].tolist())
            self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
            torch.save(self.embeddings, self.embeddings_path)
            print(f"Saved {len(self)} embeddings.")
            
        self.labels = torch.tensor(
            self._extract_labels(df, num_classes), 
            dtype=torch.float32
        )

    def _extract_labels(self, df, num_classes):
        labels = np.zeros((len(df), num_classes), dtype=np.float32)
        
        for n, label in enumerate(df["labels"]):
            labels[n, [int(i) for i in str(label).split(",")]] = 1

        return labels
      
    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return (self.embeddings[idx], self.labels[idx])


class ContextualEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None

    def encode(self, texts):
        if not self.model:
            self.model = SentenceTransformer(self.model_name)
        return self.model.encode(texts, normalize_embeddings=True)


class StaticEmbedder:
    def __init__(self, model_name="glove-twitter-200"):
        self.model_name = model_name
        self.model = None

    def encode(self, texts):
        if not self.model:
            self.model = api.load(self.model_name)
            
        embeddings = []
        for text in texts:
            tokens = [t.lower() for t in nltk.word_tokenize(text) if t.isalpha()]
            vectors = [self.model[token] for token in tokens if token in self.model]
            if vectors:
                embeddings.append(np.mean(vectors, axis=0))
            else:
                embeddings.append(np.zeros(self.model.vector_size, dtype=np.float32))

        return np.array(embeddings, dtype=np.float32)


