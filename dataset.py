import os
import pandas as pd
import numpy as np
import nltk
import torch
import gensim.downloader as api
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer


class CommentData(Dataset):
    def __init__(
        self,
        path,
        embedder,
        emotions_path = "data/emotions.csv",
        task = "emotions",
    ):
        data = pd.read_csv(
            path, 
            sep="\t", 
            header=None, 
            names=["text", "labels", "id"]
        )
        
        emotions = pd.read_csv(emotions_path)
        sentiments = {"negative" : 0, "positive": 1, "ambiguous": 2, "neutral": 3}
        self.sentiment_map = emotions["sentiment"].map(sentiments).to_numpy()
        
        self.task = task
        self.labels = torch.tensor(self._extract_labels(data), dtype=torch.float32)
        self.embeddings_path = f"{os.path.splitext(path)[0]}_{embedder.model_name}.pt"
        
        if os.path.exists(self.embeddings_path):
            self.embeddings = torch.load(
                self.embeddings_path,
                weights_only=True,
            )
            print(f"Loaded {len(self)} cached embeddings.")
        else:
            print("Encoding comments")
            
            embeddings = embedder.encode(data["text"].tolist())
            self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
            
            torch.save(self.embeddings, self.embeddings_path)
            print(f"Saved {len(self)} embeddings.")

    
    def _extract_labels(self, df):
        num_classes = 28 if self.task == "emotions" else 4
        labels = np.zeros((len(df), num_classes), dtype=np.float32)

        for n, label in enumerate(df["labels"]):
            ids = [int(i) for i in str(label).split(",")]

            if self.task == "emotions":
                labels[n, ids] = 1
            else:
                labels[n, self.sentiment_map[ids]] = 1

        return labels
      
    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return (self.embeddings[idx], self.labels[idx])


class ContextualEmbedder:
    def __init__(self, model_name):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def encode(self, texts):
        if not self.model:
            self.model = SentenceTransformer(self.model_name, device=self.device)
        return self.model.encode(texts, normalize_embeddings=True)


class StaticEmbedder:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None

    def encode(self, texts):
        if not self.model:
            self.model = api.load(self.model_name)
            
        embeddings = []
        for text in texts:
            tokens = [t.lower() for t in nltk.word_tokenize(text)]
            vectors = [self.model[token] for token in tokens if token in self.model]
            if vectors:
                embeddings.append(np.mean(vectors, axis=0))
            else:
                embeddings.append(np.zeros(self.model.vector_size, dtype=np.float32))

        return np.array(embeddings, dtype=np.float32)
