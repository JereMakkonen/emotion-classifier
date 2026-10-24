import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, recall_score
from dataset import CommentData, ContextualEmbedder, StaticEmbedder

class EmotionClassifier(nn.Module):
    def __init__(
        self, 
        input_dim, 
        num_classes, 
        hidden_dim=256,
        dropout=0.4
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.network(x)


class EmotionTrainer:
    def __init__(
        self,
        model,
        train_data,
        valid_data,
        test_data,
        num_epochs=50,
        model_path="data/emotion_classifier.pt",
    ):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.model = model.to(self.device)
        
        for dataset in (train_data, valid_data, test_data):
            dataset.embeddings = dataset.embeddings.to(self.device)
            dataset.labels = dataset.labels.to(self.device)
        
        self.num_epochs = num_epochs
        self.model_path = model_path
        self.train_loader = DataLoader(train_data, batch_size=1024, shuffle=True)
        self.valid_loader = DataLoader(valid_data, batch_size=1024, shuffle=False)
        self.test_loader = DataLoader(test_data, batch_size=1024, shuffle=False)

        # Label weights
        pos_counts = train_data.labels.sum(dim=0)
        neg_counts = len(train_data.labels) - pos_counts
        pos_weight = torch.sqrt(neg_counts / pos_counts).float().to(self.device)
        
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.optimizer = optim.AdamW(self.model.parameters())
        self.best_f1 = 0.0


    def evaluate(self, loader):
        self.model.eval()
        predictions, labels_all =  [], []
        total_loss = 0 

        with torch.no_grad():
            for embeddings, labels in loader:
                logits = self.model(embeddings)
                total_loss += self.criterion(logits, labels).item() * len(labels)

                predictions.append((torch.sigmoid(logits) >= 0.5).int().cpu())
                labels_all.append(labels.int().cpu())

        y_pred = torch.cat(predictions).numpy()
        y_true = torch.cat(labels_all).numpy()

        return (
            total_loss / len(loader.dataset),
            recall_score(y_true, y_pred, average="macro", zero_division=0),
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        )
        
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0

        for embeddings, labels in self.train_loader:
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(embeddings), labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)


    def train(self):
        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch()
            loss, recall, f1 = self.evaluate(self.valid_loader)

            print(
                f"Epoch {epoch + 1}/{self.num_epochs}:\t"
                f"Training loss: {train_loss:.4f} | "
                f"Validation loss: {loss:.4f} | "
                f"Validation recall: {recall:.4f} | "
                f"Validation F1-score: {f1:.4f}"
            )

            if f1 > self.best_f1:
                self.best_f1 = f1
                torch.save(self.model.state_dict(), self.model_path)

        self.model.load_state_dict(torch.load(self.model_path, weights_only=True))
        return self.model


    def test(self):
        loss, recall, f1 = self.evaluate(self.test_loader)
        print(
            f"Test data loss: {loss:.4f} | "
            f"Test data recall: {recall:.4f} | "
            f"Test data F1-score: {f1:.4f}"
        )


if __name__ == "__main__":
    
    embedder = ContextualEmbedder()
    train_data = CommentData("data/train.tsv", embedder)
    test_data = CommentData("data/test.tsv", embedder)
    valid_data = CommentData("data/dev.tsv", embedder)
    
    model = EmotionClassifier(
        input_dim=train_data.embeddings.shape[1],
        num_classes=train_data.labels.shape[1],
    )
    
    trainer = EmotionTrainer(
        model=model,
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
    )

    trainer.train()
    trainer.test()
    