
import os
import json
import random
import argparse
from typing import List, Dict, Any, Tuple

import numpy as np # type: ignore
import faiss  # type: ignore
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
from torch.optim import AdamW # type: ignore
from torch.utils.data import DataLoader, Dataset # type: ignore
from tqdm import tqdm # type: ignore
import matplotlib.pyplot as plt # type: ignore

from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    DPRQuestionEncoder,
    AutoTokenizer,
)

from config import RAGTRAIN as CFG


# -------------------------------
# Dataset
# -------------------------------
class QADataset(Dataset):
    """Simple dataset wrapper for question-answer pairs"""

    def __init__(self, qa_list: List[Dict[str, Any]]):
        self.qa_list = qa_list

    def __len__(self):
        return len(self.qa_list)

    def __getitem__(self, idx: int):
        return self.qa_list[idx]


# -------------------------------
# Trainer
# -------------------------------
class RAGTrainer:
    def __init__(self, q_model_path: str, bart_model_path: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load retriever (DPR) and generator (BART)
        self.q_encoder = DPRQuestionEncoder.from_pretrained(q_model_path).to(self.device)
        self.q_tokenizer = AutoTokenizer.from_pretrained(q_model_path)

        self.generator = BartForConditionalGeneration.from_pretrained(bart_model_path).to(self.device)
        self.gen_tokenizer = BartTokenizer.from_pretrained(bart_model_path)

        # Directories from config
        self.log_dir = CFG["log_dir"]
        self.ckpt_dir = CFG["checkpoint_dir"]
        self.save_dir = CFG["save_dir"]
        self.acc_dir = CFG["accuracy_log_dir"]

        for d in [self.log_dir, self.ckpt_dir, self.save_dir, self.acc_dir]:
            os.makedirs(d, exist_ok=True)

    # -------------------------------
    # Utilities
    # -------------------------------
    def set_seed(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def encode_questions(self, questions: List[str]) -> np.ndarray:
        """Get embeddings from DPR encoder"""
        inputs = self.q_tokenizer(
            questions, return_tensors="pt", padding=True, truncation=True, max_length=256
        ).to(self.device)
        with torch.no_grad():
            reps = self.q_encoder(**inputs).pooler_output
        return F.normalize(reps, dim=1).cpu().numpy()

    def retrieve_passages(
        self, query_vecs: np.ndarray, faiss_index: faiss.Index, passages: List[str], top_k: int = 10
    ) -> Tuple[List[List[str]], List[np.ndarray]]:
        """Retrieve top-k docs for each query"""
        scores, indices = faiss_index.search(query_vecs.astype(np.float32), top_k)
        docs = [[passages[i] for i in idx] for idx in indices]
        probs = [np.exp(s) / np.exp(s).sum() for s in scores]  # softmax
        return docs, probs

    def exact_match_accuracy(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        preds = logits.argmax(dim=-1)
        correct = (preds == labels).float()
        return correct.mean().item()

    # -------------------------------
    # Training Core
    # -------------------------------
    def train_epoch(
        self,
        dataloader: DataLoader,
        faiss_index: faiss.Index,
        passages: List[str],
        optimizer: torch.optim.Optimizer,
        epoch: int,
    ):
        self.q_encoder.train()
        self.generator.train()

        loss_fn = nn.CrossEntropyLoss()
        total_loss, total_acc, step = 0, 0, 0  ### step is batch 
        step_losses, step_accs = [], []

        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            step += 1
            questions = [item["question"] for item in batch]
            answers = [item["answer"] for item in batch]

            # Encode queries and retrieve passages
            q_vecs = self.encode_questions(questions) # numpy array [B x 768]
            retrieved_docs, doc_probs = self.retrieve_passages(q_vecs, faiss_index, passages)  # list of list of docs and list of numpy array

            batch_loss, batch_acc = 0, 0
            for i in range(len(batch)):
                # Prompt creation
                prompts = [
                    f"Context: {ctx}\nQuestion: {questions[i]}"
                    for ctx in retrieved_docs[i]
                ] # list(length -> top_k) of prompts for a questions 

                # Encode contexts
                encoder_outputs = []
                for p in prompts:
                    enc_inp = self.gen_tokenizer(p, return_tensors="pt", truncation=True, padding=True).to(self.device)
                    out = self.generator.model.encoder(**enc_inp) # (1, L_enc, 1024) token level encoding
                    encoder_outputs.append(out) # list of length top_k each with (1, L_enc, 1024)

                # Target preparation
                target = self.gen_tokenizer(answers[i], return_tensors="pt").input_ids.to(self.device) # shape:(1, L_tgt)
                dec_inp = target[:, :-1]   # input for decoder for teacher forcing shifted to right, droped EOS token, shape (1, L_tgt-1)
                labels = target[:, 1:]  # this is for loss calculation, droped start token, shape (1, L_tgt-1)

                # Generate logits for each context
                logits_list = []
                for k, enc in enumerate(encoder_outputs):
                    out = self.generator(encoder_outputs=enc, decoder_input_ids=dec_inp, return_dict=True)  #shape: (1, L_tgt-1, V)
                    logits = F.log_softmax(out.logits, dim=-1) #shape: (1, L_tgt-1, V)
                    logits_list.append(logits * doc_probs[i][k])  # list of length top_k witch tesors of  shape (1, L_tgt-1, V)

                # Combine
                final_logits = torch.stack(logits_list).sum(dim=0).squeeze(0) # .sum(dim=0) → (1, L_tgt-1, V), .squeeze(0) → (L_tgt-1, V)
                labels = labels.squeeze(0)

                loss = loss_fn(final_logits, labels)
                acc = self.exact_match_accuracy(final_logits, labels)

                batch_loss += loss
                batch_acc += acc

            batch_loss /= len(batch)
            batch_acc /= len(batch)

            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += batch_loss.item()
            total_acc += batch_acc

            step_losses.append((step, batch_loss.item()))
            step_accs.append((step, batch_acc))

            if step % 100 == 0:
                print(f"[Epoch {epoch} Step {step}] Loss: {batch_loss.item():.4f}, Acc: {batch_acc:.4f} ")

            if step % 1000 == 0:
                self.save_checkpoint(epoch, step)
                self.log_step_accuracy(epoch, step, batch_acc)

        avg_loss = total_loss / len(dataloader)
        avg_acc = total_acc / len(dataloader)

        self.log_epoch_losses(epoch, step_losses)
        # self.log_epoch_accuracy(epoch, step_accs)
        self.plot_epoch_curves(epoch, step_losses, step_accs)

        return avg_loss, avg_acc

    # -------------------------------
    # Saving + Logging
    # -------------------------------
    def save_checkpoint(self, epoch: int, step: int):
        path = os.path.join(self.ckpt_dir, f"epoch{epoch}_step{step}")
        os.makedirs(path, exist_ok=True)
        self.q_encoder.save_pretrained(f"{path}/q_encoder")
        self.q_tokenizer.save_pretrained(f"{path}/q_encoder")
        self.generator.save_pretrained(f"{path}/generator")
        self.gen_tokenizer.save_pretrained(f"{path}/generator")

    def save_models(self, epoch: int):
        q_path = os.path.join(self.save_dir, f"q_encoder_epoch{epoch}")
        g_path = os.path.join(self.save_dir, f"generator_epoch{epoch}")
        self.q_encoder.save_pretrained(q_path)
        self.q_tokenizer.save_pretrained(q_path)
        self.generator.save_pretrained(g_path)
        self.gen_tokenizer.save_pretrained(g_path)
        print(f"✅ Models saved: {q_path}, {g_path}")

    def log_epoch_losses(self, epoch: int, step_losses: List[Tuple[int, float]]):
        with open(os.path.join(self.log_dir, f"epoch{epoch}_loss.txt"), "w") as f:
            for step, loss in step_losses:
                f.write(f"Step {step}: {loss:.4f}\n")

    def log_step_accuracy(self, epoch: int, step: int, acc: float):
        with open(os.path.join(self.acc_dir, f"epoch{epoch}_step{step}_acc.txt"), "w") as f:
            f.write(f"Step {step}: {acc:.4f}\n")

    # def log_epoch_accuracy(self, epoch: int, step_accs: List[Tuple[int, float]]):
    #     with open(os.path.join(self.acc_dir, f"epoch{epoch}_acc.txt"), "w") as f:
    #         for step, acc in step_accs:
    #             f.write(f"Step {step}: {acc:.4f}\n")

    def plot_epoch_curves(self, epoch: int, step_losses, step_accs):
        steps = [s for s, _ in step_losses]
        losses = [l for _, l in step_losses]
        accs = [a for _, a in step_accs]

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(steps, losses, label="Loss")
        plt.title(f"Epoch {epoch} Loss")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(steps, accs, color="orange", label="Accuracy")
        plt.title(f"Epoch {epoch} Accuracy")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f"epoch{epoch}_tokenloss_acc.png"))
        plt.close()

    # -------------------------------
    # Main Training Loop
    # -------------------------------
    def train(self, qa_file, index_file, passages_file, num_epochs, batch_size, lr):
        # Load Data
        with open(qa_file) as f:
            qa_data = [json.loads(line) for line in f]
        faiss_index = faiss.read_index(index_file)
        with open(passages_file) as f:
            passages = [line.strip() for line in f]

        dataset = QADataset(qa_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)

        optimizer = AdamW(list(self.q_encoder.parameters()) + list(self.generator.parameters()), lr=lr)

        all_losses, all_accs = [], []
        for epoch in range(num_epochs):
            print(f"\n----- Epoch {epoch+1}/{num_epochs} -----")
            avg_loss, avg_acc = self.train_epoch(dataloader, faiss_index, passages, optimizer, epoch+1)
            print(f"Epoch {epoch} Done | Loss={avg_loss:.4f} | Acc={avg_acc:.4f}")
            all_losses.append(avg_loss)
            all_accs.append(avg_acc)
            self.save_models(epoch+1)

        # Plot across epochs
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, num_epochs + 1), all_losses, label="Loss")
        plt.plot(range(1, num_epochs + 1), all_accs, label="Accuracy", color="orange")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.log_dir, "tokentrain_summary.png"))
        plt.close()


# -------------------------------
# CLI Entry
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_file", required=True)
    parser.add_argument("--index_file", required=True)
    parser.add_argument("--passages_file", required=True)
    parser.add_argument("--q_model_path", default="models/question_encoder")
    parser.add_argument("--bart_model_path", default="models/BARTLarge")
    args = parser.parse_args()

    trainer = RAGTrainer(args.q_model_path, args.bart_model_path, device=CFG.get("device"))
    trainer.set_seed(CFG.get("seed", 42))

    trainer.train(
        qa_file=args.qa_file,
        index_file=args.index_file,
        passages_file=args.passages_file,
        num_epochs=CFG.get("num_epochs", 1),
        batch_size=CFG.get("batch_size", 8),
        lr=CFG.get("learning_rate", 2e-5),
    )


if __name__ == "__main__":
    main()

# python train_ragtoken.py --qa_file TriviaQA/train/qa_pairs.jsonl --index_file TriviaQA/train_index.faiss --passages_file TriviaQA/train_index.txt