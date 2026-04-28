import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
from torch.utils.data import DataLoader, Dataset # type: ignore
from transformers import get_scheduler
from torch.optim import AdamW # type: ignore
from tqdm import tqdm # type: ignore
import json
import numpy as np # type: ignore
import faiss # type: ignore
import random
import os
from transformers import (
    DPRQuestionEncoder, AutoTokenizer,
    BartForConditionalGeneration, BartTokenizer,
)

# ========== Setup ==========
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
DEVICE = "cuda:2"

# ========== Load Models ==========
q_model_path = "token_checkpoints/epoch1_step13000/q_encoder"
gen_model_path = "token_checkpoints/epoch1_step13000/generator"
q_encoder = DPRQuestionEncoder.from_pretrained(q_model_path).to(DEVICE)
q_tokenizer = AutoTokenizer.from_pretrained(q_model_path)
bart_model = BartForConditionalGeneration.from_pretrained(gen_model_path).to(DEVICE)
bart_tokenizer = BartTokenizer.from_pretrained(gen_model_path)

# ========== Dataset ==========
class QADataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
    
# ========== Query Encoding ========== 
def encode_queries(queries):
    inputs = q_tokenizer(queries, return_tensors="pt", padding=True, truncation=True, max_length=256).to(DEVICE)
    with torch.no_grad():
        reps = q_encoder(**inputs).pooler_output
    return F.normalize(reps, dim=1).cpu().numpy()

# ========== FAISS Retrieval ==========
def retrieve_top_k_batch(query_embeddings, faiss_index, loaded_texts, top_k=5):
    scores, indices = faiss_index.search(query_embeddings.astype(np.float32), top_k)
    all_retrieved = [[loaded_texts[i] for i in ind] for ind in indices]
    all_probs = [np.exp(s) / np.exp(s).sum() for s in scores]
    return all_retrieved, all_probs

#========= Generate prediction for EM==========
def calulate_accuracy(pred_answers, answers):
    total = 0
    correct = 0
    # print(pred_answers)
    # print(answers)
    for pred_answer, answer in zip(pred_answers, answers):
        #print(pred_answer, answer)
        if pred_answer.strip().lower() == answer.strip().lower():
            correct += 1
        total += 1
        #print(correct)
    accuracy = 100 * correct / total
    return accuracy
# ================ generate predictions ==========
def generate_bart(query: str, texts, retrieval_probs, max_len: int = 64):
    # Step 1: Retrieve top-k contexts
    #retrieved_passages, retrieval_probs = retrieve_top_k(query_embedding, index, texts, top_k)

    # Step 2: Encode each context+query pair
    encoder_outputs = []
    for passage in texts:
        prompt = f"Given the context, answer the question.\n\nContext: {passage}\nQuestion: {query}"
        inputs = bart_tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
        with torch.no_grad():
            enc_out = bart_model.model.encoder(**inputs)
        encoder_outputs.append(enc_out)

    # Step 3: Greedy decoding with RAG-token formula
    input_ids = torch.tensor([[bart_model.config.decoder_start_token_id]], device=DEVICE)  # start token
    generated_tokens = []

    for _ in range(max_len):
        all_probs = []

        for i in range(len(texts)):
            out = bart_model(
                encoder_outputs=encoder_outputs[i],
                decoder_input_ids=input_ids,
                return_dict=True,
            )
            logits = out.logits[:, -1, :]  # shape: (1, vocab_size)
            probs = F.softmax(logits, dim=-1)
            weighted = probs * retrieval_probs[i]
            all_probs.append(weighted)

        mixed_probs = torch.stack(all_probs).sum(dim=0)
        next_token_id = torch.argmax(mixed_probs, dim=-1)
        next_token = next_token_id.item()

        if next_token == bart_tokenizer.eos_token_id:
            break

        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)
        generated_tokens.append(next_token)

    return bart_tokenizer.decode(generated_tokens, skip_special_tokens=True)

# ============= Main Function ====================

def main():
    ## for test data
    faiss_index = faiss.read_index("TriviaQA/test_index.faiss")
    with open("TriviaQA/test_index.txt") as f:
        corpus = [line.strip() for line in f]
    with open("TriviaQA/test/qa_pairs.jsonl") as f:
        qa_data = [json.loads(line) for line in f]
        
    # ### for sample data
    # faiss_index = faiss.read_index("index_sample_new.faiss")
    # with open("sample_new.txt") as f:
    #     corpus = [line.strip() for line in f]
    # with open("sample/qa_pairs.json") as f:
    #     qa_data = [json.loads(line) for line in f]
    
    gold = []
    for item in qa_data:
        gold.append(item['answer'])

    dataset = QADataset(qa_data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=lambda x: x)
    bart_model.eval()
    q_encoder.eval()
    pred_answers = []
    for batch in tqdm(dataloader):
        questions = [x["question"] for x in batch]
        answers = [x["answer"] for x in batch]
        query_embeddings = encode_queries(questions)
        retrieved, probs = retrieve_top_k_batch(query_embeddings, faiss_index, corpus)
        for i in range(len(batch)):
            pred = generate_bart(questions[i], retrieved[i], probs[i], max_len=8)
            pred_answers.append(pred)
            # print(questions[i])
            # print(answers[i])
            # print(pred)
            # print("***********************************")
            
    accuracy = calulate_accuracy(pred_answers, gold)
    print(f"\nExact Match Accuracy: {accuracy:.2f}%")
    accuracy_path = f"epoch1_step13000.txt"
    with open(accuracy_path, "w") as f:
        f.write(f"\nExact Match Accuracy: {accuracy:.4f}%")
main()