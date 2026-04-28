# Central configuration for non-path parameters

VECTOR_INDEX = {
    "chunk_size": 100,
    "batch_size": 64,
    "index_name": "train_index",
    "device": "cuda:2",  
}

TRAIN = {
    "num_epochs": 5,
    "batch_size": 16,
    "learning_rate": 2e-4,
    "device": "cuda:3",  
    "seed": 42,
}

EVALUATE = {
    "batch_size": 32,
    "top_k": 4,
    "max_len": 8,
    "device": None,
    "seed": 42,
}

T5 = {
    "max_length": 8,
    "device": None,
} 

RAGTRAIN = {
    "num_epochs": 1,
    "batch_size": 8,
    "learning_rate": 5e-6,
    "device": "cuda:3",  
    "seed": 42,
    "log_dir":  "token_loss_log",
    "checkpoint_dir": "token_checkpoints",
    "save_dir": "token_saved_models",
    "accuracy_log_dir": "token_accuracy_logs", 
}