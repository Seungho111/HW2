import os

class Config:
    # Use a lightweight language model suitable for local environments. 
    # Qwen 0.5B is small and has great multilingual (including Chinese) capabilities.
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen1.5-0.5B-Chat") 
    
    # Switch to 'cuda' if you intend to run this on a GPU in your MLOps pipeline
    DEVICE = os.getenv("DEVICE", "cpu") 
