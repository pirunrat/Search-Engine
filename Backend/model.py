from utils import Utils
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn.functional as F

class Model:
    def __init__(self,model_path) -> None:

        # Loading all necsessary objects
        word2index = Utils('./files/word2index.pkl')
        vocabs = Utils('./files/vocabs.pkl')
        index2word = Utils('./files/index2word.pkl')
        self.vocabs = vocabs.load()
        self.word2index = word2index.load()
        self.index2word = index2word.load()
    

        # Loading the model
        model_util = Utils(model_path)
        self.model = model_util.load_pytorch_model()
        


    def get_embed_vec(self, word):
        id_tensor = torch.LongTensor([self.word2index[word]])
        v_embed = self.model.embedding_v(id_tensor)
        u_embed = self.model.embedding_u(id_tensor)
        word_embed = (v_embed + u_embed) / 2
        return word_embed



    
    


    def predict_similar_words(self, input_word, k):
        # Step 1: Get the embedding for the input word
        input_index = self.word2index.get(input_word, -1)
        if input_index == -1:
            print(f"Word '{input_word}' not in vocabulary.")
            return ["Not Found"]

        input_tensor = torch.LongTensor([input_index])
        input_embedding = self.model.embedding_v(input_tensor)

        # Step 2: Calculate similarity with other words
        all_indices = [self.word2index[word] for word in self.vocabs if word in self.word2index]
        if not all_indices:
            print("No words in vocabulary for comparison.")
            return []

        all_tensor = torch.LongTensor(all_indices)
        all_embeddings = self.model.embedding_u(all_tensor)

        # Ensure correct dimensions for cosine similarity
        input_embedding = input_embedding.unsqueeze(0)
        similarities = F.cosine_similarity(input_embedding, all_embeddings, dim=2)

        # Step 3: Get top-K similar words
        _, top_indices = torch.topk(similarities, k + 1)  # +1 to exclude the input word itself
        similar_words = [self.index2word[index.item()] for index in top_indices[0] if index.item() != input_index]

        return similar_words
        
        

        