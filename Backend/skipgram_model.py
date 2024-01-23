import torch.nn as nn
import torch
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F

class Skipgram(nn.Module):

    def __init__(self, vocab_size, emb_size):
        
        
        
        super(Skipgram, self).__init__()
        self.embedding_v = nn.Embedding(vocab_size, emb_size)
        self.embedding_u = nn.Embedding(vocab_size, emb_size)

    def forward(self, center_words, target_words, all_vocabs):
        center_embeds = self.embedding_v(center_words)  # [batch_size, 1, emb_size]
        target_embeds = self.embedding_u(target_words)  # [batch_size, 1, emb_size]
        all_embeds = self.embedding_u(all_vocabs)  # [batch_size, voc_size, emb_size]

        scores = target_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2)
        # [batch_size, 1, emb_size] @ [batch_size, emb_size, 1] = [batch_size, 1, 1] = [batch_size, 1]

        norm_scores = all_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2)
        # [batch_size, voc_size, emb_size] @ [batch_size, emb_size, 1] = [batch_size, voc_size, 1] = [batch_size, voc_size]

        nll = -torch.mean(torch.log(torch.exp(scores) / torch.sum(torch.exp(norm_scores), 1).unsqueeze(1)))  # log-softmax
        # scalar (loss must be scalar)

        return nll  # negative log likelihood
    
    

    def get_embed_vec(self, word):
        id_tensor = torch.LongTensor([self.word2index[word]])
        v_embed = self.model.embedding_v(id_tensor)
        u_embed = self.model.embedding_u(id_tensor)
        word_embed = (v_embed + u_embed) / 2
        return word_embed
    
    