#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn


# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(f)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
       	super(ModelEmbeddings, self).__init__()


        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1f
        pad_token_idx = vocab.char2id['<pad>']

        self.embed_size = embed_size

        self.embed_char_size = 50

        self.char_embed = nn.Embedding(len(vocab.char2id),

                                          self.embed_char_size,

                                           pad_token_idx)

        self.cnn = CNN(f=self.embed_size)

        self.highway = Highway(embed_size=self.embed_size)

        self.dropout = nn.Dropout(p=0.3)
        ### END YOUR CODE

    def forward(self, input_tensor):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input_tensor: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1f 
        src_len, b, _ = input_tensor.shape

        emb = self.char_embed(input_tensor)

        emb_reshaped = emb.reshape(emb.size(0)*emb.size(1),emb.size(2),emb.size(3)).permute(0,2,1)
        conv_out = self.cnn(emb_reshaped)
        highway = self.highway(conv_out) 
        word_emb= self.dropout(highway)
        output = word_emb.reshape(src_len, b, word_emb.size(1))
        return output        
        ### END YOUR CODE
