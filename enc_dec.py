# coding: utf-8
#---------------------------------------------------------------------
'''
Neural Machine Translation - Encoder Decoder model
    Chainer implementation of an encoder-decoder sequence to sequence
    model using bi-directional LSTM encoder
'''
#---------------------------------------------------------------------

# In[]:

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.functions.array import concat
import pdb

# In[]:
# Import configuration file

from nmt_config import *

# In[]:

class EncoderDecoder(Chain):

    '''
    Constructor to initialize model
    Params:
        vsize_enc   - vocabulary size for source language (fed into encoder)
        vsize_dec   - vocabulary size for target language (fed into decoder)
        n_units     - size of the LSTMs
        attn        - specifies whether to use attention
    '''
    def __init__(self, vsize_enc, vsize_dec,
                 nlayers_enc, nlayers_dec,
                 n_units, gpuid, attn=False):
        super(EncoderDecoder, self).__init__()
        #--------------------------------------------------------------------
        # add encoder layers
        #--------------------------------------------------------------------
        # add embedding layer
        self.add_link("embed_enc", L.EmbedID(vsize_enc, n_units))

        '''
        ___QUESTION-1-DESCRIBE-A-START___

        - Explain the following lines of code
        - Think about what add_link() does and how can we access Links added in Chainer.
        - Why are there two loops or adding links?

        the codes below will add LSTM layers in our RNN architecture.
        chaner use a term "link" to define the module to save and operate the parameter variable, simply link is a layer or another model which we will like to attach in our neural network (chain)
        - add_link is used to attach link (in this case layer) to our chain, chain is a class that consist of one or more link. we can think of chain as a "chain that tie up together our link".
        - there are two loops because we are making a a bidirectional RNN so the last one with rev suffix is the layer for the right-to-left order input
        '''
        self.lstm_enc = ["L{0:d}_enc".format(i) for i in range(nlayers_enc)]

        for lstm_name in self.lstm_enc:
            self.add_link(lstm_name, L.LSTM(n_units, n_units))


        self.lstm_rev_enc = ["L{0:d}_rev_enc".format(i) for i in range(nlayers_enc)]
        for lstm_name in self.lstm_rev_enc:
            self.add_link(lstm_name, L.LSTM(n_units, n_units))
        '''
        ___QUESTION-1-DESCRIBE-A-END___
        '''

        #--------------------------------------------------------------------
        # add decoder layers
        #----------------F----------------------------------------------------
        # add embedding layer
        '''
        ___QUESTION-1-DESCRIBE-B-START___
        Comment on the input and output sizes of the following layers:

        - L.EmbedID(vsize_dec, 2*n_units)
        for word embeeding layer, the input size will be the size of target vocabulary  (vsize_dec), the output will be 2 times the number of hidden unit in LSTM (n_units)
        - L.LSTM(2*n_units, 2*n_units)
        for word embeeding layer, the input size should be similar to the number of output from previous layer so the input size will be 2*n_units and the output size is similar as input
        - L.Linear(2*n_units, vsize_dec)
        same as before the input for this layer should be similar to previous LSTM layer, the diffrence is the output of this neural network will be the number of target vocabulary (size of prediction output)

        Why are we using multipliers over the base number of units (n_units)?
        We multiply the n_units by 2 because we will concatenated the the output of left-to-right and right-to-left layers encoder.
        '''

        self.add_link("embed_dec", L.EmbedID(vsize_dec, 2*n_units))

        self.MID_LAYER = "mid-layer"
        # add LSTM layers
        self.lstm_dec = ["L{0:d}_dec".format(i) for i in range(nlayers_dec)]
        for lstm_name in self.lstm_dec:
            self.add_link(lstm_name, L.LSTM(2*n_units, 2*n_units))

        linear_size_input = 2*n_units
        if attn > 0:
            # __QUESTION Add attention
            # linear_size_input = 2*n_units*2 # we concatinate c and ht then the size of the input of linear will be doubled
            self.add_link(self.MID_LAYER, L.Linear(4*n_units, 2*n_units))
            # pdb.set_trace()

        # Save the attention preference
        # __QUESTION you should use this flag to check if attention
        # has been selected. Your code should work with and without attention
        self.attn = attn
        self.alpha = []
        self.ct = []
        self.ht = []
        # add output layer
        self.add_link("out", L.Linear(2*n_units, vsize_dec))
        '''
        ___QUESTION-1-DESCRIBE-B-END___
        '''

        # Store GPU id
        self.gpuid = gpuid
        self.n_units = n_units

    def reset_state(self):
        # reset the state of LSTM layers
        for lstm_name in self.lstm_enc + self.lstm_rev_enc + self.lstm_dec:
            self[lstm_name].reset_state()
        self.loss = 0

    '''
        ___QUESTION-1-DESCRIBE-C-START___

        Describe what the function set_decoder_state() is doing. What are c_state and h_state?
        because our encoder produce two version of output snce we have two version of input (left-to-right and right-to-left)
        the otput c and h need to be concate together before becoming input for the decoder
    '''
    def set_decoder_state(self):
        xp = cuda.cupy if self.gpuid >= 0 else np
        c_state = F.concat((self[self.lstm_enc[-1]].c, self[self.lstm_rev_enc[-1]].c))
        h_state = F.concat((self[self.lstm_enc[-1]].h, self[self.lstm_rev_enc[-1]].h))
        self[self.lstm_dec[0]].set_state(c_state, h_state)

    '''___QUESTION-1-DESCRIBE-C-END___'''

    '''
    Function to feed an input word through the embedding and lstm layers
        args:
        embed_layer: embeddings layer to use
        lstm_layer:  list of names of lstm layers to use
    '''
    def feed_lstm(self, word, embed_layer, lstm_layer_list, train, dropout=False):
        # get embedding for word
        embed_id = embed_layer(word)
        # feed into first LSTM layer
        hs = self[lstm_layer_list[0]](embed_id)
        # feed into remaining LSTM layers
        for lstm_layer in lstm_layer_list[1:]:
            if dropout:
                hs = F.dropout(hs,ratio=0.2,train=train)
            hs = self[lstm_layer](hs)

    ## change code to word
    # Function to encode an source sentence word
    # I only put dropout in the encoding
    def encode(self, word, lstm_layer_list, train):
        self.feed_lstm(word, self.embed_enc, lstm_layer_list, train, dropout=True)

    # Function to decode a target sentence word
    def decode(self, word, train):
        self.feed_lstm(word, self.embed_dec, self.lstm_dec, train)

    '''

    '''
    def encode_list(self, in_word_list, train=True):
        # in word is just a french_word_idx
        xp = cuda.cupy if self.gpuid >= 0 else np
        # convert list of tokens into chainer variable list
        var_en = (Variable(cuda.to_gpu(xp.asarray(in_word_list, dtype=np.int32).reshape((-1,1)),
                           volatile=(not train))))

        var_rev_en = (Variable(cuda.to_gpu(xp.asarray(in_word_list[::-1], dtype=np.int32).reshape((-1,1)),
                           volatile=(not train))))

        # array to store hidden states for each word
        # enc_states = xp.empty((0,2*self.n_units), dtype=xp.float32)
        first_entry = True
        # encode tokens
        for f_word, r_word in zip(var_en, var_rev_en):
            '''
            ___QUESTION-1-DESCRIBE-D-START___

            - Explain why we are performing two encode operations
            because we are feeding our RNN model (training) with two version of the same input which are left-to-right and right-to-left
            '''
            self.encode(f_word, self.lstm_enc, train)
            self.encode(r_word, self.lstm_rev_enc, train)

            '''___QUESTION-1-DESCRIBE-D-END___'''


            # __QUESTION -- Following code is to assist with ATTENTION
            # enc_states stores the hidden state vectors of the encoder
            # this can be used for implementing attention
            if first_entry == False:
                forward_states = F.concat((forward_states, self[self.lstm_enc[-1]].h), axis=0)
                backward_states = F.concat((self[self.lstm_rev_enc[-1]].h, backward_states), axis=0)
            else:
                forward_states = self[self.lstm_enc[-1]].h
                backward_states = self[self.lstm_rev_enc[-1]].h
                first_entry = False

        enc_states = F.concat((forward_states, backward_states), axis=1)

        return enc_states


    # Select a word from a probability distribution
    # should return a chainer variable as in index
    def select_word(self, prob, train=True, sample=False):
        xp = cuda.cupy if self.gpuid >= 0 else np
        if not sample:
            indx = xp.argmax(prob.data[0])
            pred_word = Variable(cuda.to_gpu(xp.asarray([indx], dtype=np.int32), volatile=not train))
        else:
            '''
            ___QUESTION-2-SAMPLE

            - Add code to sample from the probability distribution to
            choose the next word
            '''
            indx = np.random.choice(len(prob.data[0]),1,p=prob.data[0])
            pred_word = Variable(cuda.to_gpu(xp.asarray([indx], dtype=np.int32), volatile=not train))
        return pred_word

    def calculate_alignment(self,enc_states,train=True):
        xp = cuda.cupy if self.gpuid >= 0 else np
        self.alpha = []
        self.ct = []
        self.ht = self[self.lstm_dec[-1]].h
        _exp_score = []
        for prev_hs in enc_states:
            _score = xp.dot(self.ht.data[0].T,prev_hs.data)
            _exp_score.append(xp.exp(_score))
        # calculate score for all-t
        for score in _exp_score:
            self.alpha.append(float(score)/sum(_exp_score))

        # after we got alignment we need to multiply it the hs again (weighted average)
        sum_val = 0
        for a,prev_hs in zip(self.alpha,enc_states):
            one_val = prev_hs.data * a
            sum_val += one_val
        self.ct = chainer.Variable(cuda.to_gpu(np.array([sum_val]),volatile=(not train)))
        ht_lambda = F.concat((self.ct, self.ht), axis=1)
        # compute loss
        _out = self[self.MID_LAYER](ht_lambda)
        predicted_out = self.out(chainer.functions.tanh(_out))
        return predicted_out, self.alpha

    def encode_decode_train(self, in_word_list, out_word_list, train=True, sample=False):
        xp = cuda.cupy if self.gpuid >= 0 else np
        self.reset_state()
        # Add GO_ID, EOS_ID to decoder input
        decoder_word_list = [GO_ID] + out_word_list + [EOS_ID]
        # encode list of words/tokens
        enc_states = self.encode_list(in_word_list, train=train)
        # initialize decoder LSTM to final encoder state
        self.set_decoder_state()
        # decode and compute loss
        # convert list of tokens into chainer variable list
        var_dec = (Variable(cuda.to_gpu(cuda.to_gpu(xp.asarray(decoder_word_list, dtype=np.int32).reshape((-1,1)),
                            volatile=not train))))
        # Initialise first decoded word to GOID
        pred_word = Variable(cuda.to_gpu(xp.asarray([GO_ID], dtype=np.int32), volatile=not train))
        # compute loss

        self.loss = 0
        # decode tokens
        for next_word_var in var_dec[1:]:
            self.decode(pred_word, train=train)
            if self.attn == NO_ATTN:
                predicted_out = self.out(self[self.lstm_dec[-1]].h)
            else: #### use attention
                predicted_out,self.alpha = self.calculate_alignment(enc_states,train)
                # pdb.set_trace()
            prob = F.softmax(predicted_out)
            pred_word = self.select_word(prob, train=train, sample=False)
            # pred_word = Variable(xp.asarray([pred_word.data], dtype=np.int32), volatile=not train)
            '''
            ___QUESTION-1-DESCRIBE-E-START___
            Explain what loss is computed with an example
            What does this value mean?out
            this value mean how
            need to be edited => loss function that expresses your preference to what kinds of outputs y you’d like to see in response to your input sequences x.

            '''
            self.loss += F.softmax_cross_entropy(predicted_out, next_word_var)
            '''___QUESTION-1-DESCRIBE-E-END___'''

        report({"loss":self.loss},self)

        return self.loss

    def decoder_predict(self, start_word, enc_states, max_predict_len=MAX_PREDICT_LEN, sample=False):
        xp = cuda.cupy if self.gpuid >= 0 else np

        # __QUESTION -- Following code is to assist with ATTENTION
        # alpha_arr should store the alphas for every predicted word
        alpha_arr = xp.empty((0,enc_states.shape[0]), dtype=xp.float32)

        # return list of predicted words
        predicted_sent = []
        # load start symbol
        pred_word = Variable(cuda.to_gpu(xp.asarray([start_word], dtype=np.int32), volatile=True))
        pred_count = 0

        # start prediction loop
        while pred_count < max_predict_len and (int(pred_word.data) != (EOS_ID)):
            self.decode(pred_word, train=False)

            if self.attn == NO_ATTN:
                prob = F.softmax(self.out(self[self.lstm_dec[-1]].h))
            else:
                predicted_out,alha_arr = self.calculate_alignment(enc_states,False)
                prob = F.softmax(predicted_out)


            pred_word = self.select_word(prob, train=False, sample=sample)
            # add integer id of predicted word to output list
            predicted_sent.append(int(pred_word.data))
            pred_count += 1
        # __QUESTION Add attention
        # When implementing attention, make sure to use alpha_array to store
        # your attention vectors.
        # The visualisation function in nmt_translate.py assumes such an array as input.
        return predicted_sent, alpha_arr

    ##@ they feed the in_word_list with fr_ids
    def encode_decode_predict(self, in_word_list, max_predict_len=20, sample=False):
        xp = cuda.cupy if self.gpuid >= 0 else np
        self.reset_state()
        # encode list of words/tokens
        in_word_list_no_padding = [w for w in in_word_list if w != PAD_ID]
        ## @ do ffed forward training input (this time we don't train) just
        enc_states = self.encode_list(in_word_list, train=False)
        # initialize decoder LSTM to final encoder state
        self.set_decoder_state()
        # decode starting with GO_ID
        predicted_sent, alpha_arr = self.decoder_predict(GO_ID, enc_states,
                                                         max_predict_len, sample=sample)
        return predicted_sent, alpha_arr

# In[]:
