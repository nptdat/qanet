import os
import time
import json
from typing import List
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K
import keras.layers as KL
from keras.layers import *
from keras.layers import Activation
import keras.models as KM
from keras.engine.topology import Layer
from keras.optimizers import Adam
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, TensorBoard
from tqdm import tqdm
import random


from keras.regularizers import l2
from keras.initializers import VarianceScaling
from keras.initializers import RandomUniform

from config import Config as cf
from common import SquadData
from utilities import get_batch, mask_logits, tokenize, tokenize_long_text, to_chars

# Fix seed
random.seed(2018)
np.random.seed(2018)
tf.set_random_seed(2018)




################# Embedding & Highway networks
def init_embedding_weights(word_vectors=None, dropout_rate=0.0):
    def init(shape, dtype=None):
        if word_vectors is not None:   # pretrained word vectors
            weights = tf.convert_to_tensor(word_vectors, dtype=tf.float32)
        else:
            weights = K.random_uniform(shape, 0., 1., dtype=tf.float32)
        if dropout_rate > 0.0:
            weights = KL.Dropout(rate=dropout_rate)(weights)
        return weights

    return init


def word_embedding_graph(word_vectors, dropout_rate=0.0, name='w_emb'):
    return KL.Embedding(
        input_dim=word_vectors.shape[0],
        output_dim=word_vectors.shape[1],
        embeddings_initializer=init_embedding_weights(word_vectors, dropout_rate),
        trainable=False,
        name=name
    )


def char_embedding_graph(input_dim, output_dim, cf, dropout_rate=0.0, regularizer=None, name='c_emb'):
    layers = KM.Sequential(name=name)
    layers.add(KL.Embedding(
        input_dim=input_dim,
        output_dim=output_dim,
        embeddings_initializer=init_embedding_weights(None, dropout_rate),
        trainable=True,
        embeddings_regularizer=regularizer,
        name=name+'_emb')
    )
    layers.add(KL.Lambda(lambda x: tf.reshape(x, (-1, cf.WORD_LEN, output_dim)), name=name+'_reshape'))
    layers.add(KL.Conv1D(cf.C_CONV_FILTERS, cf.C_CONV_KERNEL, activation='relu', kernel_regularizer=regularizer, name=name+'_conv1d'))
    layers.add(KL.GlobalMaxPool1D(name=name+'_pool1d'))
    return layers


def highway_graph(dim, num_layers=2, h_activation='relu', regularizer=None, name='highway'):
    sublayers = []
    for k in range(num_layers):
        sublayers.append(KL.TimeDistributed(
            KL.Dense(dim, activation='relu', kernel_regularizer=regularizer),
            name=name+'_outputs_'+str(k))
        )
        sublayers.append(KL.TimeDistributed(
            KL.Dense(dim, activation='sigmoid', kernel_regularizer=regularizer),
            name=name+'_transform_'+str(k))
        )
    # Note that TimeDistributed will fix the length of the sequence for the 1st call.
    # So, for multi-call, the input sequences must have the same length.
    return sublayers

def highway_forward(x, sublayers, dropout_rate=0.0):
    """
    Ref: https://arxiv.org/pdf/1505.00387.pdf
    """
    def highway(inputs):
        x, H, T = inputs
        return H*T + x*(1-T)

    for k in range(len(sublayers)//2):  # Loop through num of highway layers
        H = sublayers[k*2](x)
        T = sublayers[k*2+1](x)
        y = KL.Lambda(highway)([x, H, T])
        if dropout_rate > 0.0:
            y = KL.Dropout(rate=dropout_rate)(y)
        x = y
    return x



########################## Multihead attention
def call_activation(x, activation):
    if isinstance(activation, str):
        activation = Activation(activation)
    return activation(x)

def scaled_dot_product_attention(Q, K, V, mask=None, scale=1, dropout_rate=0.0):
    """
    Scaled Dot-Product attention.
    Ref: https://arxiv.org/pdf/1706.03762.pdf (3.2.1 Scaled Dot-Product Attention)

    Args:
        Q: query with shape (b, q, d_k)
        K: keys with shape (b, m, d_k)
        V: values with shape (b, m, d_v)
        scale: the scale factor of the dot-product of Q & K.
                   If None, the scale will be d_k.
                   If 1, no scale.
        dropout_rate: dropout right after the softmax layer

    Returns:
        Attention with shape (b, q, d_v) as the weighted sum of values in V,
        where the weights are calculated from the compatibility function between Q & K.
    """
    if scale is None:
        scale = tf.shape(Q)[-1]
    scale = tf.to_float(scale)
    weights = tf.matmul(Q, K, transpose_b=True)
    weights = weights / tf.sqrt(scale)
    if mask is not None:
        weights = mask_logits(weights, tf.expand_dims(mask, axis=1))
    weights = tf.nn.softmax(weights, axis=-1)
    if dropout_rate > 0:
        weights = KL.Dropout(rate=dropout_rate)(weights)
    attn = tf.matmul(weights, V)
    return attn


class MultiheadAttention(KL.Layer):
    def __init__(self, num_heads=8, initializer='glorot_uniform', activation=None, dropout_rate=0.0, regularizer=None, **kwargs):
        self.h = num_heads
        self.initializer = initializer
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.regularizer = regularizer
        super(MultiheadAttention, self).__init__(**kwargs)


    def build(self, input_shape):
        Q_shape = input_shape[0]
        self.d_model = Q_shape[-1]
        self.d_k = self.d_v = self.d_model // self.h
        self.W_Q = []
        self.W_K = []
        self.W_V = []
        for i in range(self.h):
            self.W_Q.append(self.add_weight(name='kernel',
                                            shape=(1, self.d_model, self.d_k),
                                            initializer=self.initializer,
                                            regularizer=self.regularizer,
                                            trainable=True))
            self.W_K.append(self.add_weight(name='kernel',
                                            shape=(1, self.d_model, self.d_k),
                                            initializer=self.initializer,
                                            regularizer=self.regularizer,
                                            trainable=True))
            self.W_V.append(self.add_weight(name='kernel',
                                            shape=(1, self.d_model, self.d_k),
                                            initializer=self.initializer,
                                            regularizer=self.regularizer,
                                            trainable=True))
        self.W_O = self.add_weight(name='kernel',
                                   shape=(1, self.h * self.d_v, self.d_model),
                                   initializer=self.initializer,
                                   regularizer=self.regularizer,
                                   trainable=True)
        super(MultiheadAttention, self).build(input_shape)


    def call(self, inputs):
        mask = None
        if isinstance(inputs, list) or isinstance(inputs, tuple):
            Q = KK = V = inputs[0]
            if len(inputs) > 1:
                mask = inputs[1]
        else:
            Q = KK = V = inputs

        heads = []
        for i in range(self.h):
            Q_i = K.conv1d(Q, self.W_Q[i])
            if self.activation is not None:
                Q_i = call_activation(Q_i, self.activation)

            K_i = K.conv1d(KK, self.W_K[i])
            if self.activation is not None:
                K_i = call_activation(K_i, self.activation)

            V_i = K.conv1d(V, self.W_V[i])
            if self.activation is not None:
                V_i = call_activation(V_i, self.activation)

            head = scaled_dot_product_attention(Q_i, K_i, V_i, mask=mask, scale=self.d_k, dropout_rate=self.dropout_rate)
            heads.append(head)

        multi = tf.concat(heads, axis=-1)
        attention = K.conv1d(multi, self.W_O)
        if self.activation is not None:
            attention = call_activation(attention, self.activation)
        return attention



########################## EncoderBlock
class PositionEncoding(Layer):
    def __init__(self, **kwargs):
        super(PositionEncoding, self).__init__(**kwargs)

    def call(self, x):
        """
        Args:
            x: tensor of shape [b, length, dim]
        Returns:
            Tensor with the same shape, which is the sum of x and its position encoding.
        """

        length, d = tf.shape(x)[1:]
        pe = self.position_encoding(length, d, name=self.name)
        return x+pe

    def build(self, input_shape):
        super(PositionEncoding, self).build(input_shape)

    def position_encoding(self, length, d, name='pe'):
        arr = tf.expand_dims(tf.to_float(tf.range(d//2))*2., -1)              # (d/2, 1)
        arr = tf.pow(tf.constant(1.0e4), arr / tf.to_float(d))
        pos = tf.expand_dims(tf.to_float(tf.range(length)), 1)                # (length, 1)
        cycles = pos / tf.transpose(arr)                                                        # (length, 1) / (1, d/2) -> (length, d/2)
        concat = tf.concat([tf.expand_dims(tf.sin(cycles), -1), tf.expand_dims(tf.cos(cycles), -1)], axis=-1)       # (length, d/2, 2)
        pe = tf.reshape(concat, (1, length, d))                                                  # (length, d)
        return pe

    def compute_output_shape(self, input_shape):
        return input_shape


class LayerNormalization1D(Layer):
    def __init__(self, eps = K.epsilon(), regularizer=None, **kwargs):
        self.eps = eps
        self.regularizer = regularizer
        super(LayerNormalization1D, self).__init__(**kwargs)


    def build(self, input_shape):
        self.gain = self.add_weight(name='layer_norm_gain',
                                    shape=((input_shape[-1])),
                                    initializer=tf.ones_initializer(),
                                    regularizer=self.regularizer,
                                    trainable=True)
        self.bias = self.add_weight(name='layer_norm_bias',
                                    shape=((input_shape[-1])),
                                    initializer=tf.zeros_initializer(),
                                    regularizer=self.regularizer,
                                    trainable=True)
        super(LayerNormalization1D, self).build(input_shape)


    def call(self, x):
        """
        Normalize the layers.
        Ref: https://arxiv.org/pdf/1607.06450.pdf
        Args:
            inputs: [batch, steps, dim]
        Returns:
            outputs: [batch, steps, dim]
        """
        mean, variance = tf.nn.moments(x, [-1], keep_dims=True)
        norm = (x - mean) / tf.sqrt(variance + self.eps)  # add epilon to avoid "divide by 0"
        return self.gain * norm + self.bias

    def compute_output_shape(self, input_shape):
        return input_shape



class EncoderBlocks():
    """
    Implementation of One Encoder Block as in
    https://arxiv.org/pdf/1804.09541.pdf

    This class inspire the structure of Keras Layer. However, it does not derive from
    the Layer class to use Keras high-level layers, so that it can avoid weights
    management.
    """
    def __init__(self,
                 input_dim,
                 output_dim,
                 num_convs,
                 num_blocks,
                 cf,
                 preactivation='relu',
                 initializer='glorot_uniform',
                 regularizer=None,
                 dropout_rate=0.0,
                 name='encoder'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_convs = num_convs
        self.num_blocks = num_blocks
        self.cf = cf
        self.preactivation = preactivation
        self.initializer = initializer
        self.regularizer = regularizer
        self.dropout_rate = dropout_rate
        self.name = name

        self.layers = self._build_encoder_blocks(name=self.name)


    def __call__(self, x, mask=None):
        return self._call_encoder_blocks(x, mask)


    def _build_conv(self, name='conv'):
        layers = []
        layers.append(KL.Lambda(lambda x: tf.expand_dims(x, axis=2), name=name+'_expand'))
        layers.append(
            KL.DepthwiseConv2D(
            (self.cf.KERNEL_SIZE, 1),
            padding='same',
            depthwise_initializer=self.initializer,
            depthwise_regularizer=self.regularizer,
            name=name+'_conv2d'
            )
        )
        layers.append(KL.Lambda(lambda x: tf.squeeze(x, axis=2), name=name+'_squeeze'))
        return layers


    def _build_attention(self, name='attn'):
        dim = self.output_dim
        layers = []
        layers.append(KL.Conv1D(
            dim,
            kernel_size=1,
            activation=self.cf.FFN_ACTIVATION,
            kernel_initializer=self.initializer,
            kernel_regularizer=self.regularizer,
            name=name+'_dense_0'))
        layers.append(KL.Conv1D(
            dim,
            kernel_size=1,
            activation=self.cf.FFN_ACTIVATION,
            kernel_initializer=self.initializer,
            kernel_regularizer=self.regularizer,
            name=name+'_dense_1'))
        layers.append(MultiheadAttention(
            num_heads=self.cf.SELF_ATTN_HEADS,
            activation=self.cf.SELF_ATTN_ACTIVATION,
            dropout_rate=self.cf.SELF_ATTN_DROPOUT_RATE,
            regularizer=self.regularizer,
            name=name+'_multihead'
        ))
        return layers


    def _build_position_wise_feedforward(self, hidden_sizes=[2048, 512], name='ffn'):
        layers = []
        for i, filters in enumerate(hidden_sizes[:-1]):
            layers.append(KL.Conv1D(
                filters,
                kernel_size=1,
                activation=self.cf.FFN_ACTIVATION,
                kernel_initializer=self.initializer,
                kernel_regularizer=self.regularizer,
                name=name+'_'+str(i)))

        layers.append(KL.Conv1D(
            hidden_sizes[-1],
            kernel_size=1,
            activation=None,
            kernel_initializer=self.initializer,
            kernel_regularizer=self.regularizer, name=name+'_'+str(i+1)))     # No activation
        return layers


    def _build_residual_block(self, op, mask=None, name='resblock'):
        """
        Args:
            op: Basic operations of the block: 'conv', 'self-attention', 'ffn'
        """
        dim = self.output_dim
        layers = []
        layers.append(LayerNormalization1D(regularizer=self.regularizer, name=name+'_layernorm'))    # LayerNorm
        if self.dropout_rate > 0.0:
            layers.append(KL.Dropout(rate=self.dropout_rate, name=name+'_dropout1'))

        if op == 'conv':
            layers.extend(self._build_conv(name=name))
        elif op == 'self-attention':
            layers.extend(self._build_attention(name=name))
        else:  # 'ffn'
            layers.extend(self._build_position_wise_feedforward(
                hidden_sizes=[dim//4]+[dim],
                name=name+'_ffn'
            ))

        if self.dropout_rate > 0.0:
            layers.append(KL.Dropout(rate=self.dropout_rate, name=name+'_dropout2'))
        layers.append(KL.Add())

        return layers


    def _build_encoder_block(self, name='encoder'):
        layers = []
        dim = self.output_dim

        for i in range(self.num_convs):
            layers.extend(self._build_residual_block('conv', name=name+'_convblock_'+str(i)))

        layers.extend(self._build_residual_block('self-attention', name=name+'_attnblock'))
        layers.extend(self._build_residual_block('ffn', name=name+'_ffnblock'))
        return layers


    def _build_encoder_blocks(self, name='encoder'):
        layers = []
        if self.input_dim != self.output_dim:
            layers.append(KL.Conv1D(self.output_dim, 1, padding='same', kernel_regularizer=self.regularizer, name=name+'_projconv')) # Project 500-dim into 128-dim with Conv1D
        for i in range(self.num_blocks):
            layers.extend(self._build_encoder_block(name=name+f'_{i}'))
        return layers


    # Layer-calling functions
    def _call_conv(self, x, layer_it):
        x = next(layer_it)(x)  # Lambda
        x = next(layer_it)(x)  # DepthwiseConv2D
        x = next(layer_it)(x)  # squeeze
        return x

    def _call_attention(self, x, layer_it, mask=None):
        x = next(layer_it)(x)  # Dense1
        x = next(layer_it)(x)  # Dense2
        x = next(layer_it)([x, mask])  # MultiHeadAttention
        return x

    def _call_position_wise_feedforward(self, x, layer_it, hidden_sizes=[2048, 512]):
        for i in range(len(hidden_sizes[:-1])):
            x = next(layer_it)(x)  # Conv1D for hidden layers
        x = next(layer_it)(x)      # Conv1D for output
        return x


    def _call_residual_block(self, x, layer_it, op, mask=None):
        dim = self.output_dim
        x_out = x
        x_out = next(layer_it)(x_out)  # LayerNorm
        if self.dropout_rate > 0.0:
            x_out = next(layer_it)(x_out)

        if op == 'conv':
            x_out = self._call_conv(x_out, layer_it)
        elif op == 'self-attention':
            x_out = self._call_attention(x_out, layer_it, mask)
        else:  # 'ffn'
            x_out = self._call_position_wise_feedforward(x_out, layer_it, hidden_sizes=[dim//4]+[dim])

        if self.dropout_rate > 0.0:
            x_out = next(layer_it)(x_out)
        x = next(layer_it)([x, x_out])      # Add
        return x


    def _call_encoder_block(self, x, layer_it, mask=None):
        for k in range(self.num_convs):
            x = self._call_residual_block(x, layer_it, 'conv')
        x = self._call_residual_block(x, layer_it, 'self-attention', mask)
        x = self._call_residual_block(x, layer_it, 'ffn')
        return x


    def _call_encoder_blocks(self, x, mask=None):
        layer_it = iter(self.layers)

        if self.input_dim != self.output_dim:
            x = next(layer_it)(x)       # Project 500-dim into 128-dim with Conv1D
        for k in range(self.num_blocks):
            x = self._call_encoder_block(x, layer_it, mask)
        return x


############################# Context-Query attention
# Similarity matrix S
class SimilarityLayer(KL.Layer):
    def __init__(self, cf=None, regularizer=None, **kwargs):
        self.cf = cf
        self.regularizer = regularizer
        super(SimilarityLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.n = input_shape[0][-1]
        self.m = input_shape[1][-1]
        self.d = input_shape[0][1]
        self.kernel = self.add_weight(name='kernel',
                                      shape=(1, self.d * 3, 1),
                                      initializer='glorot_uniform',
                                      regularizer=self.regularizer,
                                      trainable=True)
        super(SimilarityLayer, self).build(input_shape)

    def call(self, inputs):
        """
        Args:
            inputs: [(None, d, n), (None, d, m)]  ([C, Q])
        """
        n, m = self.n, self.m
        C, Q = inputs
        C_cols = tf.split(C, [1]*n, axis=-1)
        C_cols_tile = [tf.tile(C_cols[i], [1,1,m]) for i in range(n)]  # repeat: col0, ...(m times)..., col0, col1, ..., col1, ..., col(n-1), ..., col(n-1)
        C_tile = tf.concat(C_cols_tile, axis=-1)
        Q_tile = tf.tile(Q, [1, 1, n])   # repeat: col0, col1, ..., col(m-1), ..., col0, col1, ..., col(m-1)  [n times]
        mul_CQ = tf.multiply(Q_tile, C_tile)
        concat_CQ = tf.concat([Q_tile, C_tile, mul_CQ], axis=1)
        K_concat_CQ = tf.transpose(concat_CQ, [0,2,1])
        f_qc = K.conv1d(K_concat_CQ, self.kernel, 1)
        S = tf.reshape(f_qc, [-1, n, m])
        return S

    def compute_output_shape(self, input_shape):
        return (None, self.n, self.m)



class SpanPredictor(KL.Layer):
    def __init__(self, cf, **kwargs):
        self.cf = cf
        super(SpanPredictor, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SpanPredictor, self).build(input_shape)

    def call(self, inputs):
        p1, p2 = inputs
        pp = tf.matmul(tf.expand_dims(p1, -1), tf.expand_dims(p2, 1))
        low_trig = tf.matrix_band_part(pp, 0, self.cf.ANSWER_LEN)  # mask the upper triangle -> only lower triangle has positive values
        start_idx = tf.argmax(tf.reduce_max(low_trig, axis=2), axis=1)
        end_idx = tf.argmax(tf.reduce_max(low_trig, axis=1), axis=1)
        start_idx = tf.cast(tf.reshape(start_idx, (-1, 1)), tf.float32)
        end_idx = tf.cast(tf.reshape(end_idx, (-1, 1)), tf.float32)
        return [start_idx, end_idx]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0][0],1), (input_shape[0][0],1)]



############################## google-research ##############################
VERY_NEGATIVE_NUMBER = -1e29
VERY_SMALL_NUMBER = 1e-29

class BiAttentionDCN(KL.Layer):
    """
    Reference from https://github.com/google-research/google-research/blob/master/qanet/squad_helper.py#L476
    """
    def __init__(self, **kwargs):
        super(BiAttentionDCN, self).__init__(**kwargs)

    def build(self, input_shape):
        d = input_shape[0][-1]
        self.w1 = self.add_weight(name='w1',
                                  shape=[d, 1],
                                  initializer=RandomUniform(),
                                  trainable=True)
        self.w2 = self.add_weight(name='w2',
                                  shape=[d, 1],
                                  initializer=RandomUniform(),
                                  trainable=True)
        self.w3 = self.add_weight(name='w3',
                                  shape=[1, 1, d],
                                  initializer=RandomUniform(),
                                  trainable=True)
        super(BiAttentionDCN, self).build(input_shape)


    def call(self, inputs):
        a, b, mask_a, mask_b = inputs
        logits = self.bi_attention_memory_efficient_dcn(a, b, mask_a, mask_b)
        return logits

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[1][1], input_shape[1][2]*4)

    def bi_attention_memory_efficient_dcn(self,
        a,
        b,
        mask_a=None,
        mask_b=None,  # pylint: disable=unused-argument
        activation='none',
        sim_func='trilinear'):
        """Applies biattention as in the DCN paper."""

        d = a.shape.as_list()[-1]
        if activation != 'none':
            with tf.variable_scope(activation):
                a = tf.nn.relu(a)
                b = tf.nn.relu(b)

        # logits = tf.convert_to_tensor(np.ones((d, d)))
        logits = tf.transpose(
            self.trilinear_memory_efficient(a, b, d), perm=[0, 2, 1])  # [bs,len_b,len_a]
        b2a = b2a_attention(logits, a, mask_a)

        a2b = a2b_attention_dcn(logits, b)
        x = tf.concat([b, b2a, b * b2a, b * a2b], 2)
        return x



    def trilinear_memory_efficient(self, a, b, d, use_activation=False):
      """W1a + W2b + aW3b."""
      n = tf.shape(a)[0]

      len_a = tf.shape(a)[1]
      len_b = tf.shape(b)[1]

      w1, w2, w3 = self.w1, self.w2, self.w3

      a_reshape = tf.reshape(a, [-1, d])  # [bs*len_a, d]
      b_reshape = tf.reshape(b, [-1, d])  # [bs*len_b, d]

      part_1 = tf.reshape(tf.matmul(a_reshape, w1), [n, len_a])  # [bs, len_a]
      part_1 = tf.tile(tf.expand_dims(part_1, 2),
                       [1, 1, len_b])  # [bs, len_a, len_b]

      part_2 = tf.reshape(tf.matmul(b_reshape, w2), [n, len_b])  # [bs, len_b]
      part_2 = tf.tile(tf.expand_dims(part_2, 1),
                       [1, len_a, 1])  # [bs, len_a, len_b]

      a_w3 = a * w3  # [bs, len_a, d]
      part_3 = tf.matmul(a_w3, tf.transpose(b, perm=[0, 2, 1]))  # [bs,len_a,len_b]

      ## return the unnormalized logits matrix : [bs,len_a,len_b]
      if use_activation:
          return tf.nn.relu(part_1 + part_2 + part_3)
      return part_1 + part_2 + part_3


def b2a_attention(logits, a, mask_a=None):
    """Context-to-query attention."""
    if len(mask_a.get_shape()) == 1:
        mask_a = tf.sequence_mask(mask_a, tf.shape(a)[1])
    if len(mask_a.get_shape()) == 2:
        mask_a = tf.expand_dims(mask_a, 1)
    logits = mask_logits(logits, mask_a)
    probabilities = tf.nn.softmax(logits)  # [bs,len_b,len_a]
    b2a = tf.matmul(probabilities, a)  # [bs, len_b, d]
    return b2a


def a2b_attention_dcn(logits, b):
    """Query-to-context attention."""
    prob1 = tf.nn.softmax(logits)  # [bs,len_b,len_a]
    prob2 = tf.nn.softmax(tf.transpose(logits, perm=[0, 2,
                                                   1]))  # [bs,len_a,len_b]
    a2b = tf.matmul(tf.matmul(prob1, prob2), b)  # [bs,len_b,d]
    return a2b



###################
class QANetCallback(Callback):
    """
    Control the training process of QANet as the following:
    1. Accumulate trainable weights and apply Exponential Moving Average
    2. Warm-up scheme at the first 1,000 steps

    There are some problems training with Keras Model.fit function:
    1. Querying model's trainable params is too slow (4x slow)
    2. The global loss rapidly increases after several hundreds of iterations

    Although I adapt the Keras Callback class, I train the model with train_on_batch function instead of fit function.
    """
    def __init__(self, decay=0.999, **kwargs):
        self.decay = decay
        self.step = 0
        super(QANetCallback, self).__init__(**kwargs)

    def set_model(self, model):
        self.model = model
        self.lr = K.get_value(model.optimizer.lr)
        self.ema_weights = {}
        for w in tqdm(model.trainable_weights):
            self.ema_weights[w.name] = K.get_value(w)


    def on_batch_end(self, batch, logs=None):
        # Calculate EMA weights
        for w in self.model.trainable_weights:
            self.ema_weights[w.name] = self.decay * self.ema_weights[w.name]  + (1-self.decay) * K.get_value(w)

        # Warm-up the first 1000 steps
        if self.step < 1000:
            self.step += 1
            lr = (self.lr / np.log(1000)) * np.log(self.step)
        else:
            lr = self.lr
        K.set_value(self.model.optimizer.lr, lr)


    def on_epoch_end(self, epoch, logs=None):
        # Use EMA weights instead
        for w in tqdm(self.model.trainable_weights):
            K.set_value(w, self.ema_weights[w.name])



class QANetModel(object):
    def __init__(self, cf, data_train: List[List]=None, data: SquadData=None):
        self.cf = cf
        self.data_train = data_train
        self.data = data
        self.keras_model = self._build()


    def _build(self):
        print('---- Building QANet model...')

        cf = self.cf
        word_vectors, char_vectors, train_ques_ids, X_train, y_train, val_ques_ids, X_valid, y_valid = self.data_train
        init = VarianceScaling(scale=1.0, mode='fan_in', distribution='normal')
        reg = l2(cf.REGULARIZER_L2_LAMBDA)

        # TODO: for inference, set 0 to the followings dropout rates
        word_embed_dropout = cf.WORD_EMBED_DROPOUT
        char_embed_dropout = cf.CHAR_EMBED_DROPOUT
        layer_dropout = cf.LAYER_DROPOUT_RATE

        ########## Input definition ##########
        contextw_inp = KL.Input(shape=(cf.CONTEXT_LEN, ), name='contextw_inp')
        queryw_inp = KL.Input(shape=(cf.QUERY_LEN, ), name='queryw_inp')
        contextc_inp = KL.Input(shape=(cf.CONTEXT_LEN, cf.WORD_LEN), name='contextc_inp')
        queryc_inp = KL.Input(shape=(cf.QUERY_LEN, cf.WORD_LEN), name='queryc_inp')

        c_mask = KL.Lambda(lambda x: tf.cast(tf.cast(x, tf.bool), tf.float32))(contextw_inp) # [bs, c_len]
        q_mask = KL.Lambda(lambda x: tf.cast(tf.cast(x, tf.bool), tf.float32))(queryw_inp)

        num_chars = len(char_vectors)

        ###################### 1. Input Embedding Layer
        # Word embedding
        word_emb_layer = word_embedding_graph(
            word_vectors,
            dropout_rate=word_embed_dropout,
            name='w_emb'
        )
        contextw_emb = word_emb_layer(contextw_inp)
        queryw_emb = word_emb_layer(queryw_inp)

        # Char embedding
        char_emb_layer = char_embedding_graph(
            num_chars,
            cf.CHAR_DIM,
            cf,
            dropout_rate=char_embed_dropout,
            regularizer=reg,
            name='c_emb')
        contextc_emb = char_emb_layer(contextc_inp)
        contextc_emb = KL.Lambda(lambda x: tf.reshape(x, (-1, cf.CONTEXT_LEN, cf.C_EMB_SIZE)), name='contextc_reshape')(contextc_emb)
        queryc_emb = char_emb_layer(queryc_inp)
        queryc_emb = KL.Lambda(lambda x: tf.reshape(x, (-1, cf.QUERY_LEN, cf.C_EMB_SIZE)), name='queryc_reshape')(queryc_emb)

        # Merge embedding
        context_emb = KL.Concatenate(name='concat_context_emb')([contextw_emb, contextc_emb])
        query_emb = KL.Concatenate(name='concat_query_emb')([queryw_emb, queryc_emb])
        if layer_dropout > 0.0:
            context_emb = KL.Dropout(rate=layer_dropout)(context_emb)
            query_emb = KL.Dropout(rate=layer_dropout)(query_emb)
        emb_dim = cf.W_EMB_SIZE + cf.C_EMB_SIZE
        emb = KL.Concatenate(axis=1)([context_emb, query_emb])

        # Highway network
        layers = highway_graph(emb_dim, num_layers=2, h_activation='relu', regularizer=reg)
        highway = highway_forward(emb, layers, dropout_rate=layer_dropout)
        context, query = KL.Lambda(lambda x: tf.split(x, [cf.CONTEXT_LEN, cf.QUERY_LEN], axis=1))(highway)


        ###################### 2. Embedding Encoder Layer
        encoder_blocks = EncoderBlocks(
            emb_dim,
            cf.D_MODEL,
            4,
            1,
            cf,
            initializer=init,
            regularizer=reg,
            dropout_rate=layer_dropout,
            name='encoder')
        context_enc = encoder_blocks(context, c_mask)
        query_enc = encoder_blocks(query, q_mask)


        ###################### 3. Context-Query Attention Layer
        bi_attention = BiAttentionDCN()
        cq_attn = bi_attention([query_enc, context_enc, q_mask, c_mask])


        ###################### 4. Model Encoder Layer
        model_encoder_layer = EncoderBlocks(
            cf.D_MODEL*4,
            cf.D_MODEL*4,
            2,
            7,
            cf,
            initializer=init,
            regularizer=reg,
            dropout_rate=layer_dropout,
            name='model_encoder')
        M0 = model_encoder_layer(cq_attn, c_mask)
        M1 = model_encoder_layer(M0, c_mask)
        M2 = model_encoder_layer(M1, c_mask)


        ###################### 5. Output Layer
        start_input = KL.Concatenate(axis=2)([M0, M1])  # left branch
        start_linear = KL.Conv1D(1, 1, kernel_initializer=init, kernel_regularizer=reg)(start_input)
        if layer_dropout > 0.0:
            start_linear = KL.Dropout(rate=layer_dropout)(start_linear)
        start_linear = KL.Lambda(lambda x: K.squeeze(x, axis=2))(start_linear)
        start_linear = Lambda(lambda x: mask_logits(x[0], x[1]))([start_linear, c_mask])
        p1 = KL.Softmax(axis=1)(start_linear)

        end_input = KL.Concatenate(axis=2)([M0, M2])  # right branch
        end_linear = KL.Conv1D(1, 1, kernel_initializer=init, kernel_regularizer=reg)(end_input)
        if layer_dropout > 0.0:
            end_linear = KL.Dropout(rate=layer_dropout)(end_linear)
        end_linear = KL.Lambda(lambda x: K.squeeze(x, axis=2))(end_linear)
        end_linear = Lambda(lambda x: mask_logits(x[0], x[1]))([end_linear, c_mask])
        p2 = KL.Softmax(axis=1)(end_linear)

        start, end = SpanPredictor(cf)([p1, p2])
        model = KM.Model(inputs=[contextw_inp, queryw_inp, contextc_inp, queryc_inp], outputs=[p1, p2, start, end])
        return model


    def summary(self):
        self.keras_model.summary()


    def compile(self):
        ####################### Compile model
        cf = self.cf
        optimizer = Adam(lr=cf.LEARNING_RATE, beta_1=cf.ADAM_B1, beta_2=cf.ADAM_B2, epsilon=cf.ADAM_EPS, clipnorm=5.)
        self.keras_model.compile(
            optimizer=optimizer,
            loss=['categorical_crossentropy', 'categorical_crossentropy', 'mae', 'mae'],
            loss_weights=[1, 1, 0, 0])


    def train(self, batch_size=4, epochs=25):
        cf = self.cf
        self.compile()
        model = self.keras_model
        word_vectors, char_vectors, train_ques_ids, X_train, y_train, val_ques_ids, X_valid, y_valid = self.data_train

        qanet_cb = QANetCallback(decay=cf.EMA_DECAY)
        tb = TensorBoard(log_dir=cf.TENSORBOARD_PATH,
                         histogram_freq=0,
                         write_graph=False,
                         write_images=False,
                         update_freq=cf.TENSORBOARD_UPDATE_FREQ)

        # Call set_model for all callbacks
        qanet_cb.set_model(model)
        tb.set_model(model)

        ep_list = []
        avg_train_loss_list = []
        em_score_list = []
        f1_score_list = []

        global_steps = 0
        gt_start_list, gt_end_list = y_valid[2:]
        for ep in range(1, epochs+1):   # Epoch num start from 1
            print('----------- Training for epoch {}...'.format(ep))
            # Train
            batch = 0
            sum_loss = 0
            num_batches = (len(X_train[0])-1) // batch_size + 1
            for X_batch, y_batch in get_batch(X_train, y_train, batch_size=batch_size, shuffle=True):
                batch_logs = {
                    'batch': batch,
                    'size': len(X_batch[0])
                }
                tb.on_batch_begin(batch, batch_logs)

                loss, loss_p1, loss_p2, loss_start, loss_end = model.train_on_batch(X_batch, y_batch)
                sum_loss += loss
                avg_loss = sum_loss / (batch + 1)
                print('Epoch: {}/{}, Batch: {}/{}, Accumulative average loss: {:.4f}, Loss: {:.4f}, Loss_P1: {:.4f}, Loss_P2: {:.4f}, Loss_start: {:.4f}, Loss_end: {:.4f}'.format(
                    ep, epochs, batch, num_batches, avg_loss, loss, loss_p1, loss_p2, loss_start, loss_end))
                batch_logs.update({
                    'loss': loss,
                    'loss_p1': loss_p1,
                    'loss_p2': loss_p2
                })
                qanet_cb.on_batch_end(batch, batch_logs)
                tb.on_batch_end(batch, batch_logs)

                global_steps += 1
                batch += 1


            ep_list.append(ep)
            avg_train_loss_list.append(avg_loss)

            print('Backing up temp weights...')
            model.save_weights(cf.TEMP_MODEL_PATH)
            qanet_cb.on_epoch_end(ep)     # Apply EMA weights
            model.save_weights(cf.MODEL_PATH % str(ep))

            print('----------- Validating for epoch {}...'.format(ep))
            valid_scores = self.validate(X_valid, y_valid, gt_start_list, gt_end_list, batch_size=cf.BATCH_SIZE)
            em_score_list.append(valid_scores['exact_match'])
            f1_score_list.append(valid_scores['f1'])
            print('------- Result of epoch: {}/{}, Average_train_loss: {:.6f}, EM: {:.4f}, F1: {:.4f}\n'.format(ep, epochs, avg_loss, valid_scores['exact_match'], valid_scores['f1']))

            tb.on_epoch_end(ep, {'f1': valid_scores['f1'], 'em': valid_scores['exact_match']})

            # Write result to CSV file
            result = pd.DataFrame({
                'epoch': ep_list,
                'avg_train_loss': avg_train_loss_list,
                'em': em_score_list,
                'f1': f1_score_list
            })
            result.to_csv(cf.RESULT_LOG, index=None)

            # Restore the original weights to continue training
            print('Restoring temp weights...')
            model.load_weights(cf.TEMP_MODEL_PATH)

        tb.on_train_end(None)


    def validate(self, X_valid, y_valid, gt_start_list, gt_end_list, batch_size=4):
        cf = self.cf
        model = self.keras_model

        # Evaluate on validation data
        pred_starts = []
        pred_ends = []
        for X_batch, y_batch in get_batch(X_valid, y_valid, batch_size=batch_size):
            p1, p2, starts, ends = model.predict_on_batch(X_batch)
            pred_starts.extend(np.squeeze(starts, axis=-1))
            pred_ends.extend(np.squeeze(ends, axis=-1))
        pred_starts = np.array(pred_starts).astype(np.int32)
        pred_ends = np.array(pred_ends).astype(np.int32)
        return SquadData.evaluate(gt_start_list, gt_end_list, pred_starts, pred_ends)


    def load_weights(self, filename):
        self.keras_model.load_weights(filename)


    def ask(self, context, query_text):
        vocab = self.data.vocab

        # Parse context
        raw_context = context
        context_toks = tokenize_long_text(context)
        context_toks = [t.strip(' ') for t in context_toks]
        context_chars = to_chars(context_toks, cf.WORD_LEN, cf.PAD_CHAR)
        contextw = vocab.vectorize(context_toks, cf.CONTEXT_LEN)
        contextc = vocab.vectorize_c(context_chars, cf.CONTEXT_LEN, cf.WORD_LEN)

        # Parse query
        q_toks = tokenize(query_text)
        queryw = vocab.vectorize(q_toks, cf.QUERY_LEN)
        question_chars = to_chars(q_toks, cf.WORD_LEN, cf.PAD_CHAR)
        queryc = vocab.vectorize_c(question_chars, cf.QUERY_LEN, cf.WORD_LEN)

        # Build input
        X_batch = [[np.array(contextw)], [np.array(queryw)], [np.array(contextc)], [np.array(queryc)]]

        # Predict
        p1, p2, starts, ends = self.keras_model.predict_on_batch(X_batch)
        start = int(np.squeeze(starts, axis=-1)[0])
        end = int(np.squeeze(ends, axis=-1)[0])
        answer = [context_toks[i] for i in range(start, end+1)]
        return answer
