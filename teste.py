#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seq2Seq (Encoder-Decoder) com Atenção em Keras/TensorFlow
---------------------------------------------------------
- Implementa um modelo Encoder-Decoder com Atenção (estilo Luong).
- Inclui pipeline mínimo com TextVectorization, dataset toy e laço de treino.
- Mostra como montar modelos de inferência para decodificação passo a passo (greedy).

Requisitos:
  pip install tensorflow>=2.12

Execução:
  python seq2seq_attention_keras.py
"""

import tensorflow as tf
import keras
from keras import layers
import numpy as np

# ============================================================
# 1) Dados toy (exemplo didático)
#    Você deve substituir por seus próprios pares (src -> tgt)
# ============================================================
pairs = [
    ("1 2 3", "um dois tres"),
    ("2 3 4", "dois tres quatro"),
    ("3 4 5", "tres quatro cinco"),
    ("4 5 6", "quatro cinco seis"),
    ("5 6 7", "cinco seis sete"),
    ("6 7 8", "seis sete oito"),
    ("7 8 9", "sete oito nove"),
    ("8 9 10", "oito nove dez"),
]

# Tokens especiais no destino
src_texts = [s for s, _ in pairs]
tgt_texts_in = [f"<start> {t}" for _, t in pairs]
tgt_texts_out = [f"{t} <end>" for _, t in pairs]

# ============================================================
# 2) Vetorização
# ============================================================
src_vectorizer = layers.TextVectorization(
    standardize="lower_and_strip_punctuation",
    split="whitespace",
    max_tokens=1000,
    output_mode="int",
    output_sequence_length=8,
)
tgt_vectorizer = layers.TextVectorization(
    standardize="lower_and_strip_punctuation",
    split="whitespace",
    max_tokens=1000,
    output_mode="int",
    output_sequence_length=8,
)

src_vectorizer.adapt(src_texts)
tgt_vectorizer.adapt(tgt_texts_in + tgt_texts_out)

src_vocab = src_vectorizer.get_vocabulary()
tgt_vocab = tgt_vectorizer.get_vocabulary()
src_vocab_size = len(src_vocab)
tgt_vocab_size = len(tgt_vocab)

index_to_tgt = np.array(tgt_vocab)
tgt_to_index = {w: i for i, w in enumerate(tgt_vocab)}
start_id = tgt_to_index.get("<start>", 1)
end_id = tgt_to_index.get("<end>", 2)

X_src = src_vectorizer(tf.constant(src_texts))
X_tgt_in = tgt_vectorizer(tf.constant(tgt_texts_in))
X_tgt_out = tgt_vectorizer(tf.constant(tgt_texts_out))

train_ds = tf.data.Dataset.from_tensor_slices(
    ({"encoder_inputs": X_src, "decoder_inputs": X_tgt_in}, X_tgt_out)
).shuffle(32).batch(4)

# ============================================================
# 3) Modelo Encoder-Decoder com Atenção
# ============================================================
embed_dim = 64
enc_units = 128
dec_units = 128

# ENCODER
enc_inputs = keras.Input(shape=(8,), dtype="int32", name="encoder_inputs")
enc_emb = layers.Embedding(input_dim=src_vocab_size, output_dim=embed_dim, mask_zero=True)(enc_inputs)
enc_outputs, state_h, state_c = layers.LSTM(enc_units, return_sequences=True, return_state=True)(enc_emb)

# DECODER
dec_inputs = keras.Input(shape=(8,), dtype="int32", name="decoder_inputs")
dec_emb = layers.Embedding(input_dim=tgt_vocab_size, output_dim=embed_dim, mask_zero=True)(dec_inputs)
dec_outputs, _, _ = layers.LSTM(dec_units, return_sequences=True, return_state=True)(dec_emb, initial_state=[state_h, state_c])

# Atenção
attn = layers.Attention()([dec_outputs, enc_outputs])
dec_concat = layers.Concatenate(axis=-1)([dec_outputs, attn])

outputs = layers.TimeDistributed(layers.Dense(tgt_vocab_size, activation="softmax"))(dec_concat)

model = keras.Model(inputs=[enc_inputs, dec_inputs], outputs=outputs)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# ============================================================
# 4) Treino
# ============================================================
model.fit(train_ds, epochs=300, verbose=0)
print("Treino concluído.")

# ============================================================
# 5) Inferência (decodificação greedy passo a passo)
# ============================================================
encoder_model = keras.Model(enc_inputs, [enc_outputs, state_h, state_c])

dec_input_token = keras.Input(shape=(1,), dtype="int32")
prev_state_h = keras.Input(shape=(dec_units,))
prev_state_c = keras.Input(shape=(dec_units,))
enc_outputs_infer = keras.Input(shape=(8, enc_units))

dec_emb_layer = model.layers[4]
dec_lstm_layer = model.layers[5]
attn_layer = model.layers[6]
td_layer = model.layers[8]

x = dec_emb_layer(dec_input_token)
dec_out, state_h_new, state_c_new = dec_lstm_layer(x, initial_state=[prev_state_h, prev_state_c])
context = attn_layer([dec_out, enc_outputs_infer])
concat = layers.Concatenate(axis=-1)([dec_out, context])
token_probs = td_layer(concat)
token_probs = layers.Lambda(lambda t: t[:, 0, :])(token_probs)

decoder_step_model = keras.Model(
    [dec_input_token, prev_state_h, prev_state_c, enc_outputs_infer],
    [token_probs, state_h_new, state_c_new]
)

def decode_greedy(input_text, max_len=20):
    src_vec = src_vectorizer([input_text])
    enc_outs, h, c = encoder_model.predict(src_vec, verbose=0)
    state_h_t, state_c_t = h, c
    current_token = np.array([[start_id]])
    out_tokens = []
    for _ in range(max_len):
        token_probs, state_h_t, state_c_t = decoder_step_model.predict(
            [current_token, state_h_t, state_c_t, enc_outs], verbose=0
        )
        next_id = int(np.argmax(token_probs[0]))
        if next_id == end_id:
            break
        out_tokens.append(index_to_tgt[next_id])
        current_token = np.array([[next_id]])
    return " ".join(out_tokens)

tests = ["1 2 3", "7 8 9", "3 4 5"]
for t in tests:
    print(f"{t} -> {decode_greedy(t)}")
