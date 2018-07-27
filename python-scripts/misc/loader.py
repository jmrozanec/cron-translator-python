from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
from random import shuffle
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

separator="|"

lines = open('dataset_en.psv').read().split('\n')
shuffle(lines)
lines = [x for x in lines if x]

source_sentences = []
target_sentences = []
source_chars = set()
target_chars = set()
nb_samples = 250000

# Process english and french sentences
for line in range(nb_samples):
    if len(lines[line])!=0:
        source_line = str(lines[line]).split(separator)[0]
        # Append '\t' for start of the sentence and '\n' to signify end of the sentence
        target_line = "\t " + str(lines[line]).split(separator)[1] + '\n'
        source_sentences.append(source_line)
        target_sentences.append(target_line)
    	for ch in source_line.split(" "):
            if (ch not in source_chars):
                source_chars.add("{} ".format(ch))            
        for ch in target_line.split(" "):
            if (ch not in target_chars):
                target_chars.add("{} ".format(ch))

target_chars = sorted(list(target_chars))
source_chars = sorted(list(source_chars))

# dictionary to index each english character - key is index and value is english character
source_index_to_char_dict = {}

# dictionary to get english character given its index - key is english character and value is index
source_char_to_index_dict = {}

for k, v in enumerate(source_chars):
    source_index_to_char_dict[k] = v
    source_char_to_index_dict[v] = k

# dictionary to index each french character - key is index and value is french character
target_index_to_char_dict = {}

# dictionary to get french character given its index - key is french character and value is index
target_char_to_index_dict = {}
for k, v in enumerate(target_chars):
    target_index_to_char_dict[k] = v
    target_char_to_index_dict[v] = k

max_len_source_sent = max([len(line) for line in source_sentences])
max_len_target_sent = max([len(line) for line in target_sentences])

tokenized_source_sentences = np.zeros(shape = (nb_samples,max_len_source_sent,len(source_chars)), dtype='float32')
tokenized_target_sentences = np.zeros(shape = (nb_samples,max_len_target_sent,len(target_chars)), dtype='float32')
target_data = np.zeros((nb_samples, max_len_target_sent, len(target_chars)), dtype='float32')

# Vectorize the english and french sentences
for i in range(nb_samples):
    for k,ch in enumerate(source_sentences[i].split(" ")):
        tokenized_source_sentences[i,k,source_char_to_index_dict["{} ".format(ch)]] = 1
    for k,ch in enumerate(target_sentences[i].split(" ")):
        tokenized_target_sentences[i,k,target_char_to_index_dict["{} ".format(ch)]] = 1
        # decoder_target_data will be ahead by one timestep and will not include the start character.
        if k > 0:
            target_data[i,k-1,target_char_to_index_dict["{} ".format(ch)]] = 1

# Encoder model
encoder_input = Input(shape=(None, len(source_chars)))
encoder_LSTM = LSTM(256, return_state = True)
encoder_outputs, encoder_h, encoder_c = encoder_LSTM(encoder_input)
encoder_states = [encoder_h, encoder_c]

# Decoder model
decoder_input = Input(shape=(None,len(target_chars)))
decoder_LSTM = LSTM(256,return_sequences=True, return_state = True)
decoder_out, _ , _ = decoder_LSTM(decoder_input, initial_state=encoder_states)
decoder_dense = Dense(len(target_chars), activation='softmax')
decoder_out = decoder_dense(decoder_out)

model = Model(inputs=[encoder_input, decoder_input], outputs=[decoder_out])

# Run training
model.compile(optimizer='adam', loss='categorical_crossentropy')
checkpoint = ModelCheckpoint("{}-best.h5".format("translator"), monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
#model.fit(x=[tokenized_source_sentences,tokenized_target_sentences], y=target_data, batch_size=64, callbacks=callbacks_list, epochs=4, validation_split=0.2)
model.load_weights("{}-best.h5".format("translator"))
# Inference models for testing

# Encoder inference model
encoder_model_inf = Model(encoder_input, encoder_states)

# Decoder inference model
decoder_state_input_h = Input(shape=(256,))
decoder_state_input_c = Input(shape=(256,))
decoder_input_states = [decoder_state_input_h, decoder_state_input_c]

decoder_out, decoder_h, decoder_c = decoder_LSTM(decoder_input, initial_state=decoder_input_states)
decoder_states = [decoder_h , decoder_c]
decoder_out = decoder_dense(decoder_out)
decoder_model_inf = Model(inputs=[decoder_input] + decoder_input_states, outputs=[decoder_out] + decoder_states )

def decode_seq(inp_seq):
    # Initial states value is coming from the encoder 
    states_val = encoder_model_inf.predict(inp_seq)
    
    target_seq = np.zeros((1, 1, len(target_chars)))
    target_seq[0, 0, target_char_to_index_dict["\t "]] = 1
    
    translated_sent = ''
    stop_condition = False
    
    while not stop_condition:
        decoder_out, decoder_h, decoder_c = decoder_model_inf.predict(x=[target_seq] + states_val)
        
        max_val_index = np.argmax(decoder_out[0,-1,:])
        sampled_target_char = target_index_to_char_dict[max_val_index]
        translated_sent += sampled_target_char
        
        if ( (sampled_target_char == '\n') or (len(translated_sent) > max_len_target_sent)) :
            stop_condition = True
        
        target_seq = np.zeros((1, 1, len(target_chars)))
        target_seq[0, 0, max_val_index] = 1
        
        states_val = [decoder_h, decoder_c]
    return translated_sent

for seq_index in range(10):
    inp_seq = tokenized_source_sentences[seq_index:seq_index+1]
    translated_sent = decode_seq(inp_seq)
    print('-')
    print('Input sentence:', source_sentences[seq_index])
    print('Decoded sentence:', translated_sent)

