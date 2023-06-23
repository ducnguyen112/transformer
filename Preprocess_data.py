from datasets import load_dataset
import re
from sklearn.model_selection import train_test_split
from numba_pre_data import ContainerHolder
import tensorflow as tf
import os
import pickle

class En_Vi_Dataset:
    def __init__(self, inp_lang, targ_lang, vocab_folder):
        self.inp_lang = inp_lang
        self.targ_lang = targ_lang
        
        self.vocab_folder = vocab_folder
        self.inp_tokenizer_path = '{}{}_tokenizer.pickle'.format(self.vocab_folder, self.inp_lang)
        self.targ_tokenizer_path = '{}{}_tokenizer.pickle'.format(self.vocab_folder, self.targ_lang)
        
        self.inp_tokenizer = None
        self.targ_tokenizer = None
    def split_data(self, type_data):
        container = ContainerHolder()
        len_data = len(type_data)
        for i in range(1, len_data):
            x = self.preprocess_sentence(type_data[i]['en'],62)
            y = self.preprocess_sentence(type_data[i]['vi'],62)
            container.input_data.append(x)
            container.target_data.append(y)
            if i%20 == 0:
                print('---Processing {:0.2f}'.format(i*100/len_data), '% ---' )
        return list(container.input_data), list(container.target_data)
    def build_tokenizer(self, lang):
        # # TODO: Update document
        # if not lang_tokenizer:
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')

        lang_tokenizer.fit_on_texts(lang)
        return lang_tokenizer
    def tokenize(self, lang_tokenizer, lang, max_length):
        # TODO: Update document
        # Padding
        tensor = lang_tokenizer.texts_to_sequences(lang)
        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post', maxlen=max_length)
        return tensor
    def preprocess_sentence(self, w, max_length):
        BOS = '<start>'
        EOS = '<end>'
        UTF_8 = 'UTF-8'
        w = w.lower().strip()
        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)
        w = w.strip()

        # Truncate Length up to ideal_length
        w = " ".join(w.split()[:max_length+1])
        # Add start and end token
        w = '{} {} {}'.format(BOS, w, EOS)
        return w
    def load_dataset(self, num_examples):
        train, test, val = load_dataset("mt_eng_vietnamese", "iwslt2015-vi-en", split=['train', 'test', 'validation'])

        train = train['translation'][:20001]
        test = test['translation']
        val = val['translation']

        print(train[:num_examples])

        en_train_dataset, vi_train_dataset = self.split_data(train)
        # en_test_dataset, vi_test_dataset = self.split_data(test)
        # en_val_dataset, vi_val_dataset = self.split_data(val)


        self.inp_tokenizer = self.build_tokenizer(en_train_dataset)
        self.targ_tokenizer = self.build_tokenizer(vi_train_dataset)

        inp_tensor = self.tokenize(self.inp_tokenizer, en_train_dataset, 62)
        targ_tensor = self.tokenize(self.targ_tokenizer, vi_train_dataset, 62)

        # Saving Tokenizer
        print('=============Saving Tokenizer================')
        print('Begin...')

        if not os.path.exists(self.vocab_folder):
            try:
                os.makedirs(self.vocab_folder)
            except OSError as e: 
                raise IOError("Failed to create folders")

        with open(self.inp_tokenizer_path, 'wb') as handle:
            pickle.dump(self.inp_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(self.targ_tokenizer_path, 'wb') as handle:
            pickle.dump(self.targ_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('Done!!!')

        return inp_tensor, targ_tensor

    def build_dataset(self, buffer_size, batch_size, num_examples):
        inp_tensor, targ_tensor = self.load_dataset(num_examples)

        inp_tensor_train, inp_tensor_val, targ_tensor_train, targ_tensor_val = train_test_split(inp_tensor, targ_tensor, test_size=0.2)

        train_dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(inp_tensor_train, dtype=tf.int64), tf.convert_to_tensor(targ_tensor_train, dtype=tf.int64)))

        train_dataset = train_dataset.cache().shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(inp_tensor_val, dtype=tf.int64), tf.convert_to_tensor(targ_tensor_val, dtype=tf.int64)))

        val_dataset = val_dataset.cache().shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return train_dataset, val_dataset
