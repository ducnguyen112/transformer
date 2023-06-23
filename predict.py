import pickle
file = open("./vocab_folder/en_tokenizer.pickle", 'rb')
en_tokenizer = pickle.load(file)
a = en_tokenizer.texts_to_sequences("i love you")
print(a)
