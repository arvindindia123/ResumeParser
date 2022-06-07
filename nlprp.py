# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import PyPDF2
import spacy
import pickle
import random
import train_model

train_data=pickle.load(open('train_data.pkl','rb'))
nlp=spacy.blank('en')

train_model.train_model(train_data)
nlp.to_disk('nlp_model')
nlp_model=spacy.load('nlp_model')
print(train_data[0][0])

doc=nlp_model(train_data[0][0])

print('here')
for ent in doc.ents:
    print(f'{ent.label_upper():{30}}- {ent.text}')

