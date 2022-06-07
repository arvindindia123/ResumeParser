import spacy
import pickle
import random
nlp=spacy.load("en_core_web_sm")
# doc = nlp("Indians spent over $71 billion on clothes in 2018")
#
# for ent in doc.ents:
#     print(ent.text, ent.label_)

with open('train_data.pkl','rb') as f:
    text=f.read()
print(text)
# doc=nlp(text)
#
# print(doc)