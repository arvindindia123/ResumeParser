import spacy
import pickle
import random
from spacy.training.example import Example

train_data=pickle.load(open('train_data.pkl','rb'))
# print(train_data)
nlp=spacy.blank('en')
def train_model(train_data):
    if 'ner' not in nlp.pipe_names:
        ner=nlp.create_pipe('ner')
        # nlp.add_pipe(ner',last=True)
        nlp.add_pipe('ner', last= True)



    for _,annotation in train_data:
        for ent in annotation['entities']:
            ner.add_label(ent[2])

    other_pipes= [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(* other_pipes):
        optimizer=nlp.begin_training()
        for itn in range(10):
            print('starting iteration')
            random.shuffle(train_data)
            losses={}
            index=0
            for text, annotation in train_data:
                try:
                    nlp.update(
                        [text],
                        [annotation],
                        drop=0.2,
                        sgd=optimizer,
                        losses=losses)
                except Exception as e:
                    print(e)
                    pass
            print(losses)