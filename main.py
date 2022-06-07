# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import PyPDF2
import spacy
import pickle
import random
def print_hi(name):

    train_data=pickle.load(open('train_data.pkl','rb'))
    print(train_data)
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
                        pass
                    print(text)
                print(losses)

    train_model(train_data)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
   print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
