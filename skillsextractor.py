import pandas as pd
import nameextractor as ne
import docfilereader as docr
import spacy

"""Load a spaCy model from an installed package or a local path.
extract_candidate_name function extracts the name of a candidate from
the resume content passed to it
Library is developed as part of research project in Usha Martin University

"""

# pre-trained model
nlp = spacy.load('en_core_web_sm')


def extract_candidate_skills(resume_text):
    nlp_text = nlp(resume_text)
    noun_chunks = nlp_text.noun_chunks
    # removing stop word, implement tokens
    tokens = [token.text for token in nlp_text if not token.is_stop]
    print(tokens)
    #
    # reading the csv file
    data = pd.read_csv("allskills.csv")
    # extract values
    candidate_skills = list(data.columns.values)
    skillset = []
    #     # check for one-grams (example: python)
    #
    #
    for token in tokens:
        if token.lower() in candidate_skills:
            skillset.append(token)

    for token in noun_chunks:
        token = token.text.lower().strip()
        if token in candidate_skills:
            skillset.append(token)

    return [i.capitalize() for i in set([i.lower() for i in skillset])]


resume_content = docr.extract_text_from_doc('')
skills = extract_candidate_skills(resume_content)
print("Skills of the candidate : " + str(skills))
