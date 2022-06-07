import re
import spacy
import nameextractor as ne
from nltk.corpus import stopwords
import docfilereader as docr

"""Load a spaCy model from an installed package or a local path.
extract_candidate_education function extracts the education of a candidate from
the resume connect passed to it
Library is developed as part of research project in Usha Martin University
   
"""

# load pre-trained model
nlp = spacy.load('en_core_web_sm')
# Grad all general stop words
STOPWORDS = set(stopwords.words('english'))
# Education Degrees
CANDIDATE_EDUCATION = [
            'P.hD','PHD','P.H.D',
            'BE','B.E.', 'B.E', 'BS', 'B.S',
            'ME', 'M.E', 'M.E.', 'MS', 'M.S',
            'BTECH', 'B.TECH', 'M.TECH', 'MTECH',
            'SSC', 'HSC', 'CBSE', 'ICSE', 'X', 'XII'
        ]
def extract_candidate_education(resume_text):
    nlp_text = nlp(resume_text)
    # Sentence Tokenizer
    nlp_text.sents

    nlp_text = [sent.text.strip() for sent in nlp_text.sents]
    edu = {}
    # Extract education degree
    for index, text in enumerate(nlp_text):
        for tex in text.split():
            # Replace all special symbols
            tex = re.sub(r'[?|$|.|!|,]', r'', tex)
            if tex.upper() in CANDIDATE_EDUCATION and tex not in STOPWORDS:
                edu[tex] = text + nlp_text[index]
    # Extract year[19]
    education = []
    for key in edu.keys():
        year = re.search(re.compile(r'(((20|19)(\d{2})))'), edu[key])
        if year:
            education.append((key, ''.join(year[0])))
        else:
            education.append(key)
    return education


resume_content = docr.extract_text_from_doc('.')
education = extract_candidate_education(resume_content)
print("Education of the Candidate : "+str(education))
