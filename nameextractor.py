import spacy
from spacy.matcher import Matcher
import docfilereader as docr

"""Load a spaCy model from an installed package or a local path.
extract_candidate_name function extracts the name of a candidate from
the resume content passed to it
Library is developed as part of research project in Usha Martin University

"""

# load pre-trained model
nlp = spacy.load('en_core_web_sm')
# initialize matcher with a vocab
matcher = Matcher(nlp.vocab)


def extract_candidate_name(resume_text):
    print(resume_text)
    nlp_text = nlp(resume_text)
    # First name and Last name are always Proper Nouns
    pattern = [{'POS': 'PROPN'}, {'POS': 'PROPN'}]

    matcher.add('NAME', [pattern])
    matches = matcher(nlp_text, as_spans=True)
    print(matches[0])
    return matches[0]


resume_content = docr.extract_text_from_doc('')
candidate_name = extract_candidate_name(resume_content)
print("Candidate Name : "+str(candidate_name))


#
# # print(resumeContent)
# phone=""
# def extract_mobile_number(text):
#     print(resumeContent)
#     phone = re.findall(re.compile(
#         r'(?:(?:\+?([1-9]|[0-9][0-9]|[0-9][0-9][0-9])\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([0-9][1-9]|[0-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?([0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(\d+))?'),
#                        text)
#     if phone:
#         number = ''.join(phone[0])
#         if len(number) > 10:
#             return '+' + number
#         else:
#             return number
#
# num=extract_mobile_number(resumeContent)
# print('Phone Number'+num)
#
# def extract_email(email):
#     email = re.findall("([^@|\s]+@[^@]+\.[^@|\s]+)", email)
#     if email:
#         try:
#             return email[0].split()[0].strip(';')
#         except IndexError:
#             return None
# email=extract_email(resumeContent)
# print(email)
