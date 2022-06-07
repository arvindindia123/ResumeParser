import docx2txt
import re

"""
extract_text_from_doc function extracts the text of the given resume
Library is developed as part of research project in Usha Martin University

"""


def extract_text_from_doc(doc_path):
    print('Reading doc/docx file')
    temp = docx2txt.process("Smith Resume.docx")
    text = [line.replace('\t', ' ') for line in temp.split('\n') if line]
    return ' '.join(text)


resumeContent = extract_text_from_doc('')
print(resumeContent)



