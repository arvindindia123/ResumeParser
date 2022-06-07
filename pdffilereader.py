import sys, fitz

"""
pip install PyMuPDF
This PyMuPDF library works better than PyPDF2 , tested with all types of PDF
extract_text_from_doc function extracts the text of the given resume
Library is developed as part of research project in Usha Martin University

"""


def read_text_from_pdf(filename):
    doc = fitz.open(filename)
    text = ""
    for page in doc:
        text = text + str(page.get_text())

    tx = " ".join(text.split('\n'))
    return tx


resume_content = read_text_from_pdf('Resume-Arvind.pdf')
print(resume_content)
