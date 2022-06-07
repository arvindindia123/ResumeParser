import re
import docfilereader as docr

"""
extract_email function extracts the email of the given candidate from the resume
Library is developed as part of research project in Usha Martin University

"""


def extract_email(resume):
    email_content = re.findall("([^@|\s]+@[^@]+\.[^@|\s]+)", resume)
    if email_content:
        try:
            return email_content[0].split()[0].strip(';')
        except IndexError:
            return None


resume_content = docr.extract_text_from_doc(".")
email = extract_email(resume_content)
print("Email Address extracted:"+email)
