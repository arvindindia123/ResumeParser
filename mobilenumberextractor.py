import re
import docfilereader as docr

"""
extract_mobile_number function extracts the mobile number of the given candidate 
from the given resume
Library is developed as part of research project in Usha Martin University

"""

mobile_no = ""


def extract_mobile_number(resume):

    mobile_no = re.findall(re.compile(
        r'(?:(?:\+?([1-9]|[0-9][0-9]|[0-9][0-9][0-9])\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([0-9][1-9]|[0-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?([0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(\d+))?'),
        resume)
    if mobile_no:
        temp_number = ''.join(mobile_no[0])
        if len(temp_number) > 10:
            return '+' + temp_number
        else:
            return temp_number


resume_content = docr.extract_text_from_doc(".")
num = extract_mobile_number(resume_content)
print('Mobile Number : ' + num)
