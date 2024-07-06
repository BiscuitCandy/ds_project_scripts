
#   THIS CODE DOESNOT WORK IN NORMAL BL's
#   THIS CODE HAS SPECIFIC REQUIRED LIBRARRIES
#   KINDLY LOOK OUT FOR THOSE REQUIREMENTS


import gensim
import mysql.connector
from io import StringIO
import csv
import threading
import re
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

data_ = open("test_data.txt", 'w+')
hmd_ = open("test_hmd.txt", 'w+')
datarow_ = open("test_datarow.txt", 'w+')

def process_res(cell_value):
    cell_value = re.sub(r'\b0\.0\b|\b0\b', 'ZERO', cell_value)
    cell_value = re.sub(r'\d+(\.\d+)?\s*-\s*\d+(\.\d+)?', 'RANGE', cell_value)
    cell_value = re.sub(r'-\d+(\.\d+)?', 'NEG', cell_value)
    cell_value = re.sub(r'\d+(\.\d+)?%', 'PERCENT', cell_value)
    cell_value = re.sub(r'0(\.\d+)?', 'SMALLPOS', cell_value)
    cell_value = re.sub(r'\d{1,2}/\d{1,2}/\d{4}', 'DATE', cell_value)
    cell_value = re.sub(r'<', 'LESS', cell_value)
    cell_value = re.sub(r'>', 'GREATER', cell_value)
    cell_value = re.sub(r'~', 'APPROX', cell_value)
    cell_value = re.sub(r'/mg\b', 'UNITMG', cell_value)
    cell_value = re.sub(r'/kg\b', 'UNITKG', cell_value)
    cell_value = re.sub(r'/ml\b', 'UNITML', cell_value)
    cell_value = re.sub(r'/L\b', 'UNITL', cell_value)
    cell_value = re.sub(r'([\w]*(Year|year|time|times|Times|Time|days|Days)[\w]*)', 'UNITTIME', cell_value)
    cell_value = re.sub(r'[a-z|\_]*/ml', 'UNITML', cell_value)
    cell_value = re.sub(r'[a-z|\_]*/mg', 'UNITMG', cell_value)
    cell_value = re.sub(r'[a-z|\_]*/kg', 'UNITKG', cell_value)
    return cell_value


substitutions = {
    r'\'0.0\'|\'0\'': 'ZERO',
    r'\d+(\.\d+)?\s*\[\s*\d+(\.\d+)?\s*to\s*\d+(\.\d+)?\s*\]': 'RANGE',  # numerical ranges
    r'-\d+(\.\d+)?': 'NEG',  # negative numbers
    r'\d+(\.\d+)?%': 'PERCENT',  # percentages
    r'\d+(\.\d+)?': 'FLOAT',  # floating point numbers
    r'0(\.\d+)?': 'SMALLPOS',  # numbers between 0 and 1
    r'\d{1,2}/\d{1,2}/\d{4}': 'DATE',  # date format mm/dd/yyyy
    r'<': 'LESS',  # less than symbol
    r'>': 'GREATER',  # greater than symbol
    r'~': 'APPROX',  # approximately equal to symbol
    r'/mg\b': 'UNITMG',  # units
    r'/kg\b': 'UNITKG',
    r'/ml\b': 'UNITML',
    r'/L\b': 'UNITL',
    r'([\w]*(Year|year|time|times|Times|Time|days|Days)[\w]*)': 'UNITTIME',
    r'[a-z|\_]*/ml': 'UNITML',
    r'[a-z|\_]*/mg': 'UNITMG',
    r'[a-z|\_]*/kg': 'UNITKG',
}


def process_cell_value(cell_value):
    for pattern, subst in substitutions.items():
        cell_value = re.sub(pattern, subst, cell_value)
    cell_value = " ".join(cell_value.split())  # remove extra whitespaces
    return cell_value if cell_value else 'nan'

# Function to process each record
def process_record(record):
    # Load CSV from the BLOB
    csv_content = record[5].decode("utf-8")
    csv_file = StringIO(csv_content)
    csv_reader = list(csv.reader(csv_file, delimiter='|'))

    hmd_count = record[3]

    # Convert csv_reader to list and find rows with the specified labels
    csv_list = list(csv_reader)
    hmd_rows = [row for row in csv_list if row[-1] == "HMD"]
    d_rows = [row for row in csv_list if row[-1] == "D"]

    if hmd_count == 1:
        hmdrow = hmd_rows[0][1]
        datarow = random.choice(d_rows)[1]
        p_hmdrow = process_row(hmdrow)
        p_datarow = process_row(datarow)

        print("*", "data", "*", p_datarow)
        data_.write(" ".join(p_datarow) + "\n")
        print("*", "hmd", "*", p_hmdrow)
        hmd_.write(" ".join(p_hmdrow) + "\n")

    elif hmd_count == 2:
        hmdrow1 = hmd_rows[0][1]
        hmdrow2 = hmd_rows[1][1]
        datarow = random.choice(d_rows)[1]
        print("******", datarow)
        datarow_.write(datarow + "\n")



def process_row(row):
    row = row.replace("[", "")
    row = row.replace("]", "")
    row = row.replace("(", "")
    row = row.replace(")", "")
    row = row.replace("'", "")
    terms = []
    for term_group in row.split(','):
        for term in term_group.split():
            processed_term = re.sub(r'[^\w]', '', process_cell_value(term)).lower()
            terms.append(processed_term)
    return terms

# Connect to the MySQL database
def fetch_records():
    config = {
        'user': 'root',
        'password': '',
        'host': 'localhost',
        'database': 'test',
    }

    # Connect to the database
    cnx = mysql.connector.connect(**config)
    cursor = cnx.cursor()

    # Fetch records from cl_intermediate_data
    cursor.execute("SELECT * FROM cl_intermediate_data LIMIT 10")
    records = cursor.fetchall()
     
    # Process each record using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(process_record, records)

    # Close the database connection
    cursor.close()
    cnx.close()


if __name__ == "__main__":
    fetch_records()
