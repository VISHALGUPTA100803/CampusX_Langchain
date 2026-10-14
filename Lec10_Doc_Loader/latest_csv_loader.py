import csv
from langchain_core.documents import Document

file_path = "Social_Network_Ads.csv"
docs = []

with open(file_path, encoding="utf-8", newline="") as file:
    reader = csv.DictReader(file)

    for row_number, row in enumerate(reader):
        document = Document(
            page_content="\n".join(
                f"{column}: {value}" for column, value in row.items()
            ),
            metadata={
                "source": file_path,
                "row": row_number
            }
        )
        docs.append(document)

print(len(docs))
print(docs[0]) 

# 1. Read each CSV record as a dictionary
# reader = csv.DictReader(file)
# DictReader uses the first row as dictionary keys and each following row as their values.
# For this CSV:
# Gender,Age,Purchased
# Male,19,0
# Female,25,1
# It produces rows like:
# {"Gender": "Male", "Age": "19", "Purchased": "0"}
# {"Gender": "Female", "Age": "25", "Purchased": "1"}
# Notice that these values are strings, including "19" and "0".
# 2. Get the row number and row together
# for row_number, row in enumerate(reader):
# enumerate() adds a counter starting at 0:
# Iteration	row_number	row
# First	0	{"Gender": "Male", "Age": "19", "Purchased": "0"}
# Second	1	{"Gender": "Female", "Age": "25", "Purchased": "1"}


# 3. Turn each dictionary into text
# page_content="\n".join(
#     f"{column}: {value}" for column, value in row.items()
# )
# There are three parts.
# row.items() gives the key-value pairs:
# ("Gender", "Male")
# ("Age", "19")
# ("Purchased", "0")
# The f-string formats each pair:
# f"{column}: {value}"
# This produces:
# Gender: Male
# Age: 19
# Purchased: 0
# "\n".join(...) combines those strings with a newline between them:
# "Gender: Male\nAge: 19\nPurchased: 0"
# That combined string becomes the document’s page_content.