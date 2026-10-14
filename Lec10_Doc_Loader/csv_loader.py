from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path ="Social_Network_Ads.csv")

# docs = loader.load()

# print(len(docs))
# print(docs[0])

docs = loader.lazy_load()

for document in docs:
    print(document)



# Why file_path works but Path gives an error
# In this call:
# loader = CSVLoader(file_path="Social_Network_Ads.csv")
# file_path is the parameter name defined by CSVLoader.
# If you wrote:
# loader = CSVLoader(Path="Social_Network_Ads.csv")
# it fails because Path is not a supported parameter name. Python keyword names must match exactly.
# The Path class is a separate concept—it represents filesystem paths. It doesn’t replace the keyword file_path.