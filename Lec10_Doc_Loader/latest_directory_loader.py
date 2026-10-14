from pathlib import Path
from pypdf import PdfReader
from langchain_core.documents import Document

folder_path = Path("Books")
#docs = []

# for pdf_path in sorted(folder_path.glob("*.pdf")):
#     reader = PdfReader(pdf_path)

#     for page_number, page in enumerate(reader.pages):
#         document = Document(
#             page_content=page.extract_text() or "",
#             metadata={
#                 "source": str(pdf_path),
#                 "page": page_number,
#                 "total_pages": len(reader.pages)
#             }
#         )
#         docs.append(document)

# if docs:
#     print(docs[0])

# print(f"length {len(docs)} type {type(docs)}") 

#   Lazy Load

def lazy_load():
    for pdf_path in sorted(folder_path.glob("*.pdf")):
        reader = PdfReader(pdf_path)
    

    for page_number, page in enumerate(reader.pages):
        document = Document(
            page_content=page.extract_text() or "",
            metadata={
                "source": str(pdf_path),
                "page": page_number,
                "total_pages": len(reader.pages)
            }
        )
        yield document 

docs = lazy_load()

for document in docs:
    print(document)
