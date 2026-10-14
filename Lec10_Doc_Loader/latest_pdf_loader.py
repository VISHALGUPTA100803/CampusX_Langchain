from pypdf import PdfReader
from langchain_core.documents import Document

file_path = "dl-curriculum.pdf"
reader = PdfReader(file_path)

docs = []

for page_number, page in enumerate(reader.pages):
    docs.append(
        Document(
            page_content=page.extract_text() or "",
            metadata={
                "source": file_path,
                "page": page_number,
                "total_pages": len(reader.pages)
            }
        )
    )

print(len(docs))

print(docs[0].page_content)
print(docs[0].metadata)

# You don’t need Path because PdfReader accepts a file path as a string:
# file_path = "dl-curriculum.pdf"
# reader = PdfReader(file_path)
# PdfReader opens the file and parses the PDF internally.
# In your text-file example, we used:
# Path(file_path).read_text(encoding="utf-8")
# because Path.read_text() did the reading.
# Here, PdfReader does the reading, so importing Path is optional. Both approaches use a path; only the first uses the Path class.