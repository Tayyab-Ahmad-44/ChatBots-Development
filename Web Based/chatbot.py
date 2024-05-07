import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.document_loaders import FireCrawlLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.docstore.document import Document
# from firecrawl import FireCrawlLoader


# firecrawl_api_key = os.environ["FIRECRAWL_API_KEY"]
firecrawl_api_key = "fc-fa417913d3054e63b2dd318d100c8f63"


urls = [
    "https://en.wikipedia.org/wiki/Computer_programming",
    "https://en.wikipedia.org/wiki/List_of_programming_languages"
]

docs = [FireCrawlLoader(api_key = firecrawl_api_key, url = url, mode = "scrape").load() for url in urls]

# Split Documents

docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size = 250, chunk_overlap = 0
)

docs_splits = text_splitter.split_documents(docs_list)

print(len(docs_splits))