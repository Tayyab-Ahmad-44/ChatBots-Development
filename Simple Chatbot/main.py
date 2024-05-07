import argparse
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
import os

PROMPT_TEMPLATE = """
You are an Ai-powered chatbot designed to provide information based on the following context:

----------


{context}


----------

Answer the question based on the above context: {question}
"""

# groq_api_key = os.environ["GROQ_API_KEY"]


def main():
    #Create CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    
    #Prepare the DB.
    embeddings = OllamaEmbeddings(model = "nomic-embed-text")
    db = FAISS.load_local("faiss-index", embeddings, allow_dangerous_deserialization=True)
    
    #Search the DB.
    embedding_vector = embeddings.embed_query(query_text)
    docs = db.similarity_search_by_vector(embedding_vector)  
    print(docs)
    
    # Format and print docs
    context_text = "\n\n---\n\n".join([doc.page_content for doc in docs])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context = context_text, question = query_text)
    print(prompt)
    
    
    # Passing Context and Question to LLM
    model = ChatGroq(temperature = 0.2, model = "llama3-70b-8192", groq_api_key=groq_api_key)
    response = model.invoke(prompt)    
    print(response.content)  
    
    
if __name__ == "__main__":
    main()