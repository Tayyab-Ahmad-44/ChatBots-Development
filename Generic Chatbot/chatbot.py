import fitz
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
import chainlit as cl 
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()


groq_api_key = os.environ["GROQ_API_KEY"]


llm_groq = ChatGroq(
    groq_api_key = groq_api_key, 
    model = "llama3-70b-8192",
    temperature = 0.6,
)



@cl.on_chat_start
async def on_chat_start():
    files = None # Initialize NULL to store uploaded files
    
    # Wait for user to upload files
    while files is None:
        files = await cl.AskFileMessage(
            content = "Please upload one or more pdf files to begin!",
            accept = ["application/pdf"],
            max_size_mb = 800, # Optionally limit the size of pdf
            max_files = 10,
            timeout = 180, # Set a timeout for user response
        ).send()
        
    # Process each input file
    texts = []
    metadatas = []

    for file in files:
        pdf = fitz.open(file.path)

        pdf_text = ""
        # Iterate through pages and extract text
        for page in pdf:
        # Extract text with optional sorting for reading order
            extracted_page_text = page.get_text(sort=True)
            pdf_text += extracted_page_text + "\n"  # Add newline for separation

        # Split the text into chunks.
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
        file_texts = text_splitter.split_text(pdf_text)
        texts.extend(file_texts)

        # Correcting dictionary creation:
        file_metadatas = [{"source": f"{i}-{file.name}"} for i in range(len(file_texts))]
        
        print("Metadata being passed:", file_metadatas)

        metadatas.extend(file_metadatas)
        
    # Create a Chroma Vector Store
    print("HI")
    embeddings = OllamaEmbeddings(model = "nomic-embed-text")
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadatas = metadatas
    )
    print("BY")

    # Initialize message history for conversation
    message_history = ChatMessageHistory()

    # Memory for conversational context
    memory = ConversationBufferMemory(
        memory_key = "chat_history",
        output_key = "answer",
        chat_memory = message_history,
        return_messages = True,
    )

    # Create a chain that uses Chroma Vector Store
    chain = ConversationalRetrievalChain.from_llm(
        llm = llm_groq,
        chain_type = "stuff",
        retriever = docsearch.as_retriever(),
        memory = memory,
        return_source_documents = True,
    )

    # Sending an image with number of files
    elements = [
        cl.Image(name = "image", display = "inline", path = "hot.jpg")
    ]

    # Inform the user that processsing has been ended. You can chat now.
    msg = cl.Message(content = f"Processed {len(files)} files done. You can now ask questions!", elements = elements)
    await msg.send()

    # Store the chain in user session
    cl.user_session.set("chain", chain)



@cl.on_message
async def main(message: cl.Message):
    # Retrieve the chain from user session
    chain = cl.user_session.get("chain")

    # Call back happens asynchronously/parallel
    cb = cl.AsyncLangchainCallbackHandler()

    # Call the chain with user's message content
    res = await chain.ainvoke(message.content, callbacks = [cb])
    answer = res["answer"]
    source_documents = res["source_documents"]

    text_elements = [] # Initialize list to store text elements

    # Process source_documents if available
    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"

            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content = source_doc.page_content, name = source_name)                
            )

        source_names = [text_el.name for text_el in text_elements]

        # Add source references to the answer
        if source_names:
            answer += f"\nSources: {', ' .join(source_names)}"
        else:
            answer += "No sources found!"

    # Return Results
    await cl.Message(content = answer, elements = text_elements).send()