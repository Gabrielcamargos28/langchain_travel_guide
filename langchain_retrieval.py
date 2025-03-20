from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.chains import SimpleSequentialChain
from langchain.chains import LLMChain
from langchain.chains import ConversationChain
from langchain.globals import set_debug
from langchain.memory import ConversationBufferMemory
import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA

from dotenv import load_dotenv

load_dotenv()
set_debug(True)

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    api_key=os.getenv("OPENAI_API_KEY"))

carregador = TextLoader("GTB_standard_Nov23.txt", encoding="utf-8") 

quebrador = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

documentos = carregador.load()
textos = quebrador.split_documents(documentos)

print(textos)

embeddings = OpenAIEmbeddings()

db = FAISS.from_documents(textos, embeddings)

qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())

pergunta = "Como devo proceder caso tenha um item comprado roubado"
resultado = qa_chain.invoke({"query": pergunta})
print(resultado)