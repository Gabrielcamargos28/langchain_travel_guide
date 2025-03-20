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
from langchain.document_loaders import PyPDFLoader, TextLoader

from dotenv import load_dotenv

load_dotenv()
set_debug(True)

pasta = "pdfs/"

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    api_key=os.getenv("OPENAI_API_KEY"))


arquivos_pdfs = [os.path.join(pasta, f) for f in os.listdir(pasta) if f.endswith(".pdf")]

carregador = TextLoader("GTB_standard_Nov23.txt", encoding="utf-8") 

quebrador = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

#documentos = carregador.load()

documentos = []
arquivo_txt = os.path.join(pasta, "GTB_standard_Nov23.txt")

for arquivo in arquivos_pdfs:
    carregador = PyPDFLoader(arquivo)
    documentos.extend(carregador.load())

if os.path.exists(arquivo_txt):
    carregador_txt = TextLoader(arquivo_txt, encoding="utf-8")
    documentos.extend(carregador_txt.load())

textos = quebrador.split_documents(documentos)


print(textos)

embeddings = OpenAIEmbeddings()

db = FAISS.from_documents(textos, embeddings)

qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())

pergunta = "Como devo proceder caso tenha um item comprado roubado"
resultado = qa_chain.invoke({"query": pergunta})
print(resultado)