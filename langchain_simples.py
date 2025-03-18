from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate

load_dotenv()

numero_de_dias = 7
numero_de_criancas = 2
atividade = "praia"

# código omitido

modelo_do_prompt = PromptTemplate.from_template(
    "Crie um roteiro de viagem de {numero_de_dias} dias, para uma família com {numero_de_criancas} crianças, que gostam de {atividade}."
)

prompt = f"Crie um roteiro de viagem de {numero_de_dias} dias, para uma família com {numero_de_criancas} crianças, que gostam de {atividade}."

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    api_key=os.getenv("OPENAI_API_KEY")
)

resposta = llm.invoke(modelo_do_prompt.format(numero_de_dias=numero_de_dias, numero_de_criancas=numero_de_criancas, atividade=atividade))
print(resposta)