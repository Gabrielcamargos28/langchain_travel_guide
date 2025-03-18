from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langchain.globals import set_debug
from langchain.output_parsers import DatetimeOutputParser 



load_dotenv()
set_debug(True)


llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    api_key = os.getenv("OPENAI_API_KEY")
)
parseador = DatetimeOutputParser()

modelo_data = """
    Responda a pergunta:
    {pergunta}
    {formatacao_de_saida}
"""

prompt = PromptTemplate(
    template=modelo_data,
    input_variables=["pergunta"],
    partial_variables={"formatacao_de_saida": parseador.get_format_instructions()}
)

PromptTemplate(
    input_variables=['pergunta'], 
    partial_variables={'formato_saida': "Write a datetime string that matches the following pattern: '%Y-%m-%dT%H:%M:%S.%fZ'.\n\nExamples: 0668-08-09T12:56:32.732651Z, 1213-06-23T21:01:36.868629Z, 0713-07-06T18:19:02.257488Z\n\nReturn ONLY this string, no other words!"}, template='Answer the users question:\n\n{question}\n\n{format_instructions}')

chain = prompt | llm | parseador

resposta = chain.invoke({"pergunta": "Quando a bitcoin foi fundada?"})

print(resposta)


