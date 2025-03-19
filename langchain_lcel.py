from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain_core.pydantic_v1 import Field, BaseModel
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

class Destino(BaseModel): 
    cidade = Field("cidade a visitar")
    motivo = Field("motivo pelo qual é interessante visitar")


llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    api_key = os.getenv("OPENAI_API_KEY")
)

parseador = JsonOutputParser(pydantic_object=Destino)

modelo_cidade = PromptTemplate(
    template="""
        Sugira uma cidadea dado meu interesse por {interesse}.
        {formatacao_de_saida}
        """,
        input_variables=["interesse"],
        partial_variables={"formatacao_de_saida": parseador.get_format_instructions()}
) 
modelo_resturantes = ChatPromptTemplate.from_template(
    "Sugira restaurantes populares entre locais em {cidade}."
)
modelo_cultural = ChatPromptTemplate.from_template(
    "Sugira atividades e locais culturais em {cidade}."
)
parte1 = modelo_cidade | llm | parseador    

resultado = parte1.invoke("praias") 

print(resultado) 