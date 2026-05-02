from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from factory_service import get_chat_model
from typing import List
import json
import pandas as pd


class Universidade(BaseModel):
    nome: str = Field("Nome do universidade")

class DadosUniversidadeTool(BaseTool):
    name: str = "dado_universidade"
    description: str = """
        Ferramenta de extração de informação
        Esta ferramenta deve receber apenas o nome da universidade como entrada
    """

    def _run(self, input: str) -> str:
        parser = JsonOutputParser(pydantic_object=Universidade)
        template = PromptTemplate(
            template=""" 
                                  - Você deve análisar a informação recebida e devolver apenas o nome da universidade
                                  
                                  Entrada:
                                  -------------------
                                  {nome_universidade}
                                  -------------------
                                  
                                  Formato de saída:
                                  -------------------
                                  {formato_de_saida}
                                  -------------------
                                  """,
            input_variables=["nome_universidade"],
            partial_variables={"formato_de_saida": parser.get_format_instructions()},
            return_direct=False,
        )

        llm = get_chat_model()

        cadeia = template | llm | parser

        universidade = cadeia.invoke({"nome_universidade": input})

        universidade = universidade["nome"]
        dados = busca_dados_universidade(universidade)

        return json.dumps(dados)
    
class DadosTodasUniversidadeTool(BaseTool):
    name: str = "dados_todas_universidade"
    description: str = """
        Ferramenta de extração de informação de todas as universidades
    """

    def _run(self, input: str) -> str:
        dados = busca_dados_todas_universidades()
        return dados


def busca_dados_universidade(universidade):
    dados = pd.read_csv(
        "28-langchain-desenvolva-agentes-inteligencia-artificial/data/universidades.csv"
    )
    dados_universidade = dados[dados["NOME_FACULDADE"] == universidade]
    if dados_universidade.empty:
        return {}
    return dados_universidade.iloc[:1].to_dict()

def busca_dados_todas_universidades():
    dados = pd.read_csv(
        "28-langchain-desenvolva-agentes-inteligencia-artificial/data/universidades.csv"
    )
    return dados.to_dict()
