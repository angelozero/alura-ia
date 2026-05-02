from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import json
import pandas as pd

class Estudante(BaseModel):
    nome: str = Field("Nome do estudante")
    
class DadosEstudanteTool(BaseTool):
    name: str = "dado_estudante"
    description: str = """
        Ferramenta de extração de informação
        Esta ferramenta deve receber apenas o nome do estudante como entrada
    """

    def _run(self, input: str) -> str:
        estudante = input
        estudante = estudante.lower()
        dados = busca_dados_de_estudante(estudante)
        return json.dumps(dados)

def busca_dados_de_estudante(estudante):
    dados = pd.read_csv("28-langchain-desenvolva-agentes-inteligencia-artificial/data/estudantes.csv")
    dados_com_esse_estudante = dados[dados["USUARIO"] == estudante]
    if dados_com_esse_estudante.empty:
        return {}
    return dados_com_esse_estudante.iloc[:1].to_dict()