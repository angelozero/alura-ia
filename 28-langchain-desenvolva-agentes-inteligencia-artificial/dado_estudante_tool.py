from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from factory_service import get_chat_model
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
        parser = JsonOutputParser(pydantic_object=Estudante)
        template = PromptTemplate(
            template=""" 
                                  - Você deve análisar a informação recebida e devolver apenas o nome do estudante
                                  
                                  Entrada:
                                  -------------------
                                  {nome_do_estudante}
                                  -------------------
                                  
                                  Formato de saída:
                                  -------------------
                                  {formato_de_saida}
                                  -------------------
                                  """,
            input_variables=["nome_do_estudante"],
            partial_variables={"formato_de_saida": parser.get_format_instructions()},
            return_direct=False
        )
        
        llm = get_chat_model()
        
        cadeia = template | llm | parser

        estudante = cadeia.invoke({"nome_do_estudante": input})

        estudante = estudante['nome'].lower()
        dados = busca_dados_de_estudante(estudante)
        
        return json.dumps(dados)


def busca_dados_de_estudante(estudante):
    dados = pd.read_csv(
        "28-langchain-desenvolva-agentes-inteligencia-artificial/data/estudantes.csv"
    )
    dados_com_esse_estudante = dados[dados["USUARIO"] == estudante]
    if dados_com_esse_estudante.empty:
        return {}
    return dados_com_esse_estudante.iloc[:1].to_dict()
