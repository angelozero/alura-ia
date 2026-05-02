from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from factory_service import get_chat_model
from typing import List
import json
import pandas as pd


class Nota(BaseModel):
    nome: str = Field("Nome da area de conhecimento")
    nota: float = Field("Nota na area de conhecimento")


class PerfilAcademicoEstudante(BaseModel):
    nome: str = Field("Nome do estudante")
    ano_de_conclusao: int = Field("Ano de conclusão")
    notas: List[Nota] = Field("Lista de notas das disciplinas")
    resumo: str = Field(
        "Resumo das principais caracteristicas deste estudante de forma a torna-lo unico e um forte potencial para a faculdade"
    )


class PerfilAcademicoTool(BaseTool):
    name: str = "perfil_academico"
    description: str = (
        """
        Ferramenta de criação de perfil acadêmico de um estudante. Esta ferramenta requer como entrada todos os dados do estudante
        Eu sou incapaz de buscar os dados do estudante.
        Antes de utilizar minha funcionalidade você tem como regra principal buscar os perfis dos estudantes e após isso me utilizar
        """
    )

    def _run(self, input: str) -> str:
        parser = JsonOutputParser(pydantic_object=PerfilAcademicoEstudante)
        template = PromptTemplate(
            template=""" 
                                  - Formate o estudante para o seu perfil acadêmico
                                  - Com os dados identifique as opções de universidades e cursos compatíveis com o interesse do aluno
                                  - Destaque o perfil do aluno dando enfase principalmente naquilo que faz sentido para as instituições de interesse do aluno
                                  
                                  Persona: Você é uma consulta de carreira e precisa indicar com detalhes, riqueza, mas direta ao ponto para o estudante as faculdades e as opções
                                  Informações atuais:
                                  
                                  {dados_do_estudante}
                                  {formato_de_saida}
                                  """,
            input_variables=["dados_do_estudante"],
            partial_variables={"formato_de_saida": parser.get_format_instructions()},
        )
        llm = get_chat_model()
        cadeia = template | llm | parser
        return cadeia.invoke({"dados_do_estudante": input})
