from factory_service import get_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.tools import BaseTool
from estudante import Estudante
from factory_service import get_chat_model


class DadosEstudante(BaseTool):
    name: str = "DadosEstudante"
    description: str = """Ferramenta de extração de informação"""

    def _run(self, input: str) -> str:
        llm = get_chat_model()

        parser = JsonOutputParser(pydantic_object=Estudante)

        template = PromptTemplate(
            template="""
            Você deve análisar a {input} e extrair o nome de usuario informado.
            Formato de saída:
            {formato_saida}
            """,
            input_variables=["input"],
            partial_variables={"formato_saida": parser.get_format_instructions()},
        )

        cadeia = template | llm | parser

        return cadeia.invoke({"input": input})


print(DadosEstudante().run("Quais os dados do Angelo?"))
