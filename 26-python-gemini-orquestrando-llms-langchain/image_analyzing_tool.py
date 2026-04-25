from langchain.tools import BaseTool
from langchain_service import execute

SEPARATOR = "-" * 60

class ImageAnalyzingTool(BaseTool):
    name: str = "ImageAnalyzingTool"
    description: str = """
        Utilize esta ferramenta sempre que for solicitado que você faça uma análise de uma imagem.
        
        # Entrada:
        - 'action' (str) : Nome da imagem a ser analisada, incluindo a extensão do arquivo (exemplo: 'foto.png', 'foto.jpg').
    """

    # True: A resposta da ferramenta deve ser a resposta final para a consulta, ou seja, não pode ser utilizada em conjunto com outras ferramentas.
    # False: A resposta da ferramenta pode ser utilizada em conjunto com outras ferramentas.
    return_direct: bool = False

    def _run(self, action: str) -> str:
        print(f"\n{SEPARATOR}")
        print(f"🔍 [ImageAnalyzingTool] Ferramenta acionada pelo agente")
        print(f"   📂 Imagem recebida: '{action}'")
        print(f"   ➡️  Delegando para langchain_service.execute()")
        print(SEPARATOR)
        result = execute(action)
        print(f"\n{SEPARATOR}")
        print(f"✅ [ImageAnalyzingTool] Execução concluída, retornando resultado ao agente")
        print(SEPARATOR)
        return result
