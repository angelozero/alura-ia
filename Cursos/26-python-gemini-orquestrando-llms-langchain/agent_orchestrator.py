from factory import get_chat_model
from langchain.agents import create_agent
from image_analyzing_tool import ImageAnalyzingTool

SEPARATOR = "-" * 60

class AgentOrchestrator:
    def __init__(self):
        print(f"\n{SEPARATOR}")
        print("🔧 [AgentOrchestrator] Inicializando orquestrador...")

        print("🛠️  [AgentOrchestrator] Criando ferramenta: ImageAnalyzingTool")
        image_analyzing_tool = ImageAnalyzingTool()
        print(f"   ✅ Ferramenta criada: name='{image_analyzing_tool.name}'")
        print(f"   📄 Descrição: {image_analyzing_tool.description.strip()[:80]}...")

        print("🤖 [AgentOrchestrator] Inicializando modelo LLM: gemini-3.1-pro")
        self.llm = get_chat_model(model_name="gemini-3.1-pro")
        print(f"   ✅ Modelo criado: {type(self.llm).__name__}")

        self.tools = [image_analyzing_tool]
        print(f"🧰 [AgentOrchestrator] Ferramentas registradas: {[t.name for t in self.tools]}")

        print("🕸️  [AgentOrchestrator] Criando agente com create_agent (LangGraph CompiledStateGraph)...")
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=(
                "Você é um assistente especializado em análise de imagens. "
                "Use as ferramentas disponíveis para responder às solicitações do usuário."
            ),
        )
        print(f"   ✅ Agente criado: {type(self.agent).__name__}")
        print(SEPARATOR)