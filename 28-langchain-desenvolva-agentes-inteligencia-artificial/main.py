from langchain_classic.agents import (
    AgentExecutor,
    create_openai_functions_agent,
)
from langchain_classic import hub

from factory_service import get_chat_model

from dado_estudante_tool import DadosEstudanteTool


def main():
    dados_estudante_tool = DadosEstudanteTool()
    
    tools = [dados_estudante_tool]
    
    llm = get_chat_model()
    
    prompt = hub.pull("hwchase17/openai-functions-agent")
    
    agente = create_openai_functions_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(agent=agente, tools=tools, verbose=True)
    
    agent_executor.invoke({"input": "Quem é o Angelo?"})

if __name__ == "__main__":
    main()
