from langchain_classic.agents import (
    AgentExecutor,
    create_openai_functions_agent,
    create_react_agent
)
from langchain_classic import hub
from factory_service import get_chat_model
from dado_estudante_tool import DadosEstudanteTool
from perfil_academico_tool import PerfilAcademicoTool

def execute_agent(query: str):
    llm = get_chat_model()
    tools = [DadosEstudanteTool(), PerfilAcademicoTool()]
    
    # openai functions
    # prompt = hub.pull("hwchase17/openai-functions-agent")
    # agente = create_openai_tools_agent(llm, self.tools,

    # react
    prompt = hub.pull("hwchase17/react")
    agente = create_react_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(agent=agente, tools=tools, verbose=True)
    
    return agent_executor.invoke({"input": query})