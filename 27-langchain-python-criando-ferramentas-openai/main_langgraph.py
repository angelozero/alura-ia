from factory_service import get_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
import asyncio




class Rota(TypedDict):
    destino: Literal["praia", "trilha"]

class Estado(TypedDict):
    question: str
    destino: Rota
    resposta: str

llm = get_chat_model(model_name="gpt-4o-mini")

prompt_praias = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Apresente-se como Sr Praia. Você é um especialista em praias do Brasil.",
        ),
        ("human", "{input_user_message}"),
    ]
)

prompt_trilhas = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Apresente-se como Sr Trilha. Você é um especialista em trilhas do Brasil.",
        ),
        ("human", "{input_user_message}"),
    ]
)

prompt_roteador = ChatPromptTemplate.from_messages(
    [
        ("system", "Responda apenas e única exlusivamente com 'praia' ou 'trilha'"),
        ("human", "{input_user_message}"),
    ]
)

chain_praias = prompt_praias | llm | StrOutputParser()
chain_trilhas = prompt_trilhas | llm | StrOutputParser()
chain_roteador = prompt_roteador | llm.with_structured_output(Rota)

async def no_roteador(estado: Estado, config=RunnableConfig):
    return {
        "destino": await chain_roteador.ainvoke(
            {"input_user_message": estado["question"]}, config=config
        )
    }

async def no_praias(estado: Estado, config=RunnableConfig):
    return {
        "resposta": await chain_praias.ainvoke(
            {"input_user_message": estado["question"]}, config=config
        )
    }

async def no_trilhas(estado: Estado, config=RunnableConfig):
    return {
        "resposta": await chain_trilhas.ainvoke(
            {"input_user_message": estado["question"]}, config=config
        )
    }

def escolher_rota(estado: Estado):
    print(f"\nRota escolhida: {estado['destino']['destino']}")
    if estado["destino"]["destino"] == "praia":
        return "praias"
    else:
        return "trilhas"

graph = StateGraph(Estado)
graph.add_node("roteador", no_roteador)
graph.add_node("praias", no_praias)
graph.add_node("trilhas", no_trilhas)

graph.add_edge(START, "roteador")
graph.add_conditional_edges("roteador", escolher_rota)
graph.add_edge("praias", END)
graph.add_edge("trilhas", END)

app = graph.compile()

async def main():
    question = {"question": "Quais são os 3 melhores lugares do Brasil para ver mais paisagem verde?"}
    
    try:
        resultado = await app.ainvoke(question)
        print(f"\nResposta final: {resultado['resposta']}")
    
    except Exception as e:
        print(f"Erro na execução: {e}")

if __name__ == "__main__":
    asyncio.run(main())
