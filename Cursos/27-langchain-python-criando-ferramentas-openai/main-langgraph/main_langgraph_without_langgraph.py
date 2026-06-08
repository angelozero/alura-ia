from factory_service import get_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from typing import TypedDict, Literal


def main():
    
    class Rota(TypedDict):
        destino: Literal["praia", "trilha"]
    
    llm = get_chat_model(model_name="gpt-4o-mini")

    prompt_praias = ChatPromptTemplate.from_messages(
        [
            ("system", "Apresente-se como Sr Praia. Você é um especialista em praias do Brasil."),
            ("human", "{input_user_message}"),
        ]
    )
    
    prompt_trilhas = ChatPromptTemplate.from_messages(
        [
            ("system", "Apresente-se como Sr Trilha. Você é um especialista em trilhas do Brasil."),
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
    
    def response_to_rota(question) -> Rota:
        rota = chain_roteador.invoke({"input_user_message": question})
        print(f"\nRota escolhida: {rota['destino']}")
        if rota['destino'] == "praia":
            return chain_praias.invoke({"input_user_message": question})
        else:
            return chain_trilhas.invoke({"input_user_message": question})
        
    print(response_to_rota("Quais são os 3 melhores lugares do Brasil para turismo?"))


if __name__ == "__main__":
    main()



