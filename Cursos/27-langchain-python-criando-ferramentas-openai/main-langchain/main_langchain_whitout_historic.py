from factory_service import get_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory


def main():
    llm = get_chat_model()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant in country information."),
            ("placeholder", "{query_history}"),
            ("human", "{input_user_message}"),
        ]
    )

    question_list = [
        "What is the most famous city of Brazil?",
        "And how many people live there?",
        "And what is the most famous monument in the city?",
    ]

    chain = prompt | llm | StrOutputParser()

    memory = {}
    session_history = "langchain_id"

    def session_history_func():
        if session_history not in memory:
            memory[session_history] = InMemoryChatMessageHistory()
        return memory[session_history]

    chain_with_history = RunnableWithMessageHistory(
        runnable=chain,
        get_session_history=session_history_func,
        input_messages_key="input_user_message",
        history_messages_key="query_history",
    )

    for q in question_list:
        response = chain_with_history.invoke(
            {"input_user_message": q}, config={"configurable": {"session_id": session_history}}
        )
        print(f"\n{response}\n")


if __name__ == "__main__":
    main()
