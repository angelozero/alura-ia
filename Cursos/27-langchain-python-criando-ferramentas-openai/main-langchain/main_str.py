from factory_service import get_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from country import Country


def main():
    llm = get_chat_model()

    # ==== String output parser ====
    output_parser = StrOutputParser()
    prompt_model_str = PromptTemplate(
        template="What is the capital of {country}?", input_variables=["country"]
    )

    chain = prompt_model_str | llm | output_parser

    response = chain.invoke({"country": "France"})
    print(response)


if __name__ == "__main__":
    main()
