from factory_service import get_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
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
    print("----")

    # ==== JSON output parser ====
    json_output_parser = JsonOutputParser(pydantic_object=Country)
    
    prompt_model_json = PromptTemplate(
        template="What is the capital of {country}?\n{output_format}",
        input_variables=["country"],
        partial_variables={
            "output_format": json_output_parser.get_format_instructions()
        },
    )

    chain_json = prompt_model_json | llm | json_output_parser

    response_json = chain_json.invoke({"country": "France"})
    print(response_json)


if __name__ == "__main__":
    main()
