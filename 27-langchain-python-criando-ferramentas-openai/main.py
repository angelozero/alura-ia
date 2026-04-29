from factory_service import get_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from country import Country
from city import CityList


def main():
    llm = get_chat_model()

    json_output_parser_country = JsonOutputParser(pydantic_object=Country)
    json_output_parser_city = JsonOutputParser(pydantic_object=CityList)

    prompt_country_json = PromptTemplate(
        template="What is the capital of {country} and what are its main cities?\n{output_format}",
        input_variables=["country"],
        partial_variables={
            "output_format": json_output_parser_country.get_format_instructions()
        },
    )
    
    prompt_city_json = PromptTemplate(
        template="What are the nearest cities to this capital {capital}?\n{output_format}",
        partial_variables={
            "output_format": json_output_parser_city.get_format_instructions()
        },
    )

    chain_1 = prompt_country_json | llm | json_output_parser_country

    chain_2 = prompt_city_json | llm | JsonOutputParser(pydantic_object=CityList)

    chain_final = (
        chain_1 
        | RunnableParallel({
            "country": RunnablePassthrough(), 
            "cities": chain_2
        })
    )

    response = chain_final.invoke({"country": "France"})
    
    print("\n")
    print(response)
    print("\n")

if __name__ == "__main__":
    main()
