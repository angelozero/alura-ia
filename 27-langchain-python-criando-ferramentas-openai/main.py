from factory_service import get_chat_model
from langchain_core.prompts import PromptTemplate

def main():
    
    
    prompt_model = PromptTemplate(
        template="What is the capital of {country}?",
        input_variables=["country"]
    )
    
    
    llm = get_chat_model()
    response = llm.invoke(prompt_model.format(country="France"))
    print(response.content)


if __name__ == "__main__":
    main()
