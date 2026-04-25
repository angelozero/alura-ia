from factory import get_chat_model
from my_helper import encode_image
from langchain_core.messages import SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.callbacks import StdOutCallbackHandler
from image_info import ImageInfo


def run_hello_world():
    print("--- Iniciando comunicação com a IA ---")

    try:

        # 1. Definindo modelos para cada etapa do processo
        llm_gemini_2_5_flash = get_chat_model()
        llm_gemini_3_1_pro = get_chat_model(model_name="gemini-3.1-pro")

        # 2. Caminho para a imagem de teste
        imagem = encode_image(
            "/Users/angelo/Projects/AI/alura-ia/26-python-gemini-orquestrando-llms-langchain/data/macaco.png"
        )

        # =========================================================================
        # ================== Criando primeira cadeia ==============================
        # =========================================================================

        # 3. Criando o prompt usando o ChatPromptTemplate para a primeira cadeia de processamento
        template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="""
                            Você é um assistente de descrição de imagens. Responda de forma simples e direta.

                            # Formato de Saída Esperado:
                            Descrição: 'Coloque aqui a descrição da imagem'
                            Rótulo: 'Coloque aqui uma lista com três termos chaves separados por virgula'
                        """
                ),
                HumanMessagePromptTemplate.from_template(
                    [
                        {
                            "type": "text",
                            "text": "Descreva a imagem sendo o mais simples possível.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,{imagem_informada}"
                            },
                        },
                    ]
                ),
            ]
        )

        # 4. Criando a primeira cadeia de processamento (first_chain) para obter a descrição e os rótulos da imagem
        first_chain = template | llm_gemini_2_5_flash | StrOutputParser()

        # =========================================================================
        # ================== Criando segunda cadeia ===============================
        # =========================================================================

        # 5. Criando o template para a segunda cadeia de processamento (second_chain) com json que irá gerar um resumo baseado na resposta da primeira cadeia
        json_parser = JsonOutputParser(
            pydantic_object=ImageInfo
        )
          
        template_response = PromptTemplate(
            template="""
            Gere um resumo, utilizando uma linguagem clara e objetiva, priorizando registros para consultas posteriores.
            
            # Resultado da imagem
            {chain_response_data}
            
            # Formato de saída esperada:
            {output_format}
            """,
            input_variables=["chain_response_data"],
            partial_variables={
                "output_format": {"output_format": json_parser.get_format_instructions()},
            },
        )

      

        # 6. Criando a segunda cadeia de processamento (second_chain) que irá gerar um resumo baseado na resposta da primeira cadeia
        second_chain = template_response | llm_gemini_3_1_pro | json_parser

        # =========================================================================
        # ================== Criando cadeia final =================================
        # =========================================================================

        # 7. Combinando as duas cadeias de processamento em uma cadeia final (final_chain) que irá primeiro obter a descrição e os rótulos da imagem e depois gerar um resumo baseado nessa
        final_chain = {"chain_response_data": first_chain} | second_chain

        # 8. Chama a IA usando a cadeia final (final_chain) que irá processar a imagem e gerar o resumo
        handler = StdOutCallbackHandler()
        response = final_chain.invoke(
            {"imagem_informada": imagem}, config={"callbacks": [handler]}
        )
        

        # 9. Exibindo o resultado
        print("--- Resposta da IA ---")
        print(response)
        print("-----------------------")

    except Exception as e:
        print(f"Erro ao conectar com o LiteLLM: {e}")


if __name__ == "__main__":
    run_hello_world()
