import os
import ast
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

SEPARATOR = "-" * 60


def execute(action):
    print(f"\n{SEPARATOR}")
    print("🧠 [langchain_service] execute() chamado")
    print(f"   📥 Argumento recebido: '{action}'")

    try:
        # Suporta tanto string simples ("macaco.png") quanto dict-string ('{"image_name": "macaco.png"}')
        try:
            parsed = ast.literal_eval(action)
            if isinstance(parsed, dict):
                image_name = parsed.get("image_name", "")
                print(f"   🗂️  Input interpretado como dict → image_name='{image_name}'")
            else:
                image_name = str(parsed)
                print(f"   🗂️  Input interpretado como literal → image_name='{image_name}'")
        except (ValueError, SyntaxError):
            image_name = action.strip()
            print(f"   🗂️  Input interpretado como string simples → image_name='{image_name}'")

        # =========================================================================
        # PASSO 1: Inicializando modelos LLM
        # =========================================================================
        print(f"\n{SEPARATOR}")
        print("⚙️  [PASSO 1] Inicializando modelos LLM...")
        llm_gemini_2_5_flash = get_chat_model()
        print(f"   ✅ Modelo 1 criado: {type(llm_gemini_2_5_flash).__name__} (gemini-2.0-flash) → usado na 1ª cadeia")
        llm_gemini_3_1_pro = get_chat_model(model_name="gemini-3.1-pro")
        print(f"   ✅ Modelo 2 criado: {type(llm_gemini_3_1_pro).__name__} (gemini-3.1-pro) → usado na 2ª cadeia")

        # =========================================================================
        # PASSO 2: Carregando imagem
        # =========================================================================
        print(f"\n{SEPARATOR}")
        print("🖼️  [PASSO 2] Carregando e codificando imagem em base64...")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(base_dir, "data", image_name)
        print(f"   📁 Caminho da imagem: {image_path}")
        imagem = encode_image(image_path)
        print(f"   ✅ Imagem codificada com sucesso ({len(imagem)} caracteres base64)")

        # =========================================================================
        # PASSO 3: Construindo a 1ª cadeia (descrição da imagem)
        # =========================================================================
        print(f"\n{SEPARATOR}")
        print("🔗 [PASSO 3] Construindo a 1ª cadeia: ChatPromptTemplate → LLM (flash) → StrOutputParser")
        print("   📋 Prompt: SystemMessage + HumanMessage com imagem em base64")
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
        first_chain = template | llm_gemini_2_5_flash | StrOutputParser()
        print("   ✅ 1ª cadeia montada: template | llm_gemini_2_5_flash | StrOutputParser()")

        # =========================================================================
        # PASSO 4: Construindo a 2ª cadeia (resumo JSON)
        # =========================================================================
        print(f"\n{SEPARATOR}")
        print("🔗 [PASSO 4] Construindo a 2ª cadeia: PromptTemplate → LLM (pro) → JsonOutputParser")
        json_parser = JsonOutputParser(pydantic_object=ImageInfo)
        print(f"   📐 Schema de saída: {ImageInfo.__name__} (Pydantic)")

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
                "output_format": {
                    "output_format": json_parser.get_format_instructions()
                },
            },
        )
        second_chain = template_response | llm_gemini_3_1_pro | json_parser
        print("   ✅ 2ª cadeia montada: template_response | llm_gemini_3_1_pro | JsonOutputParser()")

        # =========================================================================
        # PASSO 5: Combinando em cadeia final
        # =========================================================================
        print(f"\n{SEPARATOR}")
        print("🔗 [PASSO 5] Combinando cadeias em cadeia final:")
        print('   final_chain = {"chain_response_data": first_chain} | second_chain')
        final_chain = {"chain_response_data": first_chain} | second_chain
        print("   ✅ Cadeia final montada")

        # =========================================================================
        # PASSO 6: Invocando a cadeia final
        # =========================================================================
        print(f"\n{SEPARATOR}")
        print("🚀 [PASSO 6] Invocando cadeia final...")
        print("   ⏳ Chamando 1ª cadeia → LLM gemini-2.0-flash (descrição da imagem)...")
        handler = StdOutCallbackHandler()
        response = final_chain.invoke(
            {"imagem_informada": imagem}, config={"callbacks": [handler]}
        )

        # =========================================================================
        # PASSO 7: Resultado final
        # =========================================================================
        print(f"\n{SEPARATOR}")
        print("✅ [PASSO 7] Resposta final da cadeia:")
        print(f"   {response}")
        print(SEPARATOR)

    except Exception as e:
        print(f"\n❌ [langchain_service] Erro durante execução: {e}")
        raise

    return response
