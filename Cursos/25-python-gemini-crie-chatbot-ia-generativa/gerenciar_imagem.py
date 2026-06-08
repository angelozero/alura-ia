import base64


def gerar_imagem_base64(caminho_imagem: str) -> str:
    """
    Reads an image file and returns its base64-encoded string,
    suitable for use in LangChain multimodal messages (Ollama vision models).
    """
    with open(caminho_imagem, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_media_type(caminho_imagem: str) -> str:
    """
    Infers the MIME type from the file extension.
    Supports jpeg, png, gif and webp.
    """
    ext = caminho_imagem.rsplit(".", 1)[-1].lower()
    media_types = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp",
    }
    return media_types.get(ext, "image/jpeg")
