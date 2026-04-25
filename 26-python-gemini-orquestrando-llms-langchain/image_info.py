from pydantic import BaseModel, Field


class ImageInfo(BaseModel):
    title: str = Field(..., description="Um titulo que resuma visualmente as características da imagem utilizando apenas duas palavras")
    description: str = Field(..., description="Descrição da imagem")
    labels: list[str] = Field(..., description="Lista de rótulos associados à imagem")
