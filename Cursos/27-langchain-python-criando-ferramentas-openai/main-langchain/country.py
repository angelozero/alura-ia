from pydantic import Field, BaseModel

class Country(BaseModel):
    name: str = Field(..., description="The name of the country")
    capital: str = Field(..., description="The capital city of the country")