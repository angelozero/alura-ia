from pydantic import Field, BaseModel
from typing import List

class City(BaseModel):
    name: str = Field(..., description="The name of the city")
    
class CityList(BaseModel):
    cities: List[City] = Field(..., description="A list of nearby cities")