from pydantic import BaseModel


class ModelCreate(BaseModel):
    name: str
    uri: str


class ModelRead(BaseModel):
    id: str
    name: str
    uri: str

    class Config:
        from_attributes = True
