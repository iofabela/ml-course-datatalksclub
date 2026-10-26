import pickle
from pydantic import BaseModel

from model import normalize_record


class Lead(BaseModel):
    lead_source: str | None = None
    industry: str | None = None
    employment_status: str | None = None
    location: str | None = None
    number_of_courses_viewed: int | None = None
    annual_income: float | None = None
    interaction_count: int | None = None
    lead_score: float | None = None


CLIENT = {
  "lead_source": "organic_search",
  "industry": "technology",
  "employment_status": "employed",
  "location": "europe",
  "number_of_courses_viewed": 4,
  "annual_income": 80304.0,
  "interaction_count": 7,
  "lead_score": 0.74
}

file = 'pipeline.bin'

with open(file, "rb") as model_file:
    pipeline = pickle.load(model_file)

lead = Lead(**CLIENT)
probability = float(pipeline.predict_proba(lead.model_dump())[0, 1]) 
## note : is prefer decompose the values for avoid 'NA' or None values from requests [lead->normalize_record(…)]
# * `normalize_record()` fills in missing values using the same rules as training.
print(round(probability, 3))
