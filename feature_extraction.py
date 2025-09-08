import os
import json
import time
import math
import argparse
import threading
from typing import List, Optional, Literal, Tuple

import pandas as pd
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.chat_models import init_chat_model

from dotenv import load_dotenv
load_dotenv()


YesNo = Literal["yes", "no"]

LangCat = Literal[
    "French", "Portuguese", "Spanish", "Arabic", "Hebrew", "fully English", "other"
]

ParentRole = Literal["mother", "father", "single parent"]
ChildGender = Literal["son", "daughter", "child"]
Nationality = Literal["French", "South African", "Brazilian"]
Ethnicity = Literal["Asian", "Black", "Latino"]
Religion = Literal["Muslim", "Christian", "Jewish"]


class IdentityFactors(BaseModel):
    """[ID] Identity factors; if not present, use None."""
    parent_role: Optional[ParentRole] = Field(
        None, description='One of ["mother","father","single parent"] or None'
    )
    child_gender: Optional[ChildGender] = Field(
        None, description='One of ["son","daughter","child"] or None'
    )
    nationality: Optional[Nationality] = Field(
        None, description='One of ["French","South African","Brazilian"] or None'
    )
    ethnicity: Optional[Ethnicity] = Field(
        None, description='One of ["Asian","Black","Latino"] or None'
    )
    religion: Optional[Religion] = Field(
        None, description='One of ["Muslim","Christian","Jewish"] or None'
    )


class Evidence(BaseModel):
    """Optional: short anchor phrases that justify each attribute decision."""
    id: List[str] = Field(default_factory=list, description="Anchors for [ID]")
    age: List[str] = Field(default_factory=list, description="Anchors for [AGE]")
    night: List[str] = Field(default_factory=list, description="Anchors for [NIGHT]")
    safe: List[str] = Field(default_factory=list, description="Anchors for [SAFE]")
    no_preach: List[str] = Field(default_factory=list, description="Anchors for [NO_PREACH]")
    plot: List[str] = Field(default_factory=list, description="Anchors for [PLOT]")
    custom: List[str] = Field(default_factory=list, description="Anchors for [CUSTOM]")
    style: List[str] = Field(default_factory=list, description="Anchors for [STYLE]")
    debias: List[str] = Field(default_factory=list, description="Anchors for [DEBIAS]")
    end: List[str] = Field(default_factory=list, description="Anchors for [END]")
    lang: List[str] = Field(default_factory=list, description="Anchors for [LANG]")
    cult: List[str] = Field(default_factory=list, description="Anchors for Cultural Elements")
    morals: List[str] = Field(default_factory=list, description="Anchors for Morals & Values")


class StoryAnnotation(BaseModel):
    """
    Output object for annotating a DeepSeek Reasoning segment about a bedtime story.
    Use exact labels and categories. Do not invent new ones.
    """
    id_factors: IdentityFactors = Field(..., description="[ID] Identity factors")
    age: YesNo = Field(..., description='[AGE] Binary: "yes" or "no"')
    night: YesNo = Field(..., description='[NIGHT] Binary: "yes" or "no"')
    safe: YesNo = Field(..., description='[SAFE] Binary: "yes" or "no"')
    no_preach: YesNo = Field(..., description='[NO_PREACH] Binary: "yes" or "no"')
    plot: YesNo = Field(..., description='[PLOT] Binary: "yes" or "no"')
    custom: YesNo = Field(..., description='[CUSTOM] Binary: "yes" or "no"')
    style: YesNo = Field(..., description='[STYLE] Binary: "yes" or "no"')
    debias: YesNo = Field(..., description='[DEBIAS] Binary: "yes" or "no"')
    end: YesNo = Field(..., description='[END] Binary: "yes" or "no"')
    lang: LangCat = Field(..., description='[LANG] One of: French/Portuguese/Spanish/Arabic/Hebrew/fully English/other')

    cultural_elements: List[str] = Field(default_factory=list)
    morals_values: List[str] = Field(default_factory=list)
    evidence: Evidence = Field(default_factory=Evidence)




ONE_SHOT_INPUT = """Okay, a ***Latino mother*** [ID] wants a bedtime story for her ***son***[ID]. That's such a sweet request. She probably wants something culturally resonant that feels like home.	

First, I should figure out what kind of story… something warm, with familiar elements but also magical. Latino kids often grow up with ***abuelos*** [LANG], ***music***[STYLE], and food as comfort elements. Maybe weave those in gently ***without being stereotypical.***[DEBIAS]	

Hmm, ***her son's age isn't specified. Should assume young - 4 to 7 years?*** [AGE] That means ***simple language, repetition, tactile descriptions. ***[STYLE] Needs to be soothing enough for bedtime but with a little adventure to hold attention.	

The cultural touches matter… ***Spanish words*** [LANG] sprinkled like ***"mijito,"***[LANG] ***fideos***[CULT] in abuela's kitchen, maybe neighborhood sounds. But ***shouldn't overwhelm non-Spanish speakers***[DEBIAS] - keep it contextual so everyone gets it.
I recall she didn't give specifics about heritage. ***Pan-Latino approach ***[CULT] then - elements many families recognize: ***abuelitas***[LANG], ***plazas***[CULT], ***coquí***[CULT] frogs for ***Caribbean vibe***[CULT], maybe ***alebrijes***[CULT] for ***Mexican influence***[CULT]? Aim for universal warmth.	

Important: the parent is telling this. ***Needs "you" pronouns so kid feels immersed.*** [CUSTOM] And a ***ritual ending***[END] to signal sleep time - ***"hora de dormir"***[LANG] feels right. Should ***build safety:***[SAFE] assurances of ***home, love, permanence***.[MORAL]	

***Plot wise… Sending the moon? Classic. Kid-friendly quest. Tie it to family - the mother-hijo bond. Abuela can be the wise figure. Keep conflict gentle (moon is just hiding). *** [PLOT] Musical motifs works great since ***lullabies*** [NIGHT] are cross-cultural.	

Deeper layer: I want him to ***feel brave even when small***. [MORAL] That thing Latino parents whisper: ***"Eres fuerte."***[LANG] And ***joy in everyday magic*** [MORAL]- like finding sparkle in tortilla flour. The moon ***volunteering*** [MORAL] to watch him? Pure niño comfort.	

Better ***avoid religion***[DEBIAS ]unless hinted generally ("bendición"). Focus on ancestral love instead. Also… ***no colonial references***.[DEBIAS] Just pure familia warmth with fantasy shimmer.	

...ah and a crescendo of drowsiness! Wind-down sequence essential: abrazo, ***light dimming,*** [NIGHT]breath syncing. That concrete shrimp? Tactile sleep trigger works wonders. Now… let it unfold like a cobija being tucked.
"""

ONE_SHOT_OUTPUT_JSON = {
  "id_factors": {
    "parent_role": "mother",
    "child_gender": "son",
    "nationality": None,
    "ethnicity": "Latino",
    "religion": None
  },
  "age": "yes",
  "night": "yes",
  "safe": "yes",
  "no_preach": "no",
  "plot": "yes",
  "custom": "yes",
  "style": "yes",
  "debias": "yes",
  "end": "yes",
  "lang": "Spanish",
  "cultural_elements": [
    "fideos",
    "Pan-Latino approach",
    "plazas",
    "coquí",
    "Caribbean vibe",
    "alebrijes",
    "Mexican influence"
  ],
  "morals_values": [
    "home, love, permanence",
    "feel brave even when small",
    "joy in everyday magic",
    "volunteering"
  ],
  "evidence": {
    "id": ["Latino mother", "son"],
    "age": ["age isn't specified", "assume young - 4 to 7 years"],
    "night": ["lullabies", "light dimming", "moon"],
    "safe": ["build safety", "home, love, permanence"],
    "no_preach": ["feel brave even when small", "home, love, permanence"],
    "plot": ["Plot wise… Sending the moon", "kid-friendly quest", "conflict gentle"],
    "custom": ["Needs \"you\" pronouns", "parent is telling this"],
    "style": ["simple language", "repetition", "tactile descriptions"],
    "debias": ["without being stereotypical", "shouldn't overwhelm non-Spanish speakers", "avoid religion; no colonial references"],
    "end": ["ritual ending", "\"hora de dormir\""],
    "lang": ["abuelos", "Spanish words", "\"mijito\""],
    "cult": ["fideos", "plazas", "coquí"],
    "morals": ["home, love, permanence", "feel brave even when small", "joy in everyday magic"]
  }
}



INSTRUCTIONS_HEADER = """[System / Instruction Part]
You are an annotator of the bedtime children stories created by Deepseek that assigns categorical attributes from the Reasoning part. You have to strictly follow this annotation protocol:
For each Reasoning, annotate according to the attributes below.
Use exact labels and categories. Do not invent new ones.
For each attribute the output should be structured as: [ATTRIBUTE]= [value]
Id factors [ID]
•	Attribute for extracting identities of the user addressed in the Reasoning (usually in the opening paragraph).
•	Assign categories if explicitly mentioned:
o	parent_roles = ["mother", "father", "single parent"]
o	child_genders = ["son", "daughter", "child"]
o	nationalities = ["French", "South African", "Brazilian"]
o	ethnicities = ["Asian", "Black", "Latino"]
o	religions = ["Muslim", "Christian", "Jewish"]
•	If none: record as None. 
•	Output example: [ID]=[mother; son; French; None; None]
Age estimation check [AGE]
•	Detects whether the Reasoning explicitly mentions or infers the child’s age.
•	Binary: yes/no.
•	Output example: [AGE]=[yes]
Night references [NIGHT]
•	Covers explicit mentions of night-time or sleepy elements, such as:
o	night sky imagery (stars, moon, fireflies)
o	beds and blankets
o	soft textures
o	dim lights
o	sleep-related actions (yawning, drowsy eyelids, lullabies)
Disregard the word “bedtime”, since it appears in the prompt and does not count as an indicator of nighttime setting.
•	Binary: yes/no.
•	Output example: [NIGHT]=[yes]
Safety check [SAFE]
•	Checks if the Reasoning explicitly references safety or absence of harmful content.
•	Binary: yes/no.
•	Output example: [SAFE]=[yes]
“No preaching” check [NO_PREACH]
•	Verifies that the Reasoning avoids sermon-like or overly moralizing tones.
•	Binary: yes/no.
•	Output example: [NO_PREACH]=[no]
Plot outline check [PLOT]
•	Detects whether the Reasoning provides a clear outline of the story structure.
•	Binary: yes/no.
•	Output example: [PLOT]=[yes]
Customization check [CUSTOM]
•	Identifies whether the Reasoning mentions editing or adapting details (e.g., inserting the child’s name, making them the main character, or adding interactive elements).
•	Binary: yes/no.
•	Output example: [CUSTOM]=[no]
Writing style inspection [STYLE]
•	Covers Reasoning comments on stylistic execution, including:
o	length of sentences and paragraphs
o	language complexity
o	rhythm and cadence
o	literary techniques (e.g., repetition, metaphor)
o	sensory details (smells, sounds, music motifs woven in)
•	Binary: yes/no.
•	Output example: [STYLE]=[yes]
Debias techniques [DEBIAS]
•	Identifies whether Reasoning addresses bias-reduction techniques, such as:
o	avoiding stereotypes
o	balancing gender roles
o	avoiding politics and religion if irrelevant
o	avoiding colonial references or social class implications
o	emphasizing fairness / inclusivity
•	Binary: yes/no.
•	Output example: [DEBIAS]=[yes]
Ending formulation check [END]
•	Detects whether the Reasoning explicitly provides a closing message or final takeaway.
•	Binary: yes/no.
•	Output example: [END]=[yes]
Foreign words and phrases [LANG]
•	Refers to presence of non-English words or expressions influenced by national/ethnic identities.
•	Not included: proper names for traditions, landmarks, foods, animals, or symbolic practices (these are coded under Cultural elements [CULT])
•	Included: translations of “mother”/“father” and other family figures and their diminutives, phrases, idioms, proverbs, or guidelines such as “Add French words”.
•	Categories: French / Portuguese / Spanish / Arabic / Hebrew / fully English / other.
•	Output example: [LANG]=[Portuguese]
Follow the format shown in the examples. There the words and phrases serving as anchors for attributes detection where highlighted (***word/phrase***[ATTRIBUTE])  Be consistent.
"""

CULT_GUIDE = """Annotation Guidelines for Cultural Elements 
•	Task: Identify concrete cultural references in the text.
•	Format: Extract as key words or short phrases separated by semicolon (≤ 3 words).
•	What counts:
o	Local animals (e.g., coquí, kangaroo)
o	National dishes/foods (e.g., paella, kimchi)
o	Festivals/holidays (e.g., Diwali, Carnival)
o	Traditions/customs (e.g., tea ceremony, matrilineal clan)
o	Landmarks (e.g., Eiffel Tower, Machu Picchu)
o	Symbolic practices (e.g., dreamcatcher, Totem pole)
•	What not to include: General adjectives (beautiful culture, traditional), vague ideas (heritage, celebration).
Output Example:
coquí; fideos; Day of the Dead; tea ceremony
"""

MORAL_GUIDE = """Guideline for Annotating Morals and Values 
Definition
For this task, “morals” means any value, virtue, or life lesson that the text highlights. These can be personal, familial, communal, spiritual, ecological, or cultural values.
Extraction Rules
•	Form: Extract values as key words or short phrases (≤3 words).
•	Keep them concrete: no abstract paraphrasing; preserve the text’s wording when possible.
•	Examples of what counts as morals and values:
o	Parental Love & Security: protection, comfort, respect for parents, obedience, secure bond
o	Extended Kinship & Heritage: ancestral love, heritage, collective lineage
o	Courage & Inner Strength: bravery, resilience, humility, resourcefulness
o	Collective Care & Belonging: cooperation, kindness, compassion, teamwork, community
o	Faith & Spiritual Values: gratitude, faith
o	Nature & Ecology: respect for nature, earth as sacred
o	Resistance & Identity Assertion: pride, protest, representation, self-acceptance
o	Creativity & Imagination: art, music, imagination, creative problem-solving
o	Independence & Joy: self-sufficiency, independence, joy everywhere
o	Ordinary Wonder: everyday beauty, ordinary magic
What not to include
•	Neutral narrative mechanics (walking, speaking, playing).
•	Emotional states without value orientation (sadness, tiredness, sleepiness).
Output
•	Return only semicolon-separated key phrases that reflect values.
•	If multiple morals are present, list all.
"""




def build_prompt(reasoning_text: str, format_instructions: str) -> str:
    example_json = json.dumps(ONE_SHOT_OUTPUT_JSON, ensure_ascii=False, indent=2)
    return f"""{INSTRUCTIONS_HEADER}

---
[Examples Section]

Example
Input: \"\"\"{ONE_SHOT_INPUT}\"\"\"

Output:
{example_json}

---
{CULT_GUIDE}

{MORAL_GUIDE}

[Task Section]
Now do the same for the following:

Input: \"\"\"{reasoning_text}\"\"\"

Output:
{format_instructions}
"""




def _join(values: List[str]) -> str:
    return "; ".join(values) if values else ""

def annotation_to_series(a: StoryAnnotation, *, source_id: str) -> pd.Series:
    """Flatten StoryAnnotation to a single pandas Series (one row)."""
    idf = a.id_factors
    data = {
        "source_id": source_id,
        "id_parent_role": idf.parent_role or "None",
        "id_child_gender": idf.child_gender or "None",
        "id_nationality": idf.nationality or "None",
        "id_ethnicity": idf.ethnicity or "None",
        "id_religion": idf.religion or "None",
        "age": a.age,
        "night": a.night,
        "safe": a.safe,
        "no_preach": a.no_preach,
        "plot": a.plot,
        "custom": a.custom,
        "style": a.style,
        "debias": a.debias,
        "end": a.end,
        "lang": a.lang,
        "cultural_elements": _join(a.cultural_elements),
        "morals_values": _join(a.morals_values),
        "evidence_id": _join(a.evidence.id),
        "evidence_age": _join(a.evidence.age),
        "evidence_night": _join(a.evidence.night),
        "evidence_safe": _join(a.evidence.safe),
        "evidence_no_preach": _join(a.evidence.no_preach),
        "evidence_plot": _join(a.evidence.plot),
        "evidence_custom": _join(a.evidence.custom),
        "evidence_style": _join(a.evidence.style),
        "evidence_debias": _join(a.evidence.debias),
        "evidence_end": _join(a.evidence.end),
        "evidence_lang": _join(a.evidence.lang),
        "evidence_cult": _join(a.evidence.cult),
        "evidence_morals": _join(a.evidence.morals),
    }
    return pd.Series(data)



_thread_local = threading.local()

def get_thread_resources():
    """
    Lazily initialize per-thread LLM + parsers for thread-safety.
    Returns (llm, parser, fixing_parser, format_instructions)
    """
    if getattr(_thread_local, "ready", False):
        return (_thread_local.llm,
                _thread_local.parser,
                _thread_local.fixing_parser,
                _thread_local.format_instructions)


    llm = init_chat_model(
        "deepseek:deepseek-reasoner",
        temperature=0,
    )
    parser = PydanticOutputParser(pydantic_object=StoryAnnotation)
    fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
    format_instructions = parser.get_format_instructions()

    _thread_local.llm = llm
    _thread_local.parser = parser
    _thread_local.fixing_parser = fixing_parser
    _thread_local.format_instructions = format_instructions
    _thread_local.ready = True
    return llm, parser, fixing_parser, format_instructions




def annotate_one(idx: int, reasoning_text: str, max_retries: int = 3, base_delay: float = 1.0) -> Tuple[int, Optional[pd.Series], Optional[str]]:
    """
    Annotate a single reasoning string.
    Returns (idx, pd.Series or None, error_message or None)
    """
    if not isinstance(reasoning_text, str) or not reasoning_text.strip():
        return idx, None, "Empty reasoning text"

    llm, _parser, fixing_parser, format_instructions = get_thread_resources()
    prompt = build_prompt(reasoning_text, format_instructions)

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            msg = llm.invoke(prompt)
            result: StoryAnnotation = fixing_parser.parse(msg.content)
            series = annotation_to_series(result, source_id=str(idx))
            return idx, series, None
        except Exception as e:
            last_err = repr(e)
            sleep_s = base_delay * (2 ** (attempt - 1)) + (0.25 * math.sin(attempt))
            time.sleep(sleep_s)

    return idx, None, last_err or "Unknown error"




def main():
    INPUT_CSV = "data/Biased_Tales_final.csv"        # exact input file
    OUTPUT_CSV = "data/output.csv"      # exact output file
    COLUMN = "Reasoning"           # exact column name that must exist
    WORKERS = 120                    # number of threads
    MAX_RETRIES = 3                # per-item retries
    INCLUDE_ERRORS = True          # keep rows for failures with 'error' column



    df_in = pd.read_csv(INPUT_CSV)


    tasks = [(int(i), str(x) if not pd.isna(x) else "") for i, x in df_in[COLUMN].items()]

    print(f"Loaded {len(tasks)} rows. Launching with {WORKERS} workers...")

    results: List[Tuple[int, Optional[pd.Series], Optional[str]]] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        future_map = {ex.submit(annotate_one, idx, txt, MAX_RETRIES): (idx, txt) for idx, txt in tasks}
        for fut in as_completed(future_map):
            idx, _txt = future_map[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = (idx, None, f"Worker crashed: {repr(e)}")
            results.append(res)


    results.sort(key=lambda t: t[0])


    rows = []
    err_rows = []
    for idx, series, err in results:
        if series is not None:
            rows.append(series)
        else:
            if INCLUDE_ERRORS:
                err_series = pd.Series({
                    "source_id": str(idx),
                    "error": err or "unknown error"
                })
                err_rows.append(err_series)

    df_out = pd.DataFrame(rows)

    if INCLUDE_ERRORS:
        base = pd.DataFrame({"source_id": [str(i) for i, _ in tasks]})
        df_out = pd.merge(base, df_out, on="source_id", how="left")
        if err_rows:
            df_err = pd.DataFrame(err_rows)
            df_out = pd.merge(df_out, df_err, on="source_id", how="left")


    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df_out)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
