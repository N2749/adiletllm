# AdiletLLM
machine-learning project

### What is this project

### How to launch

#### Preparations
We highly encourage you to use venv. If you don't want to then 
proceed to [launching query](#launching-query)

# TODO: how to make venv

#### Downloading dependencies
while using venv or not:
```
pip install -r requirements.txt
```

#### Launching query
To launch you need ollama package. Info on how to download it can be found on
official ollama website https://ollama.com/download

After downloading ollama you need to pull concrete model that will be used as
answering model. You can found it in `config.ini` under `models` section.

Then when you are in project root run:
```
python src/query.py
```

### Files, and their applications

`populate_database.py` - takes all documents from $DATA\_FOLDER, splits them,
and adds to the chromadb.

`query.py` - basically main file to ask questions.

### Technologies
- Chroma DB. Vector database

### Tips for contributors:
**How to generate requirements.txt**:
```
pip install pipreqs
```
```
pipreqs .
```
