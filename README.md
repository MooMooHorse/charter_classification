# Charter Study

## Datasets

The orginal data directory is structured as follows:

```
data/
├── ChartersPanelCCG.csv
├── ChartersPanelCCG.json
├── G and CCG.xlsx
├── LabeledDataCCG.csv
├── LabeledDataCCG.json
├── processed_data.csv
├── small_dataset.csv
├── src
│   ├── __pycache__
│   │   └── data_manager.cpython-310.pyc
│   └── data_manager.py
└── zips
    ├── dataverse_files.zip
    └── hand_labelled.zip
```
*Due to something I agreed upon when I downloaded the dataset and I can't remember if there is a penalty to share all data, I just leave it not uploaded. But the necessary data are all in processed data and small dataset tables (which are quite large, still, I will remove them if it's against any policies).*

## Steps

1. in `genai/` directory, create a file named `config.json`, with the following format:

```json
{
    "api_key" : "openai-api-key"
}
```

2. Install all dependencies

```
pip3 install -r requirements.txt
```


## Functionalities

At home directory, you can run either of the following function to start evaluation.

```bash
python3 eval/src/annotation.py
```

```python
def evaluate_self_consitency(start = 0, num_charters_to_evaluate = 100):
    evaluate_study_small_dataset(version='self-consistency', start_row=start, k=num_charters_to_evaluate, temperature=0.8, max_iter=2)

def evaluate_few_shots_CoT(start = 0, num_charters_to_evaluate = 100):
    evaluate_study_small_dataset(version='few-shots-CoT', start_row=start, k=num_charters_to_evaluate, temperature=0, max_iter=3)

def evaluate_baseline(start = 0, num_charters_to_evaluate = 100):
    evaluate_study_small_dataset(version='baseline', start_row=start, k=num_charters_to_evaluate, temperature=0.7, max_iter=3)

if __name__ == '__main__':
    evaluate_baseline()
```

You can also apply the difference evaluation by running

```bash
python3 eval/src/case-study.py
```

Many other functionalities are available, but due to the time constraint and it's a starter task, I will leave documentation for these APIs as to-do.


## Designs & Experimentations

[click on link for my post to know more](https://www.notion.so/utline-d72cb50135334356bc9d36f560f4aa20)

## Code Structure

```
./
├── README.md
├── data (data manager and data)
├── docs (documentation to dataset)
├── eval (evaluation source code & results)
├── genai (open-ai api helper and prompts)
├── paths.py (all paths for the proj)
├── requirements.txt (pre-requisite of the project)
└── utils (some utilities needed)
```

## Code-Review-Wise design

* The evaluation is constructed in a way that maximizes the reuse of code, and all necessary **parameters are exposed**. 
* **Error handling** is also implemented in case openAI api failed (which can be frequent) during the execution. 
* Operations are **logged**. Metadata and logs are stored in `genai/logs`.
* Proper amount of **headers and comments** are added.
* Prompts uses **JSON format as GPT's IO**, reaching an unified format. They are put in `genai/prompts`, so that there are no hard-coded magic texts in code.

Object Oriented Programming is **used but not abused**: 

The auxiliary functionalities are the following objects

* A data manager managing the data.
* An open-AI interface (OpenAIitf) maintaining the communication to openAI endpoint.


