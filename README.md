# Question Answering System using SQuAD

## Overview
This repository contains a question-answering system built using the Hugging Face Transformers library. The system leverages a pre-trained **DistilBERT** model fine-tuned on the **SQuAD (Stanford Question Answering Dataset)** dataset to answer questions based on a given context. The model extracts answers from paragraphs of text in response to user queries, demonstrating the power of modern transformer models for natural language understanding.

## Features
- **Pre-trained DistilBERT Model**: Uses a state-of-the-art, lightweight transformer model for question-answering tasks.
- **SQuAD Dataset**: The model is tested on the **SQuAD** dataset, a widely used benchmark for question-answering systems.
- **Pipeline Integration**: Hugging Face's `pipeline` API is used for simple and efficient question-answering.
- **Custom Dataset Support**: The system can be extended to use custom datasets for inference or fine-tuning.

## Installation

To run the code, clone the repository and install the required dependencies. The environment should have Python 3.7+ and `pip` installed.

### Clone the repository

```bash
git clone https://github.com/ahmdmohamedd/qa-system-squad.git
cd qa-system-squad
```

### Install dependencies

It is recommended to create a virtual environment for the project. You can use **conda** or **venv**.

If you're using conda, create an environment with:

```bash
conda create --name qa_system python=3.8
conda activate qa_system
```

Then, install the dependencies:

```bash
pip install transformers datasets torch
```

## Dataset

The dataset used in this system is the **SQuAD (Stanford Question Answering Dataset)**, which can be downloaded from the following link:

[SQuAD Dataset](https://rajpurkar.github.io/SQuAD-explorer/)

The dataset contains question-answer pairs based on paragraphs of text. The model is trained and tested on this dataset to extract answers to questions.

## Usage

1. **Import Libraries**: Import the necessary libraries, including `transformers` and `datasets` to load and process the dataset, and `pipeline` to run the question-answering task.

2. **Running the Model**: The system uses the Hugging Face `pipeline` API to run the question-answering task. You provide a question and a context, and the system will return the extracted answer.

### Example

```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

# Load pre-trained model and tokenizer
model_name = "distilbert-base-uncased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Create the question-answering pipeline
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Example context and question
context = "The capital of France is Paris, known for the Eiffel Tower."
question = "What is the capital of France?"

# Get the answer using the pipeline
answer = qa_pipeline(question=question, context=context)

print(f"Question: {question}")
print(f"Answer: {answer['answer']}")
```

### Example Output

```
Question: What is the capital of France?
Answer: Paris
```

## Fine-Tuning the Model (Optional)

To fine-tune the model on a custom dataset, you can modify the script to train the model on your dataset of choice. The `Trainer` class from Hugging Face can be used for this purpose.

Ensure that your dataset is formatted properly with fields for `context`, `question`, `answer`, and `answer_start`. For fine-tuning, refer to the Hugging Face [training documentation](https://huggingface.co/docs/transformers/training).

## Contributing

Contributions are welcome! Feel free to fork the repository, make changes, and submit pull requests. If you find any bugs or have suggestions for improvements, please open an issue.
