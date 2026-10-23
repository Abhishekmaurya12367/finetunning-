# Gemini Fine-Tuning Project

A Python project for supervised fine-tuning of a supported
Gemini model using Google Cloud Agent Platform.

## Setup

1. Create and select a Google Cloud project.
2. Enable billing and the required APIs.
3. Install Python dependencies:

   python -m pip install -r requirements.txt

4. Authenticate:

   gcloud auth application-default login

5. Configure `.env` with your project ID and bucket name.

## Prepare and validate data

python src/prepare_dataset.py
python src/validate_dataset.py

## Upload data

python src/upload_dataset.py

## Fine-tune

python src/finetune.py

## Test the tuned model

python src/test_model.py