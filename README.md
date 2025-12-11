# NAO Zoologist — SIR Group 15

This repository contains a robotics project built by SIR Group 15. We control a NAO robot using the SIC library and supporting demos and scripts. The robot behaves as a "zoologist": it interacts with users, performs gestures, plays an animal-guessing game, and attempts to identify animals using a tailored dataset.

## Contents
 - `demos/` — Example demo programs provided for the project (desktop and NAO demos).
 - `scripts/` — Scripts written by our team (webhooks, dialogflow helpers and the animal guessing game notebook).
 - `gestures/` — Gesture assets used by the robot (recorded motion files and instructions).
 - `animal_dataset.csv` — The dataset we adapted/curated for the animal-guessing game (based on online sources and tailored to our use case).
 - `lib/` — Third-party libraries included with the repo (e.g., PyTurboJPEG wrapper and Redis wheel used by demos).
 - `conf/` — Configuration files and keys used for demos and local testing (sensitive files should be kept private).

High-level goal
The project demonstrates a NAO robot configured to behave like a zoologist that:
 - Greets and converses with users (dialogflow and local conversation demos included).
 - Plays an interactive animal-guessing game (the robot asks questions and tries to guess the animal you think of).
 - Performs gestures and motions relevant to the animals it describes.
 - Uses a curated `animal_dataset.csv` for game logic and classification/lookup.

Quick start
1. Requirements (assumptions)
   - Python 3.8+ (the project was developed with Python; adjust per your environment).
   - Access to a NAO robot or a simulator with the NAOqi SDK (if you plan to run NAO demos).
   - Optional: Redis if using conversation or caching demos included in `conf/`.
   - The SIC library and any NAO/robot-specific SDKs required to control motions and speech. If you don't have SIC installed system-wide, follow your local robotics environment setup.

2. Install dependencies (example)
```bash
# create a venv (recommended)
python3 -m venv .venv
source .venv/bin/activate

pip install . 
# installs needed packages see pyproject.toml
```

## Running code
 - To run NAO demos, ensure your NAO robot is powered on and connected to the same network as your development machine.
 - Remain in the root project directory and execute the scripts to launch guessing game and the webhook server.
  
  Prerequisites:
  - Redis
  - Ngrok

Prepare Redis and dialogflow:
```bash
redis-server start conf/redis/redis.conf
```
```bash
run-dialogflow-cx
```

Launch the webhook server:
```bash
python scripts/webhookv2/run.py
```

Expose webhook to the internet using ngrok:
```bash
ngrok http 8080
```
This will give you a public URL to set as your Dialogflow CX webhook
Now you can test using DialogFlow CX console or on the NAO robot using the `scripts/nao_dialogflow_gemini.py` script (or `scripts/nao_dialogflow_cx.py` for non-Gemini versions).

Host live

## File descriptions
Project scripts
 - `scripts/animal_guessing_game.ipynb` — Jupyter notebook showcasing the logic / dataset exploration for the animal-guessing game.
 - `scripts/new_webhook.py` — A webhook script used by some demos to integrate with external services.
 - `scripts/nao_dialogflow_cx.py` — Example integration with Dialogflow CX for conversational flows.

Dataset
 - `animal_dataset.csv` is a dataset tailored by our team, adapted from public online datasets. It contains animals, features/questions, and responses used by the guessing game logic. Keep in mind licensing of upstream sources when reusing or redistributing data.

Development notes
 - Gestures: The `gestures/` directory contains motion recordings (named per gesture) used by NAO. Use the SIC library or NAOqi motion API to play them back.
 - Libraries: `lib/` includes helper libraries and wheels used by demos; prefer installing these via PyPI if possible for maintainability.
 - Configuration: `conf/` contains local config and example keys. Do not commit private keys to public repositories; this repo contains some local test certs and example configs for convenience.

How the animal-guessing game works
 - The robot asks a series of questions derived from `animal_dataset.csv` to narrow the set of possible animals.
 - Answers (yes/no/unknown) filter the dataset until the robot guesses the animal.
 - The robot uses speech and gestures to make the experience engaging.

Contact
 - For questions about the code or the demos, contact the project owners in the SIR Group 15 team.
