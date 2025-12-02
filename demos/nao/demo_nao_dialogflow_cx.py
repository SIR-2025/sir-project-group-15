# Import basic preliminaries
import random
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from sic_framework.core.sic_application import SICApplication
from sic_framework.core import sic_logging
from threading import Thread

# Import the device(s) we will be using
from sic_framework.devices import Nao
from sic_framework.devices.nao import NaoqiTextToSpeechRequest
from sic_framework.devices.common_naoqi.naoqi_motion import NaoqiAnimationRequest, NaoPostureRequest

# Import the service(s) we will be using
from sic_framework.services.dialogflow_cx.dialogflow_cx import (
    DialogflowCX,
    DialogflowCXConf,
    DetectIntentRequest,
    QueryResult,
    RecognitionResult,
)

# Import the LLM service from your scripts folder
from scripts.google_script import GoogleGenAI, GoogleGenAIConf, GenAIRequest

from sic_framework.devices.common_naoqi.naoqi_motion_recorder import (
    NaoqiMotionRecording,
    PlayRecording,
)

# Import libraries necessary for the demo
import json
from os.path import abspath, join
import numpy as np


class NaoDialogflowCXDemo(SICApplication):
    """
    NAO Dialogflow CX demo application with LLM integration.
    """
    
    def __init__(self):
        # Call parent constructor (handles singleton initialization)
        super(NaoDialogflowCXDemo, self).__init__()
        
        # Demo-specific initialization
        self.nao_ip = "10.0.0.127"  # TODO: Replace with your NAO's IP
        
        # Get the folder where this script is located
        script_dir = os.path.dirname(abspath(__file__))
        # Go up two levels from the script location to find the conf folder
        self.dialogflow_keyfile_path = join(script_dir, "..", "..", "conf", "google", "google-key.json")
        
        self.nao = None
        self.dialogflow_cx = None
        self.session_id = np.random.randint(10000)

        self.set_log_level(sic_logging.INFO)
        
        # --- NEW STATE VARIABLES ---
        self.genai = None
        self.turn_count = 0           # Tracks turns in the game
        self.trigger_turn = 4         # Trigger LLM after this many turns
        self.in_llm_mode = False      # Flag to switch logic
        self.llm_turns_remaining = 0  # How long the open discussion lasts
        self.llm_context = ""         # To remember the question asked
        self.llm_max_turns = 2        # How many exchanges in LLM mode
        # ---------------------------
        
        self.setup()
    
    def on_recognition(self, message):
        """Callback function for Dialogflow CX recognition results."""
        if message.response:
            if hasattr(message.response, 'recognition_result') and message.response.recognition_result:
                rr = message.response.recognition_result
                if hasattr(rr, 'is_final') and rr.is_final:
                    if hasattr(rr, 'transcript'):
                        self.logger.info("Transcript: {transcript}".format(transcript=rr.transcript))
    
    def setup(self):
        """Initialize and configure NAO robot, Dialogflow CX, and Google GenAI."""
        self.logger.info("Initializing NAO robot...")
        
        # Initialize NAO
        self.nao = Nao(ip=self.nao_ip, dev_test=False)
        nao_mic = self.nao.mic
        
        self.logger.info("Initializing Dialogflow CX...")
        
        # Load the key json file
        with open(self.dialogflow_keyfile_path) as f:
            keyfile_json = json.load(f)
        
        # Agent configuration
        # TODO: Replace with your agent details
        agent_id = "d9d2ea8b-d3ac-4965-9e3e-7ea1108528c5"
        location = "europe-west4"
        
        dialogflow_conf = DialogflowCXConf(
            keyfile_json=keyfile_json,
            agent_id=agent_id,
            location=location,
            sample_rate_hertz=16000,
            language="en"
        )
        
        # Initialize Dialogflow CX with NAO's microphone as input
        self.dialogflow_cx = DialogflowCX(conf=dialogflow_conf, input_source=nao_mic)
        self.logger.info("Initialized Dialogflow CX... registering callback function")
        self.dialogflow_cx.register_callback(callback=self.on_recognition)

        # --- GENAI SETUP ---
        self.logger.info("Initializing Google GenAI...")
        # Note: Ensure "project_id" exists in your json keyfile
        genai_conf = GoogleGenAIConf(
            keyfile_json=keyfile_json,
            project_id=keyfile_json.get("project_id"), 
            location="europe-west4",
            model_name="gemini-2.5-flash"
        )
        self.genai = GoogleGenAI(conf=genai_conf)
        # -------------------

        # --- GESTURE LOADING ---
        self.logger.info("Loading all custom gestures from 'gestures/' folder...")
        self.custom_gestures = {}
        gesture_folder = "gestures"

        if os.path.exists(gesture_folder):
            for filename in os.listdir(gesture_folder):
                if filename.startswith("."): 
                    continue
                trigger_word = filename # e.g. "run"
                full_path = os.path.join(gesture_folder, filename)
                try:
                    recording = NaoqiMotionRecording.load(full_path)
                    self.custom_gestures[trigger_word] = recording
                    self.logger.info(f"Loaded gesture: '{trigger_word}'")
                except Exception as e:
                    self.logger.error(f"Failed to load {filename}: {e}")
        else:
             self.logger.warning(f"Folder '{gesture_folder}' not found!")
    
    def run(self):
        """Main application loop."""
        try:
            # Demo starts
            self.nao.tts.request(NaoqiTextToSpeechRequest("Hello, I am Nao, nice to meet you!"))
            self.logger.info(" -- Ready -- ")
            
            while not self.shutdown_event.is_set():
                
                # --- LOGIC SWITCH ---
                if self.in_llm_mode:
                    self.handle_llm_turn()
                else:
                    self.handle_game_turn()
                    
        except KeyboardInterrupt:
            self.logger.info("Demo interrupted by user")
        except Exception as e:
            self.logger.error("Exception: {}".format(e))
            import traceback
            traceback.print_exc()
        finally:
            self.shutdown()

    def handle_game_turn(self):
        """Logic for the standard Dialogflow guessing game."""
        self.logger.info(f" ----- GAME MODE (Turn {self.turn_count}/{self.trigger_turn}) -----")
        self.logger.info(" ----- Your turn to talk!")

        # 1. Request intent detection
        reply = self.dialogflow_cx.request(DetectIntentRequest(self.session_id))
        
        # 2. Log detected intent
        intent_name = reply.intent if reply.intent else "None"
        self.logger.info(f"Detected intent: {intent_name}")

        # 3. Speak and Perform Gestures
        if reply.fulfillment_message:
            self.speak_and_gesture(reply.fulfillment_message, intent_name)
        else:
            self.logger.info("No fulfillment message")

        # 4. Update Game Logic
        # We assume "intent.answer" means a valid turn took place
        if "answer" in intent_name or "yes" in intent_name or "no" in intent_name:
            self.turn_count += 1

        # 5. Check if we should trigger the LLM Open Question
        if self.turn_count >= self.trigger_turn:
            self.trigger_open_question()

    def handle_llm_turn(self):
        """Logic for the Open Question conversation using GenAI."""
        self.logger.info(f" ----- LLM MODE ({self.llm_turns_remaining} turns left) -----")
        self.logger.info(" ----- Your turn to talk!")

        # 1. Listen (We use Dialogflow just to get the transcript)
        reply = self.dialogflow_cx.request(DetectIntentRequest(self.session_id))
        user_text = reply.transcript
        self.logger.info(f"User said (LLM Context): {user_text}")

        if user_text:
            # 2. Generate Follow-up Response
            prompt = (
                f"You are a fanatic zoologist. I asked: '{self.llm_context}'. "
                f"The child replied: '{user_text}'. "
                "Respond enthusiastically to their answer in 1 sentence."
            )
            
            self.logger.info("Requesting GenAI response...")
            genai_reply = self.genai.request(GenAIRequest(prompt))
            
            # 3. Speak Response
            self.speak_and_gesture(genai_reply.text)
            
            # 4. Update LLM Logic
            self.llm_turns_remaining -= 1
            self.llm_context = genai_reply.text # Update context for next turn

            # 5. Check if LLM session is done
            if self.llm_turns_remaining <= 0:
                self.in_llm_mode = False
                self.turn_count = 0 # Reset game counter
                self.nao.tts.request(NaoqiTextToSpeechRequest("Anyway, back to the game! Do you have a new animal?"))

    def trigger_open_question(self):
        """Generates the first open question and switches modes."""
        self.logger.info("--- TRIGGERING OPEN QUESTION ---")
        
        prompt = (
            "You are a fanatic zoologist talking to a child. "
            "Ask a fun, thought-provoking open-ended question about animals. "
            "Keep it short (max 2 sentences)."
        )
        
        genai_reply = self.genai.request(GenAIRequest(prompt))
        question = genai_reply.text
        
        self.logger.info(f"GenAI Question: {question}")
        self.speak_and_gesture(question)
        
        # Switch State
        self.in_llm_mode = True
        self.llm_turns_remaining = self.llm_max_turns
        self.llm_context = question

    def speak_and_gesture(self, text, intent=None):
        """Handles TTS and Gestures simultaneously."""
        self.logger.info(f"NAO reply: {text}")

        # --- 1. Start Speaking (Threaded) ---
        def speak_task():
            self.nao.tts.request(NaoqiTextToSpeechRequest(text))
        
        tts_thread = Thread(target=speak_task)
        tts_thread.start()

        # --- 2. Perform Gestures (Non-blocking) ---
        gesture_found = False

        # A. Custom Gestures (Trigger words)
        for word, recording in self.custom_gestures.items():
            if word.lower() in text.lower():
                self.logger.info(f"Triggering custom gesture: {word}")
                self.nao.motion_record.request(PlayRecording(recording), block=False)
                gesture_found = True
                break
        
        # B. Intent-based Gestures (Game Mode)
        if not gesture_found and intent == "greeting":
             self.logger.info("Performing wave gesture")
             self.nao.motion.request(NaoPostureRequest("Stand", 0.5), block=False)
             self.nao.motion.request(NaoqiAnimationRequest("animations/Stand/Gestures/Hey_1"), block=False)
             gesture_found = True

        # C. Generic Rules
        if not gesture_found:
            if "?" in text:
                options = [
                    "animations/Stand/Gestures/Explain_1", 
                    "animations/Stand/Gestures/Explain_2", 
                    "animations/Stand/Gestures/Explain_3"
                ]
                self.nao.motion.request(NaoqiAnimationRequest(random.choice(options)), block=False)
            elif "yipie" in text.lower():
                 self.nao.motion.request(NaoqiAnimationRequest("animations/Stand/Gestures/Enthusiastic_4"), block=False)

        # --- 3. Join Thread ---
        tts_thread.join()


if __name__ == "__main__":
    # Create and run the demo
    demo = NaoDialogflowCXDemo()
    demo.run()