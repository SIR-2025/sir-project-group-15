# Import local modules
from handle_gestures import gesture_logic, setup_gestures

# NEW / CORRECT
import sys
import os
# Add the project root (parent of 'scripts') to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import basic preliminaries
import random
import os
import sys
import json
import numpy as np
from threading import Thread
from os.path import abspath, join

from sic_framework.core.sic_application import SICApplication
from sic_framework.core import sic_logging

# Import the device(s) we will be using
from sic_framework.devices.desktop import Desktop
from sic_framework.devices.common_desktop.desktop_microphone import MicrophoneConf

# Import the device(s)
from sic_framework.devices import Nao
from sic_framework.devices.nao import NaoqiTextToSpeechRequest
from sic_framework.devices.common_naoqi.naoqi_motion import NaoqiAnimationRequest, NaoPostureRequest
from sic_framework.devices.common_naoqi.naoqi_motion_recorder import PlayRecording

# Face tracking imports
from sic_framework.devices.common_naoqi.naoqi_stiffness import Stiffness
from sic_framework.devices.common_naoqi.naoqi_tracker import (
    RemoveTargetRequest,
    StartTrackRequest,
    StopAllTrackRequest,
)

# Import Services
from sic_framework.services.dialogflow_cx.dialogflow_cx import (
    DialogflowCX,
    DialogflowCXConf,
    DetectIntentRequest,
)
from sic_framework.core.message_python2 import AudioRequest

# Import LLM Service
from scripts.google_script import GoogleGenAI, GoogleGenAIConf, GenAIRequest
from elevenlabs import ElevenLabs

# Import NAO LED control requests
from sic_framework.devices.common_naoqi.naoqi_leds import (
    NaoFadeRGBRequest,
    NaoLEDRequest,
)
import time

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

class NaoDialogflowCX(SICApplication):
    """
    Merged NAO Application:
    - Dialogflow CX for structured game/chat
    - Google GenAI for open-ended context-aware questions
    - ElevenLabs for enhanced voice
    - Face Tracking
    """
    
    def __init__(self, enhanced_voice=False):
        super(NaoDialogflowCX, self).__init__()
        
        # Configuration
        self.nao_ip = "10.0.0.154"  # 10 0 0 154
        self.dialogflow_keyfile_path = join(BASE_DIR, "conf", "google", "google-key.json")
        self.elevenlabs_keyfile_path = join(BASE_DIR, "conf", "elevenlabs", "api-key.json")
        
        # Devices & Services
        self.nao = None
        self.dialogflow_cx = None
        self.genai = None
        self.tts_client = None
        
        self.session_id = np.random.randint(10000)
        self.set_log_level(sic_logging.INFO)
        self.enhanced_voice = enhanced_voice
        
        # --- STATE VARIABLES ---
        self.conversation_history = [] # Stores last few turns for context
        self.turn_count = 0           
        self.trigger_turn = 4          # Turns before triggering LLM
        self.in_llm_mode = False      
        self.llm_turns_remaining = 0  
        self.llm_context = ""         
        self.llm_max_turns = 2
        
        # Suspend/Resume variables
        self.suspended_game_reply = None 
        self.suspended_game_intent = None

        self.setup()
    
    def on_recognition(self, message):
        """Callback for intermediate recognition results."""
        if message.response:
            if hasattr(message.response, 'recognition_result') and message.response.recognition_result:
                rr = message.response.recognition_result
                if hasattr(rr, 'is_final') and rr.is_final:
                    if hasattr(rr, 'transcript'):
                        self.logger.info("Transcript: {transcript}".format(transcript=rr.transcript))
                        
    def speak_task(self, text):
        """
        Handles TTS (Standard or ElevenLabs). 
        This is designed to be run in a thread.
        """
        if not text:
            return

        if self.enhanced_voice and self.tts_client:
            try:
                speech = self.tts_client.text_to_speech.convert(
                            voice_id="tnSpp4vdxKPjI9w0GnoV",
                            output_format="pcm_16000",
                            text=text,
                            model_id="eleven_multilingual_v2"
                        ) 
                speech_bytes = b"".join(speech)
                message = AudioRequest(sample_rate=16000, waveform=speech_bytes)
                self.nao.speaker.request(message)
            except Exception as e:
                self.logger.error(f"ElevenLabs failed: {e}. Fallback to NaoTTS.")
                self.nao.tts.request(NaoqiTextToSpeechRequest(text))
        else:
            self.nao.tts.request(NaoqiTextToSpeechRequest(text))
        
    def setup(self):
        self.logger.info("Initializing NAO robot...")
        self.nao = Nao(ip=self.nao_ip)

        nao_mic = self.nao.mic

        # conf = MicrophoneConf()
        # self.desktop = Desktop(mic_conf=conf)
        # nao_mic = self.desktop.mic
        
        # --- Dialogflow Setup ---
        self.logger.info("Initializing Dialogflow CX...")
        with open(self.dialogflow_keyfile_path) as f:
            keyfile_json = json.load(f)
        
        # agent_id = "d9d2ea8b-d3ac-4965-9e3e-7ea1108528c5" # Main Agent 4447968a-ea99-4077-9ad3-5a3a0f127b7b // test: d9d2ea8b-d3ac-4965-9e3e-7ea1108528c5
        agent_id = "4447968a-ea99-4077-9ad3-5a3a0f127b7b"  # Main agent

        dialogflow_conf = DialogflowCXConf(
            keyfile_json=keyfile_json,
            agent_id=agent_id,
            location="europe-west4",
            sample_rate_hertz=16000,
            language="en"
        )
        self.dialogflow_cx = DialogflowCX(conf=dialogflow_conf, input_source=nao_mic)
        self.dialogflow_cx.register_callback(callback=self.on_recognition)
        
        # --- GenAI Setup ---
        self.logger.info("Initializing Google GenAI...")
        genai_conf = GoogleGenAIConf(
            keyfile_json=keyfile_json,
            project_id=keyfile_json.get("project_id"), 
            location="europe-west4",
            model_name="gemini-2.5-flash"
        )
        self.genai = GoogleGenAI(conf=genai_conf)

        # Warm up GenAI
        self.logger.info("Warming up GenAI connection...")
        try:
            self.genai.request(GenAIRequest("Hello"), timeout=10)
            self.logger.info("GenAI connection established.")
        except Exception as e:
            self.logger.warning(f"Warm-up failed: {e}")
        
        # --- ElevenLabs Setup ---
        if self.enhanced_voice:
            try:
                with open(self.elevenlabs_keyfile_path) as f:
                    el_json = json.load(f)
                self.tts_client = ElevenLabs(base_url="https://api.elevenlabs.io", api_key=el_json["EL_API_KEY"])
            except Exception as e:
                self.logger.warning(f"Failed to setup ElevenLabs: {e}")
                self.enhanced_voice = False

        # --- Gesture Setup ---
        gesture_folder = abspath(join(BASE_DIR, "scripts", "gestures"))
        self.custom_gestures = setup_gestures(gesture_folder, self.logger)
    
    def control_leds(self, turn_on=True):
        if turn_on:
             self.logger.info("Requesting Ear LEDs to turn on")
             self.nao.leds.request(NaoLEDRequest("EarLeds", True))
             time.sleep(1)
             self.nao.leds.request(NaoFadeRGBRequest("RightEarLeds", 0, 0, 1, 0))
             self.nao.leds.request(NaoFadeRGBRequest("LeftEarLeds", 0, 0, 1, 0))
        else:
            self.logger.info("Requesting Ear LEDs to turn off")
            self.nao.leds.request(NaoLEDRequest("EarLeds", False))

    def run(self):
        try:    
            self.logger.info(" -- Ready -- ")
            
            # Start Tracking
            self.logger.info("Starting face tracking...")
            self.nao.stiffness.request(Stiffness(stiffness=1.0, joints=["Head"]))
            self.nao.tracker.request(StartTrackRequest(target_name="Face", size=0.2, mode="Head", effector="None"))
            
            while not self.shutdown_event.is_set():
                if self.in_llm_mode:
                    self.handle_llm_turn()
                else:
                    self.handle_standard_turn()
                    
        except KeyboardInterrupt:
            self.logger.info("Demo interrupted by user")
        except Exception as e:
            self.logger.error("Exception: {}".format(e))
            import traceback
            traceback.print_exc()
        finally:
            self.shutdown()

    def handle_standard_turn(self):
        """Standard Dialogflow interaction loop."""
        self.logger.info(f" ----- STANDARD MODE (Turn {self.turn_count}/{self.trigger_turn}) -----")
        
        Thread(target=self.control_leds, args=(True,)).start()

        # 1. Detect Intent
        reply = self.dialogflow_cx.request(DetectIntentRequest(self.session_id))
        intent_name = reply.intent if reply.intent else "None"
        user_text = reply.transcript if reply.transcript else ""
        
        self.logger.info(f"Intent: {intent_name} | User: {user_text}")

        # 2. Update History & Counters
        if user_text:
            self.conversation_history.append({"role": "user", "text": user_text})
        
        # Increment count if meaningful interaction
        if "answer" in intent_name or "yes" in intent_name or "no" in intent_name or intent_name == "Default Fallback Intent":
            self.turn_count += 1

        # 3. Decision Logic
        if self.turn_count >= self.trigger_turn:
            self.logger.info("Trigger count reached. Suspending standard flow.")
            self.suspended_game_reply = reply.fulfillment_message
            self.suspended_game_intent = intent_name
            
            # Switch to Open Question
            self.trigger_open_question()
        else:
            self.control_leds(turn_on=False)
            # Normal Reply
            if reply.fulfillment_message:
                self.logger.info(f"NAO Reply: {reply.fulfillment_message}")
                self.conversation_history.append({"role": "robot", "text": reply.fulfillment_message})
                
                # Speak & Gesture (Threaded)
                tts_thread = Thread(target=self.speak_task, args=(reply.fulfillment_message,))
                tts_thread.start()
                
                # Use imported gesture logic for standard turns
                gesture_logic(self.nao, reply, self.logger, self.custom_gestures)
                
                tts_thread.join()
            else:
                self.logger.info("No fulfillment message received.")

    def handle_llm_turn(self):
        """Open-ended conversation loop using GenAI."""
        self.logger.info(f" ----- LLM MODE ({self.llm_turns_remaining} turns left) -----")

        Thread(target=self.control_leds, args=(True,)).start()

        # 1. Listen
        reply = self.dialogflow_cx.request(DetectIntentRequest(self.session_id))
        user_text = reply.transcript
        self.logger.info(f"User (LLM): {user_text}")

        if user_text:
            self.conversation_history.append({"role": "user", "text": user_text})

            self.control_leds(turn_on=False)

            # 2. Generate Response
            prompt = (
                f"I asked the user: '{self.llm_context}'. "
                f"The user (a child) replied: '{user_text}'. "
                "Respond enthusiastically and briefly , but make sure your response contains something engaging the user can respond to. Use simple language. " \
                "Do not ask anything that would give clues about the animal being guessed."
            )
            
            genai_reply = self.genai.request(GenAIRequest(prompt), timeout=30)
            robot_text = genai_reply.text
            self.conversation_history.append({"role": "robot", "text": robot_text})
            
            # 3. Speak & Simple Gesture
            self.speak_and_simple_gesture(robot_text)
            
            # 4. Update State
            self.llm_turns_remaining -= 1
            self.llm_context = robot_text

            if self.llm_turns_remaining <= 0:
                self.resume_standard_flow()

    def trigger_open_question(self):
        self.logger.info("--- TRIGGERING CONTEXTUAL OPEN QUESTION ---")
        
        # 1. Transition Phrase (Non-blocking speech)
        transition_phrase = "Wait! Before we continue..."
        # We start a thread for this so we can fetch GenAI while speaking
        Thread(target=self.speak_task, args=(transition_phrase,)).start()
        
        # 2. Build Context from History
        # Get last 3 turns to keep context relevant but concise
        recent_history = self.conversation_history[-6:] 
        history_str = "\n".join([f"{h['role']}: {h['text']}" for h in recent_history])
        
        prompt = (
            f"You are chatting with a child. Here is the conversation history:\n"
            f"{history_str}\n"
            "Based strictly on what we were just talking about, ask a fun, "
            "related, open-ended question. Do not ask anything that would give clues about the animal being guessed. "
            "Keep it short (max 2 sentences), and keep the language simple. " \
            "ONLY IF the history contains 2 open ended questions from the robot, THEN do not ask another open ended question. " \
            "Just say something quirky"
        )
        
        # 3. Fetch Question
        genai_reply = self.genai.request(GenAIRequest(prompt), timeout=30)
        question = genai_reply.text
        self.logger.info(f"GenAI Question: {question}")
        self.conversation_history.append({"role": "robot", "text": question})
        
        # 4. Speak Question (Wait for previous thread if needed, usually fine)
        # Note: In a robust system we'd join the previous thread, but here we assume timing works out
        self.speak_and_simple_gesture(question)
        
        # 5. Switch State
        self.in_llm_mode = True
        self.llm_turns_remaining = self.llm_max_turns
        self.llm_context = question

    def resume_standard_flow(self):
        self.in_llm_mode = False
        self.turn_count = 0 
        
        if self.suspended_game_reply:
            resume_text = f"Anyway! As I was saying: {self.suspended_game_reply}"
            self.logger.info(f"Resuming: {resume_text}")
            self.conversation_history.append({"role": "robot", "text": resume_text})
            
            tts_thread = Thread(target=self.speak_task, args=(resume_text,))
            tts_thread.start()
            
            # Use original intent for gestures if possible, otherwise skip or default
            # We can't easily recall the full 'reply' object, so we rely on text analysis or default gestures here
            # or we could construct a dummy reply object if gesture_logic requires it.
            # For simplicity, we use simple random gestures here:
            self.nao.motion.request(NaoqiAnimationRequest("animations/Stand/Gestures/Explain_1"), block=False)
            
            tts_thread.join()
            
            self.suspended_game_reply = None
        else:
            self.speak_task("Anyway, back to the game!")

    def speak_and_simple_gesture(self, text):
        """
        Used for LLM turns where we don't have Dialogflow intents for `gesture_logic`.
        Uses simple keyword matching like the demo.
        """
        tts_thread = Thread(target=self.speak_task, args=(text,))
        tts_thread.start()
        
        # Simple keyword matching for gestures
        gesture_found = False
        # Check custom gestures dict
        for word, recording in self.custom_gestures.items():
            if word.lower() in text.lower():
                self.nao.motion_record.request(PlayRecording(recording), block=False)
                gesture_found = True
                break
        
        # Fallback random gestures
        if not gesture_found:
            options = [
                "animations/Stand/Gestures/Explain_1", 
                "animations/Stand/Gestures/Explain_2", 
                "animations/Stand/Gestures/Explain_3"
            ]
            self.nao.motion.request(NaoqiAnimationRequest(random.choice(options)), block=False)
            
        tts_thread.join()

if __name__ == "__main__":
    demo = NaoDialogflowCX(enhanced_voice=True)
    demo.run()