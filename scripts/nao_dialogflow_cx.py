# Import local modules
from handle_gestures import gesture_logic, setup_gestures

# Import basic preliminaries
from sic_framework.core.sic_application import SICApplication
from sic_framework.core import sic_logging
from threading import Thread
from elevenlabs import ElevenLabs

# Import the device(s) we will be using
from sic_framework.devices import Nao
from sic_framework.devices.nao import NaoqiTextToSpeechRequest

# Posture and animation imports
from sic_framework.devices.common_naoqi.naoqi_motion import NaoPostureRequest

# Face tracking imports
from sic_framework.devices.common_naoqi.naoqi_stiffness import Stiffness
from sic_framework.devices.common_naoqi.naoqi_tracker import (
    StartTrackRequest,
)

# Import the service(s) we will be using
from sic_framework.services.dialogflow_cx.dialogflow_cx import (
    DialogflowCX,
    DialogflowCXConf,
    DetectIntentRequest,
)

# Import NAO LED control requests
from sic_framework.devices.common_naoqi.naoqi_leds import (
    NaoFadeRGBRequest,
    NaoLEDRequest,
)
import time

from sic_framework.core.message_python2 import AudioRequest

# Import libraries necessary for the demo
import json
from os.path import abspath, join
import numpy as np
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

class NaoDialogflowCX(SICApplication):
    """
    NAO Dialogflow CX demo application.
    
    Demonstrates NAO robot picking up your intent and replying according to your 
    trained Dialogflow CX agent.

    IMPORTANT:
    1. You need to obtain your own keyfile.json from Google Cloud and place it in conf/google/
       How to get a key? See https://social-ai-vu.github.io/social-interaction-cloud/external_apis/google_cloud.html
       Save the key in conf/google/google-key.json

    2. You need a trained Dialogflow CX agent:
       - Create an agent at https://dialogflow.cloud.google.com/cx/
       - Add intents with training phrases
       - Train the agent
       - Note the agent ID and location

    3. The Dialogflow CX service needs to be running:
       - pip install social-interaction-cloud[dialogflow-cx]
       - run-dialogflow-cx

    Note: This uses Dialogflow CX (v3), which is different from Dialogflow ES (v2).
    """
    
    def __init__(self, enhanced_voice=False):
        # Call parent constructor (handles singleton initialization)
        super(NaoDialogflowCX, self).__init__()
        
        # Demo-specific initialization
        self.nao_ip = "10.0.0.154"  # TODO: Replace with your NAO's IP address 10.15.2.86 / 10.0.0.245 / 169.254.195.109 / 10.0.0.127
        # Get the folder where this script is located
        # Go up two levels from the script location to find the conf folder
        self.dialogflow_keyfile_path = join(BASE_DIR, "conf", "google", "google-key.json")
        self.nao = None
        self.dialogflow_cx = None
        self.session_id = np.random.randint(10000)

        self.set_log_level(sic_logging.INFO)
        
        self.enhanced_voice = enhanced_voice
        
        # Log files will only be written if set_log_file is called. Must be a valid full path to a directory.
        # self.set_log_file("/Users/apple/Desktop/SAIL/SIC_Development/sic_applications/demos/nao/logs")
        
        self.setup()
    
    def on_recognition(self, message):
        """
        Callback function for Dialogflow CX recognition results.
        
        Args:
            message: The Dialogflow CX recognition result message.
        
        Returns:
            None
        """
        if message.response:
            if hasattr(message.response, 'recognition_result') and message.response.recognition_result:
                rr = message.response.recognition_result
                if hasattr(rr, 'is_final') and rr.is_final:
                    if hasattr(rr, 'transcript'):
                        self.logger.info("Transcript: {transcript}".format(transcript=rr.transcript))
                        
    def speak_task(self, text):
        if self.enhanced_voice:
            speech = self.tts_client.text_to_speech.convert(
                            voice_id="tnSpp4vdxKPjI9w0GnoV",
                            output_format="pcm_16000",
                            text=text,
                            model_id="eleven_multilingual_v2"
                        ) # returns bytes of audio file
                        
            sample_rate = 16000
            speech_bytes = b"".join(speech)

            message = AudioRequest(sample_rate=sample_rate, waveform=speech_bytes)
            self.nao.speaker.request(message)
        else:
            tts_request = NaoqiTextToSpeechRequest(text=text)
            self.nao.tts.request(tts_request)
        
    def setup(self):
        """Initialize and configure NAO robot and Dialogflow CX."""
        self.logger.info("Initializing NAO robot...")
        
        # Initialize NAO
        self.nao = Nao(ip=self.nao_ip)
        nao_mic = self.nao.mic
        
        self.logger.info("Initializing Dialogflow CX...")
        
        # Load the key json file
        with open(self.dialogflow_keyfile_path) as f:
            keyfile_json = json.load(f)
        
        # Agent configuration
        # agent_id = "d9d2ea8b-d3ac-4965-9e3e-7ea1108528c5"  # Test agent
        agent_id = "4447968a-ea99-4077-9ad3-5a3a0f127b7b"  # Main agent
        location = "europe-west4"  # Replace with your agent location if different
        # Create configuration for Dialogflow CX
        dialogflow_conf = DialogflowCXConf(
            keyfile_json=keyfile_json,
            agent_id=agent_id,
            location=location,
            sample_rate_hertz=16000,  # NAO's microphone sample rate
            language="en",
        )
        
        # Initialize Dialogflow CX with NAO's microphone as input
        self.dialogflow_cx = DialogflowCX(conf=dialogflow_conf, input_source=nao_mic)
        
        self.logger.info("Initialized Dialogflow CX... registering callback function")
        # Register a callback function to handle recognition results
        self.dialogflow_cx.register_callback(callback=self.on_recognition)
        
        # Setup ElevenLabs for TTS
        with open(abspath(join(BASE_DIR, "conf", "elevenlabs", "api-key.json"))) as f:
            keyfile_json = json.load(f)
            elvenlabs_api_key = keyfile_json["EL_API_KEY"]
        
        self.tts_client = ElevenLabs(base_url="https://api.elevenlabs.io", api_key=elvenlabs_api_key)
        
        # Load custom gestures
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
        """Main application loop."""
        try:    
            self.logger.info(" -- Ready -- ")
            
            # start tracking face
            self.logger.info("Starting face tracking...")
            self.nao.stiffness.request(Stiffness(stiffness=1.0, joints=["Head"]))
            self.nao.motion.request(NaoPostureRequest("Stand", 0.5), block=False)
            self.nao.tracker.request(
                StartTrackRequest(target_name="Face", size=0.2, mode="Head", effector="None")
            )
            
            while not self.shutdown_event.is_set():
                self.logger.info(" ----- Your turn to talk!")
                Thread(target=self.control_leds, args=(True,)).start()
                
                # Request intent detection with the current session
                reply = self.dialogflow_cx.request(DetectIntentRequest(self.session_id))
                
                # Log the detected intent
                if reply.intent:
                    self.logger.info("The detected intent: {intent} (confidence: {conf})".format(
                        intent=reply.intent,
                        conf=reply.intent_confidence if reply.intent_confidence else "N/A"
                    ))

                # Speak the agent's response using NAO's text-to-speech
                if reply.fulfillment_message: 
                    self.control_leds(turn_on=False)
                    
                    text = reply.fulfillment_message
                    self.logger.info("NAO reply: {text}".format(text=text))

                    # --- STEP 2: START SPEAKING (NON-BLOCKING) ---
                    # Thread the audio so that we can perform gestures while speaking
                    tts_thread = Thread(target=self.speak_task, args=(text,))
                    tts_thread.start()
                    
                else:
                    self.logger.info("No intent detected")
                
                # Log the transcript and perform gestures
                if reply.transcript:
                    self.logger.info("User said: {text}".format(text=reply.transcript))
                
                    gesture_logic(self.nao, reply, self.logger, self.custom_gestures)

                    # Join so the loop doesn't restart until speaking is done
                    tts_thread.join()
                else:
                    self.logger.info("No fulfillment message")
                    
        except KeyboardInterrupt:
            self.logger.info("Demo interrupted by user")
        except Exception as e:
            self.logger.error("Exception: {}".format(e))
            import traceback
            traceback.print_exc()
        finally:
            self.shutdown()


if __name__ == "__main__":
    # Create and run the demo
    demo = NaoDialogflowCX(enhanced_voice=False)
    demo.run()