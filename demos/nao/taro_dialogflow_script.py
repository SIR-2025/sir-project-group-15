# Import basic preliminaries
import random
import os
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
    
    def __init__(self):
        # Call parent constructor (handles singleton initialization)
        super(NaoDialogflowCXDemo, self).__init__()
        
        # Demo-specific initialization
        self.nao_ip = "10.0.0.127"  # TODO: Replace with your NAO's IP address 10.15.2.86 / 10.0.0.245 / 169.254.195.109 / 10.0.0.127
        # Get the folder where this script is located
        script_dir = os.path.dirname(abspath(__file__))
        # Go up two levels from the script location to find the conf folder
        self.dialogflow_keyfile_path = join(script_dir, "..", "..", "conf", "google", "google-key.json")
        self.nao = None
        self.dialogflow_cx = None
        self.session_id = np.random.randint(10000)

        self.set_log_level(sic_logging.INFO)
        
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
    
    def setup(self):
        """Initialize and configure NAO robot and Dialogflow CX."""
        self.logger.info("Initializing NAO robot...")
        
        # Initialize NAO
        self.nao = Nao(ip=self.nao_ip, dev_test=False)
        nao_mic = self.nao.mic
        
        self.logger.info("Initializing Dialogflow CX...")
        
        # Load the key json file
        with open(self.dialogflow_keyfile_path) as f:
            keyfile_json = json.load(f)
        
        # Agent configuration
        # TODO: Replace with your agent details (use verify_dialogflow_cx_agent.py to find them)
        agent_id = "d9d2ea8b-d3ac-4965-9e3e-7ea1108528c5"  # Replace with your agent ID
        location = "europe-west4"  # Replace with your agent location if different
        
        # Create configuration for Dialogflow CX
        # Note: NAO uses 16000 Hz sample rate (not 44100 like desktop)
        dialogflow_conf = DialogflowCXConf(
            keyfile_json=keyfile_json,
            agent_id=agent_id,
            location=location,
            sample_rate_hertz=16000,  # NAO's microphone sample rate
            language="en"
        )
        
        # Initialize Dialogflow CX with NAO's microphone as input
        self.dialogflow_cx = DialogflowCX(conf=dialogflow_conf, input_source=nao_mic)
        
        self.logger.info("Initialized Dialogflow CX... registering callback function")
        # Register a callback function to handle recognition results
        self.dialogflow_cx.register_callback(callback=self.on_recognition)

        # --- GESTURE LOADING ---
        self.logger.info("Loading all custom gestures from 'gestures/' folder...")
        
        self.custom_gestures = {} # Dictionary to store name -> recording
        gesture_folder = "gestures"

        # check if folder exists to avoid crashing
        if os.path.exists(gesture_folder):
            # Loop through every file in the folder
            for filename in os.listdir(gesture_folder):
                # Skip hidden system files (like .DS_Store on mac)
                if filename.startswith("."): 
                    continue
                
                # We assume the filename IS the trigger word (e.g. "run", "clap")
                # If your files have extensions (like run.motion), use: filename.split('.')[0]
                trigger_word = filename 
                
                full_path = os.path.join(gesture_folder, filename)
                
                try:
                    # Load it and store it in our dictionary
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
                self.logger.info(" ----- Your turn to talk!")
                
                # Request intent detection with the current session
                reply = self.dialogflow_cx.request(DetectIntentRequest(self.session_id))
                
                # Log the detected intent
                if reply.intent:
                    self.logger.info("The detected intent: {intent} (confidence: {conf})".format(
                        intent=reply.intent,
                        conf=reply.intent_confidence if reply.intent_confidence else "N/A"
                    ))

                # Speak the agent's response using NAO's text-to-speech
                if reply.fulfillment_message: #TODO: move this earlier
                    text = reply.fulfillment_message
                    self.logger.info("NAO reply: {text}".format(text=text))

                    def speak_task():
                        # This runs in the background
                        self.nao.tts.request(NaoqiTextToSpeechRequest(text))

                    # --- STEP 2: START SPEAKING (NON-BLOCKING) ---
                    # We start the thread, which sends the audio command immediately.
                    # The main code DOES NOT wait for this to finish.
                    tts_thread = Thread(target=speak_task)
                    tts_thread.start()
                    
                else:
                    self.logger.info("No intent detected")
                
                # Log the transcript
                if reply.transcript:
                    self.logger.info("User said: {text}".format(text=reply.transcript))
                

                    # --- GESTURE LOGIC ---
                    gesture_found = False
                    
                    # 1. Check if any of our loaded custom gestures appear in the text
                    # We iterate through our dictionary keys (run, fly, clap...)
                    for word, recording in self.custom_gestures.items():
                        if word.lower() in text.lower():
                            self.logger.info(f"Triggering custom gesture: {word}")
                            self.nao.motion_record.request(PlayRecording(recording), block=False)
                            gesture_found = True
                            break

                    # Perform gestures based on detected intent (non-blocking)
                    if reply.intent == "greeting":
                        self.logger.info("Welcome intent detected - performing wave gesture")
                        # Use send_message for non-blocking gesture execution
                        # This allows the TTS to speak while the gesture is performed
                        self.nao.motion.request(NaoPostureRequest("Stand", 0.5), block=False)
                        self.nao.motion.request(NaoqiAnimationRequest("animations/Stand/Gestures/Hey_1"), block=False)
                        #recording = NaoqiMotionRecording.load("gestures/fly")
                        #self.nao.motion_record.request(PlayRecording(recording), block=False)

                    # 2. If no custom gesture was found, check for the generic question mark
                    elif not gesture_found and "?" in text:
                        options = [
                            "animations/Stand/Gestures/Explain_1", 
                            "animations/Stand/Gestures/Explain_2", 
                            "animations/Stand/Gestures/Explain_3"
                        ]
                        selected_anim = random.choice(options)
                        self.logger.info(f"Playing built-in gesture: {selected_anim}")
                        self.nao.motion.request(NaoqiAnimationRequest(selected_anim), block=False)
                    # --- ------------------------ ---

                    elif not gesture_found and "yipie" in text:
                        self.logger.info(f"Playing celebration")
                        self.nao.motion.request(NaoqiAnimationRequest("animations/Stand/Gestures/Enthusiastic_4"), block=False)

                    # --- STEP 4: JOIN (OPTIONAL) ---
                    # Ideally, you don't need to join here, but if you want to ensure
                    # the loop doesn't restart until speaking is done:
                    tts_thread.join()
                else:
                    self.logger.info("No fulfillment message")
                
                # Log any parameters
                if reply.parameters:
                    self.logger.info("Parameters: {params}".format(params=reply.parameters))
                    
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
    demo = NaoDialogflowCXDemo()
    demo.run()