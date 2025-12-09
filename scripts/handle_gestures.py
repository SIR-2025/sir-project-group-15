from sic_framework.devices.common_naoqi.naoqi_motion import NaoqiAnimationRequest, NaoPostureRequest
from sic_framework.devices.common_naoqi.naoqi_motion_recorder import PlayRecording, NaoqiMotionRecording

import random
import os

def setup_gestures(gesture_folder, logger):
    logger.info("Loading all custom gestures from 'gestures/' folder...")
    
    custom_gestures = {} # Dictionary to store name -> recording

    # check if folder exists to avoid crashing
    if os.path.exists(gesture_folder):
        for filename in os.listdir(gesture_folder):

            if filename.startswith("."): 
                continue
            
            # We assume the filename IS the trigger word (e.g. "run", "clap")
            trigger_word = filename 
            full_path = os.path.join(gesture_folder, filename)
            
            try:
                # Load it and store it in our dictionary
                recording = NaoqiMotionRecording.load(full_path)
                custom_gestures[trigger_word] = recording
                logger.info(f"Loaded gesture: '{trigger_word}'")
            except Exception as e:
                logger.error(f"Failed to load {filename}: {e}")
    else:
        logger.warning(f"Folder '{gesture_folder}' not found!")
    
    return custom_gestures

def gesture_logic(nao, reply, logger, custom_gestures):
    text = reply.transcript
    gesture_found = False
    
    # 1. Check if any of our loaded custom gestures appear in the text
    # We iterate through our dictionary keys (run, fly, clap...)
    for word, recording in custom_gestures.items():
        if word.lower() in text.lower():
            logger.info(f"Triggering custom gesture: {word}")
            nao.motion_record.request(PlayRecording(recording), block=False)
            gesture_found = True
            break

    # Perform gestures based on detected intent (non-blocking)
    if reply.intent == "greeting":
        logger.info("Welcome intent detected - performing wave gesture")

        do_builtin_gesture(nao, "Stand", logger)
        do_builtin_gesture(nao, "animations/Stand/Gestures/Hey_1", logger)

    # 2. If no custom gesture was found, check for the generic question mark
    elif not gesture_found and "?" in text:
        options = [
            "animations/Stand/Gestures/Explain_1", 
            "animations/Stand/Gestures/Explain_2", 
            "animations/Stand/Gestures/Explain_3"
        ]
        selected_anim = random.choice(options)
        logger.info(f"Playing built-in gesture: {selected_anim}")
        do_builtin_gesture(nao, selected_anim, logger)

    elif not gesture_found and "yipie" in text:
        logger.info(f"Playing celebration")
        do_builtin_gesture(nao, "animations/Stand/Gestures/Enthusiastic_4", logger)
        
def do_custom_gesture(nao, gesture_name, logger, custom_gestures):
    """Function to trigger a custom gesture by name."""
    if gesture_name in custom_gestures:
        logger.info(f"Triggering custom gesture: {gesture_name}")
        recording = custom_gestures[gesture_name]
        nao.motion_record.request(PlayRecording(recording), block=False)
    else:
        logger.warning(f"Gesture '{gesture_name}' not found in custom gestures.")
        
def do_builtin_gesture(nao, animation_name, logger):
    """Function to trigger a built-in animation gesture by name."""
    logger.info(f"Playing built-in gesture: {animation_name}")
    nao.motion.request(NaoqiAnimationRequest(animation_name), block=False)
    