# you can run the gestures with this example in the demo_nao_motion.py file

```
from sic_framework.devices.common_naoqi.naoqi_leds import NaoLEDRequest

from sic_framework.devices.common_naoqi.naoqi_motion_recorder import (
    NaoqiMotionRecording,
    PlayRecording,
)

recording = NaoqiMotionRecording.load("path to file")
self.nao.motion_record.request(PlayRecording(recording))
