from rosbags.highlevel import AnyReader
from rosbags.image import message_to_cvimage
from pathlib import Path
import cv2

# Path to your ROS 2 bag directory
bag_path = Path('/media/2t/corl/bigy1/bigy1_0.db3')

# Output directory to save images
output_dir = Path('/media/2t/corl/out')
output_dir.mkdir(parents=True, exist_ok=True)

with AnyReader([bag_path]) as reader:
    for connection in reader.connections:
        if connection.topic == '/frontvideostream':
            for _, _, rawdata in reader.messages(connection):
                msg = reader.deserialize(rawdata, connection.msgtype)
                img = message_to_cvimage(msg)
                timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                image_filename = output_dir / f'image_{timestamp:.6f}.png'
                cv2.imwrite(str(image_filename), img)
