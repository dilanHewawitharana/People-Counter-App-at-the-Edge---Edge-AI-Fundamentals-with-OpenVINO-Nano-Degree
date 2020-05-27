"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
import statistics 

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

check_in_cnt_check = 0;
check_out_cnt_check = 0;

timer_check = False
current_count = 0
total_count = 0

frame_count = 0
duration = 0


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser

def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client

def draw_boxes(frame, result, prob_threshold, width, height):
    '''
    Draw bounding boxes onto the frame.
    '''  
    probs = result[0, 0, :, 2]
    for i, p in enumerate(probs):
        if p > prob_threshold:
            box = result[0, 0, i, 3:]
            p1_x = int(box[0] * width)
            p1_y = int(box[1] * height)
            p2_x = int(box[2] * width)
            p2_y = int(box[3] * height)
            centre_x = (p1_x + p2_x)/2
            centre_Y = (p1_y + p2_y)/2
            check_people_in(frame, centre_Y) #check human come into frame
            check_people_out(frame, centre_x) #cehck human go out from frame
            frame = cv2.rectangle(frame, (p1_x, p1_y), (p2_x, p2_y), (0, 0, 255), 2)

def check_people_in(frame, centre_Y):
    height = frame.shape[0]
    width = frame.shape[1]
    bottom_margin = int(height*0.7)
    #cv2.line(frame, (0, bottom_margin), (width, bottom_margin), (0, 255, 0), thickness=2)
        
    global check_in_cnt_check
    global current_count
    
    if(check_in_cnt_check > 0): #skip 50 frames for avoid multiple count same human
        check_in_cnt_check -= 1
    elif(centre_Y > bottom_margin): #detect human come to frame and counter update
        current_count += 1
        check_in_cnt_check = 50

        
def check_people_out(frame, centre_x):
    height = frame.shape[0]
    width = frame.shape[1]
    right_margin = int(width*0.85)
    #cv2.line(frame, (right_margin, 0), (right_margin, height), (0, 255, 0), thickness=2)
    
    global check_out_cnt_check
    global current_count
    global total_count
    
    if(check_out_cnt_check > 0): #skip 50 frames for avoid multiple count same human
        check_out_cnt_check -= 1
    elif(centre_x > right_margin): #detect human go out from frame and counter update
        current_count -= 1
        total_count += 1
        check_out_cnt_check = 50

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    model = args.model
    
    DEVICE = args.device
    CPU_EXTENSION = args.cpu_extension
    
    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(model, CPU_EXTENSION, DEVICE)
    net_input_shape = infer_network.get_input_shape()
    
    request_id=0
    
    ### TODO: Handle the input stream ###
    # Camera input stream
    if args.input == 'CAM':
        input_validated = 0

    # Image input
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') :
        single_image_mode = True
        input_validated = args.input

    # Video input stream
    else:
        input_validated = args.input
        assert os.path.isfile(args.input), "file doesn't exist"
        
    cap = cv2.VideoCapture(input_validated)
    cap.open(input_validated)
    
    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))

    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break

        ### TODO: Pre-process the image as needed ###
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        
        ### TODO: Start asynchronous inference for specified request ###
        infer_network.exec_net(p_frame, request_id)

        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:
            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output()
            # Draw the output mask onto the input 
            draw_boxes(frame, result, prob_threshold, width, height)
        
            #Time duration calculate
            global frame_count
            global current_count
            global duration
            global timer_check
            average_duration = None
            if (current_count > 0): #Timer on if there are people detedted
                frame_count += 1
                timer_check = True
            elif (current_count == 0) and (timer_check == True): #Timer off when no people detected
                timer_check = False
                frame_count += 1
                duration += frame_count
                frame_count = 0
                average_duration = (duration/cap.get(cv2.CAP_PROP_FPS))/total_count

            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            
            client.publish('person',
                           payload=json.dumps({
                               'count': current_count, 'total': total_count}),
                           qos=0, retain=False)
            if average_duration is not None:
                client.publish('person/duration',
                               payload=json.dumps({'duration': average_duration}),
                               qos=0, retain=False)
                
            cv2.putText(frame, "Current Count : %d" %current_count, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, "Total Count : %d" %total_count, (15, 35), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, "fps : %d" %cap.get(cv2.CAP_PROP_FPS), (15, 55), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

        ### TODO: Send the frame to the FFMPEG server ###
        ### TODO: Write an output image if `single_image_mode` ###
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
            
    # Release the capture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    ### TODO: Disconnect from MQTT
    client.disconnect()

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
