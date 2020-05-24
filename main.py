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
import numpy as np

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

#count history

count_history = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

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
    count = 0
    probs = result[0, 0, :, 2]
    for i, p in enumerate(probs):
        if p > prob_threshold:
            count += 1
            box = result[0, 0, i, 3:]
            p1 = (int(box[0] * width), int(box[1] * height))
            p2 = (int(box[2] * width), int(box[3] * height))
            frame = cv2.rectangle(frame, p1, p2, (0, 0, 255), 2)
    return count

def calculate_current_count(count): 
    # calculate the mean of latest 10 values and rounded into nearest integer
    index = 9
    while index > 0:
      count_history[index] = count_history[index-1]
      index -= 1
        
    count_history[0] = count
    mean = statistics.mean(count_history) 
    
    return int(round(mean))


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
    network_shape = infer_network.get_input_shape()

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

    width = int(cap.get(3))
    height = int(cap.get(4))

    in_shape = network_shape['image_tensor']

    #iniatilize variables
    request_id = 0
    
    current_count = 0
    pre_current_count = 0
    
    total_count = 0
    
    timer_on = False
    timer_start = None
    timer_end = None
    duration = 0

    ### TODO: Loop until stream is over ###
    while cap.isOpened():

        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break

        ### TODO: Pre-process the image as needed ###
        image = cv2.resize(frame, (in_shape[3], in_shape[2]))
        image_p = image.transpose((2, 0, 1))
        image_p = image_p.reshape(1, *image_p.shape)

        ### TODO: Start asynchronous inference for specified request ###
        net_input = {'image_tensor': image_p,'image_info': image_p.shape[1:]}
        average_duration = None
        infer_network.exec_net(net_input, request_id)

        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:

            ### TODO: Get the results of the inference request ###
            net_output = infer_network.get_output()

            ### TODO: Extract any desired stats from the results ###
            pre_current_count = current_count  
            obj_count = draw_boxes(frame, net_output, prob_threshold, width, height)
            current_count = calculate_current_count(obj_count)
    
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            
            #Total count update
            if pre_current_count > current_count:
                total_count += (pre_current_count - current_count)
            
            #Time duration calculate
            if (current_count > 0) and (timer_on == False): #Timer on if there are people detedted
                timer_on = True
                timer_start = time.time()
            elif (current_count == 0) and (timer_on == True): #Timer off when no people detected
                timer_on = False
                timer_end = time.time()
                duration += timer_end - timer_start
                average_duration = duration/total_count

            client.publish('person',
                           payload=json.dumps({
                               'count': current_count, 'total': total_count}),
                           qos=0, retain=False)
            if average_duration is not None:
                client.publish('person/duration',
                               payload=json.dumps({'duration': average_duration}),
                               qos=0, retain=False)
                
        ### TODO: Send the frame to the FFMPEG server ###

        ### TODO: Write an output image if `single_image_mode` ###
        frame = cv2.resize(frame, (768, 432))
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
        
    cap.release()
    cv2.destroyAllWindows()


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
