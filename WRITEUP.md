# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers

Custom layers are layers that are not included in the list of known layers. If the topology contains any layers that are not in the list of known layers, the Model Optimizer classifies them as custom.
The Model Optimizer searches the list of known layers for each layer contained in the input model topology before building the model's internal representation, optimizing the model, and producing the Intermediate Representation files.
The Inference Engine loads the layers from the input model IR files into the specified device plugin, which will search a list of known layer implementations for the device. If the topology contains layers that are not in the list of known layers for the device, the Inference Engine considers the layer to be unsupported and reports an error

## Comparing Model Performance

In my case, I have downloaded several models from the TensorFlow Object Detection Model Zoo. But most of the time, It does not correctly converted to IR with the Model Optimizer. During the converting process, there are some errors. Other problem I faced is that how to illustrate result after load some specific model into Inference Engine. 

So I have to focus my attention to one specific model that give me some trustable result when I use that model with OpenVino tool kit. Since I have use only one model in this project, I could not compare the performance.

I will explain here the step by step to how I convert “faster_rcnn_inception_v2_coco_2018_01_28” model so that it can use with OpenVino tool kit.
You can go to TensorFlow Object Detection Model Zoo using following link.
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

In there you can see and download lot of pre trained models that can use for object detection. In my case I tried following models.

* faster_rcnn_inception_v2_coco
* faster_rcnn_nas
* ssdlite_mobilenet_v2_coco

But only faster_rcnn_inception_v2_coco model gives good result when convert into IR format with model optimizer.
To download and convert into IR format, I used following steps

Command to download model.
```
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
```

Then extracted using following command.
```
tar -xvf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
```

Then go to extracted folder.
```
cd faster_rcnn_inception_v2_coco_2018_01_28
```

Command to convert tensor flow model to IR format using model optimizer.
```
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json
```
- Result images are following
    ![result image01](./images/Result/1.PNG)
    ![result image02](./images/Result/2.PNG)
    ![result image03](./images/Result/3.PNG)
    ![result image04](./images/Result/4.PNG)
## Assess Model Use Cases

* Count the number of people who attend to office as an attendance system.
* Count the number of people of a large building such as airport / train station / bus station. We may need to classify number of people come to building, number of people go out from building, count the number of people come in/out in a specific time zone....etc. Depending on the user case, statics requirement can be vary
* control the traffic light based on the people count and vehicle count of a road junction.
* As a security system. We may need to control visiting human in a perticular area. Then we can warn sound when human detect in the  danger area such as animal zoo, Army camp.

## Assess Effects on End User Needs

- By reading the documentation of pre trained model, we may can get required information such as accuracy, speed, and output. 

- By applying that particular model for application and check the accuracy for different user cases.

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [faster_rcnn_inception_v2_coco]
  - [Model Source] 
    ```
    https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
    ```
  - I converted the model to an Intermediate Representation with the following arguments...

    Command to download model.
    ```
    wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
    ```

    Then extracted using following command.
    ```
    tar -xvf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
    ```

    Then go to extracted folder.
    ```
    cd faster_rcnn_inception_v2_coco_2018_01_28
    ```

    Command to convert tensor flow model to IR format using model optimizer.
    ```
    python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json
    ```

  - I was able to convert this model into IR format succefully and generate .xml and .bin file.

  - Result images are following
    ![result image01](./images/Result/1.PNG)
    ![result image02](./images/Result/2.PNG)
    ![result image03](./images/Result/3.PNG)
    ![result image04](./images/Result/4.PNG)
  
- Model 2: [faster_rcnn_nas_coco]
  - [Model Source] 
    ```
    https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
    ```
  - I converted the model to an Intermediate Representation with the following arguments...

    Command to download model.
    ```
    wget http://download.tensorflow.org/models/object_detection/faster_rcnn_nas_coco_2018_01_28.tar.gz
    ```

    Then extracted using following command.
    ```
    tar -xvf faster_rcnn_nas_coco_2018_01_28.tar.gz
    ```

    Then go to extracted folder.
    ```
    cd faster_rcnn_nas_coco_2018_01_28
    ```

    Command to convert tensor flow model to IR format using model optimizer.
    ```
    python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json
    ```

    - Could not converted into IR format properly.
    ![result image05](./images/Result/e1.PNG)

- Model 3: [ssd_mobilenet_v2_coco]
  - [Model Source] 
    ```
    https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
    ```
  - I converted the model to an Intermediate Representation with the following arguments...

    Command to download model.
    ```
    wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
    ```

    Then extracted using following command.
    ```
    tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
    ```

    Then go to extracted folder.
    ```
    cd ssd_mobilenet_v2_coco_2018_03_29
    ```

    Command to convert tensor flow model to IR format using model optimizer.
    ```
    python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json
    ```
    - Could not converted into IR format properly.
    ![result image06](./images/Result/e2.PNG)