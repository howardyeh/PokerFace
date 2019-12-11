#
# Copyright Amazon AWS DeepLens, 2017.
#
import boto3
import os
import greengrasssdk
from threading import Timer
import time

import awscam
import cv2
from threading import Thread
import base64
import numpy as np
import tensorflow as tf 
from tf.python.saved_model.signature_def_utils_impl import predict_signature_def
from tf.python.tools import optimize_for_inference_lib

Creating a greengrass core sdk client
client = greengrasssdk.client('iot-data')

# The information exchanged between IoT and clould has
# a topic and a message body.
# This is the topic that this code uses to send messages to cloud
iotTopic = '$aws/things/{}/infer'.format(os.environ['AWS_IOT_THING_NAME'])
s3 = boto3.client('s3') #low-level functional API

bucket='deeplens-sagemaker-models-0001'
prefix="pokerface/TFartifacts/0001/output/frozen_model.pb"
path_to_pokerface = s3.get_object(Bucket=bucket, Key=prefix)
class Build_CNN:
    def __init__(self, model_filepath):

        # The file path of model
        self.model_filepath = model_filepath
        print(model_filepath)
        # Initialize the model
        self.load_graph(model_filepath = self.model_filepath)

    def load_graph(self, model_filepath):
            '''
            Lode trained model.
            '''
            print('Loading model...')
            self.graph = tf.Graph()

            with tf.gfile.FastGFile(model_filepath, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            print('Check out the input placeholders:')
            nodes = [n.name + ' => ' +  n.op for n in graph_def.node if n.op in ('Placeholder')]
            for node in nodes:
                print(node)

            with self.graph.as_default():
                # Define input tensor
                self.input = tf.placeholder(tf.float32, [None, 224, 224, 3], name="input")
                self.outpt = tf.placeholder(tf.float32,name='output')
                tf.import_graph_def(graph_def, {'input': self.input})

            self.graph.finalize()

            print('Model loading complete!')

            # Get layer names
            layers = [op.name for op in self.graph.get_operations()]
            for layer in layers:
                print(layer)
            self.sess = tf.Session(graph = self.graph)

    def inferring(self, data):

        # Know your output node name
        x =self.input
        y = self.sess.graph.get_tensor_by_name('import/output:0')
        output = self.sess.run(y, feed_dict = {x: data})

        return output


            
   
def cropFace(img, x, y, w, h):

    #Crop face
    cimg = img[y:y+h, x:x+w]

    #Convert to jpeg
    ret,jpeg = cv2.imencode('.jpg', cimg)
    face = base64.b64encode(jpeg.tobytes())

    return face


class FIFO_Thread(Thread):
    def __init__(self):
        ''' Constructor. '''
        Thread.__init__(self)

    def run(self):
        fifo_path = "/tmp/results.mjpeg"
        if not os.path.exists(fifo_path):
            os.mkfifo(fifo_path)
        f = open(fifo_path,'w')
        client.publish(topic=iotTopic, payload="Opened Pipe")
        while Write_To_FIFO:
            try:
                f.write(jpeg.tobytes())
            except IOError as e:
                continue
        


ret, frame = awscam.getLastFrame()
ret,jpeg = cv2.imencode('.jpg', frame)
Write_To_FIFO = True
def greengrass_infinite_infer_run():
    try:
        modelPath = "/opt/awscam/artifacts/mxnet_deploy_ssd_FP16_FUSED.xml"
        
        modelType = "ssd"
        input_width = 300
        input_height = 300
        prob_thresh = 0.25
        results_thread = FIFO_Thread()
        results_thread.start()

        # Send a starting message to IoT console
        client.publish(topic=iotTopic, payload="Face detection starts now")

        Load model to GPU (use {"GPU": 0} for CPU)
        mcfg = {"GPU": 1}
        model = awscam.Model(modelPath, mcfg)
        client.publish(topic=iotTopic, payload="Model loaded")
        ret, frame = awscam.getLastFrame()
        if ret == False:
            raise Exception("Failed to get frame from the stream")

        yscale = float(frame.shape[0]/input_height)
        xscale = float(frame.shape[1]/input_width)

        doInfer = True
        while doInfer:
            # Get a frame from the video stream
            ret, frame = awscam.getLastFrame()
            # Raise an exception if failing to get a frame
            if ret == False:
                raise Exception("Failed to get frame from the stream")


            # Resize frame to fit model input requirement
            frameResize = cv2.resize(frame, (input_width, input_height))

            # Run model inference on the resized frame
            inferOutput = model.doInference(frameResize)


            # Output inference result to the fifo file so it can be viewed with mplayer
            parsed_results = model.parseResult(modelType, inferOutput)['ssd']
            label = '{'
            for obj in parsed_results:
                if obj['prob'] < prob_thresh:
                    break
                xmin = int( xscale * obj['xmin'] ) + int((obj['xmin'] - input_width/2) + input_width/2)
                ymin = int( yscale * obj['ymin'] )
                xmax = int( xscale * obj['xmax'] ) + int((obj['xmax'] - input_width/2) + input_width/2)
                ymax = int( yscale * obj['ymax'] )

                #Crop face
                ################
                client.publish(topic=iotTopic, payload = "cropping face")
                try:
                    cimage = cropFace(frame, xmin, ymin, xmax-xmin, ymax-ymin)
                    lblconfidence = '"confidence" : ' + str(obj['prob'])
                    CNN=Build_CNN(path_to_pokerface)
                    expectation=CNN.inferring(image)
                    file_name = 'expectation'+'.txt'
                    response = s3.put_object(ACL='public-read', Body=expectation,Bucket=bucket,Key=file_name)
                    
                except Exception as e:
                    msg = "Crop image failed: " + str(e)
                    client.publish(topic=iotTopic, payload=msg)
                # client.publish(topic=iotTopic, payload = "Crop face complete")
                ################


                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 165, 20), 4)
                label += '"{}": {:.2f},'.format(str(obj['label']), obj['prob'] )
                label_show = '{}: {:.2f}'.format(str(obj['label']), obj['prob'] )
                cv2.putText(frame, label_show, (xmin, ymin-15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 20), 4)
            label += '"null": 0.0'
            label += '}'
            #client.publish(topic=iotTopic, payload = label)
            global jpeg
            ret,jpeg = cv2.imencode('.jpg', frame)

    except Exception as e:
        msg = "Test failed: " + str(e)
        client.publish(topic=iotTopic, payload=msg)

    # Asynchronously schedule this function to be run again in 15 seconds
    Timer(15, greengrass_infinite_infer_run).start()


# Execute the function above
greengrass_infinite_infer_run()


# This is a dummy handler and will not be invoked
# Instead the code above will be executed in an infinite loop for our example
def function_handler(event, context):
    msg = "function called: " + str(e)
    client.publish(topic=iotTopic, payload = msg)
    return
