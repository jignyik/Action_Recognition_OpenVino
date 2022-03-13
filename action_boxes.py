import numpy as np
import cv2
from module import Module
import math


class ActionBoxes(Module):
    """
    INPUT
    BGR
    [3, 480, 640, 3]
    NHWC

    OUTPUT
    1, 1, N, 7
    N: number of detected bounding boxes
    7: [image_id, label, conf, x_min, y_min, x_max, y_max]
    coordinates are in percentage of frame

    POST_PROCESS
    (3,5)
    5: [0, x_min, y_min, x_max, y_max]
    coordinates are scaled to the frame sized
    """

    def __init__(self, ie, model):
        super(ActionBoxes, self).__init__(ie, model)
        self.input_blob = next(iter(self.model.input_info))
        self.output_blob = next(iter(self.model.outputs))

        self.input_shape = self.model.input_info[self.input_blob].input_data.shape
        self.output_shape = self.model.outputs[self.output_blob].shape

        self.preds = np.array([])
        self.boxes = np.array([])
        self.box_for_action = np.array([])

    @staticmethod
    def preprocess(image):
        assert image.shape == (480, 640, 3)
        image = np.transpose(image, [2, 0, 1])
        image = np.array([image])
        return image

    def start_async(self, frame):
        inputs = self.preprocess(frame)
        self.enqueue(inputs)

    def enqueue(self, inputs):
        out = super(ActionBoxes, self).enqueue({self.input_blob: inputs})
        if out is None:
            out = np.array([])
        return out

    def postprocess(self, threshold=0.4):
        output = self.get_outputs()[0][self.output_blob].buffer
        post_output = []
        for i in output[0][0]:
            if i[1] == 1 and i[2] > threshold:
                post_output.append([i[3]*640, i[4]*480, i[5]*640, i[6]*480])
        post_output = np.array(post_output)
        self.boxes = post_output
        self.box_for_action = self.scale_boxes(self.boxes)
        return self.box_for_action

    def draw_boxes(self, frame):
        if type(self.boxes) != "None":
            if self.boxes.any():
                for i in self.boxes:
                    frame = cv2.rectangle(frame, (i[:2].astype("int")), (i[2:].astype("int")), (255, 255, 50), 2, cv2.LINE_AA)
        return frame

    def scale_boxes(self, inputs, size=256, height=480, width=640):
        """
        Scale the short side of the box to size.
        Args:
            size (int): size to scale the image.
            inputs (ndarray): bounding boxes to peform scale. The dimension is
            `num boxes` x 4.
            height (int): the height of the image.
            width (int): the width of the image.
        Returns:
            boxes (ndarray): scaled bounding boxes.
        """
        boxes = np.copy(inputs)
        if (width <= height and width == size) or (
                height <= width and height == size
        ):
            return boxes
        if width < height:
            new_height = int(math.floor((float(height) / width) * size))
            boxes *= float(new_height) / height
        else:
            new_width = int(math.floor((float(width) / height) * size))
            boxes *= float(new_width) / width

        a = boxes
        n = len(a)
        if n == 0:
            return np.array([])
        out = np.concatenate((np.zeros((n, 1)), a), axis=1).astype("float16")
        return out






