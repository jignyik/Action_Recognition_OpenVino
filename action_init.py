import numpy as np
import cv2
from module import Module
import math


class ActionRecognition(Module):
    """
    INPUT
    BGR
    slow : [1, 3, 8, 256, 341]
    fast : [1, 3, 32, 256, 341]
    box : [3, 5]
    # Input total of 64 frames with height:480px width:640px
    # with bounding box for each human detected


    OUTPUT
    [3, 80]
    3: Same as number of bounding box provided
    80: Confidence level for each class
    """

    def __init__(self, ie, model):
        super(ActionRecognition, self).__init__(ie, model)
        self.saved_frames = []
        self.frame_length = 32
        self.sample_rate = 2
        self.slowfast_alpha = 4
        self.sqn_length = self.frame_length*self.sample_rate
        self.class_names = self.load_action_class_names()
        self.new_boxes = np.array([])
        self.mean = [0.45, 0.45, 0.45]
        self.std = [0.225, 0.225, 0.225]
        self.pred_labels = None
        self.palette = np.random.randint(64, 128, (len(self.class_names), 3)).tolist()
        self.preds = np.array([])

        self.output_blob = next(iter(self.model.outputs))

        self.ori_length = 0

    @staticmethod
    def load_action_class_names():
        path_to_csv = "ava.names"
        with open(path_to_csv) as f:
            labels = f.read().split('\n')[:-1]
        return labels

    @staticmethod
    def scale(size, image):
        """
        Scale the short side of the image to size.
        Args:
            size (int): size to scale the image.
            image (array): image to perform short side scale. Dimension is
                `height` x `width` x `channel`.
        Returns:
            (ndarray): the scaled image with dimension of
                `height` x `width` x `channel`.
        """
        height = image.shape[0]
        width = image.shape[1]
        if (width <= height and width == size) or (
                height <= width and height == size
        ):
            return image
        new_width = size
        new_height = size
        if width < height:
            new_height = int(math.floor((float(height) / width) * size))
        else:
            new_width = int(math.floor((float(width) / height) * size))
        img = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
        )
        return img.astype(np.float32)

    def preprocessing(self, frames):
        new_frames = []
        for frame in frames:
            frame = self.scale(256, frame)
            new_frames.append(frame)

        frames = new_frames
        # The mean value of the video raw pixels across the R G B channels.
        mean = [0.45, 0.45, 0.45]
        # The std value of the video raw pixels across the R G B channels.
        std = [0.225, 0.225, 0.225]
        inputs = np.array(frames).astype("float16")
        inputs = inputs / 255.0
        inputs = inputs - np.array(mean)
        inputs = inputs / np.array(std)
        inputs = np.transpose(inputs, (3, 0, 1, 2))
        inputs = np.expand_dims(inputs, 0)
        index = np.linspace(0, inputs.shape[2] - 1, 32).astype("int")
        fast_pathway = np.take(inputs, index, 2)
        index = np.linspace(0, fast_pathway.shape[2] - 1,
                            fast_pathway.shape[2] // 4).astype("int")
        slow_pathway = np.take(fast_pathway, index, 2)

        return [slow_pathway, fast_pathway]

    def start_async(self, frames, boxes):
        inputs = self.preprocessing(frames)
        self.ori_length = len(boxes)
        if len(boxes) < 3 and len(boxes) != 0:
            diff = 3 - len(boxes)
            for i in range(diff):
                a = np.zeros((5,))
                a = np.expand_dims(a, axis=0)
                boxes = np.append(boxes, a, axis=0)
        else:
            boxes = boxes[:3]
        inputs.append(boxes)
        self.enqueue(inputs)

    def enqueue(self, inputs):
        return super(ActionRecognition, self).enqueue({"slow": inputs[0], 'fast': inputs[1], 'boxes': inputs[2]})

    def postprocess(self):
        outputs = self.get_outputs()[0][self.output_blob].buffer
        self.preds = outputs
        results = self.result()
        return results

    def result(self, confidence=0.5):
        if self.preds.size != 0:
            pred_masks = self.preds > confidence
            label_ids = [np.nonzero(pred_mask)[0] for pred_mask in pred_masks]
            pred_labels = [
                [self.class_names[label_id] +str("  ") +str(self.preds[0][label_id]) for label_id in perbox_label_ids]
                for perbox_label_ids in label_ids
            ]

            if self.ori_length == 1:
                pred_labels = [pred_labels[0]]
            elif self.ori_length == 2:
                pred_labels = [pred_labels[0], pred_labels[1]]
            elif self.ori_length > 3:
                print("This model can only predict up to 3 person, extra boxes are neglected")
            self.pred_labels = pred_labels
            return pred_labels

    def draw_action_on_frame(self, frame, boxes):
        if len(boxes) > 0 and self.pred_labels is not None:
            for i, box in enumerate(boxes):
                # get minimum x and y value
                label_origin = box[:2].astype("int")
                # put text on top of the box
                for u, o in enumerate(self.pred_labels[i]):
                    # increase the height by a offset
                    label_origin[1] -= 5
                    (label_width, label_height), _ = cv2.getTextSize(o, cv2.FONT_HERSHEY_SIMPLEX, .5, 2)
                    cv2.rectangle(
                        frame,
                        (label_origin[0], label_origin[1] + 5),
                        (label_origin[0] + label_width, label_origin[1] - label_height - 5),
                        self.palette[u], -1
                    )
                    cv2.putText(
                        frame, o, tuple(label_origin),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
                    )
                    label_origin[-1] -= label_height + 1
        return frame


