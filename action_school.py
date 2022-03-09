from openvino.inference_engine import IECore
from pathlib import Path
from action_init import ActionRecognition
from action_boxes import ActionBoxes


class Arguments:
    def __init__(self):
        self.boxes_model= Path(r"ssd_mobilenet_v1_fpn_coco\FP16\ssd_mobilenet_v1_fpn_coco.xml")
        self.action_model = Path(r"slowfast_3_boxes\SlowFast_32x2_R101_50_50.xml")
        self.d_fd = "CPU"
        self.perf_count = False
        self.cpu_lib = ""


class FrameProcessor:
    QUEUE_SIZE = 16

    def get_config(self, device):
        config = {
            "PERF_COUNT": "YES" if self.perf_count else "NO",
        }
        if device == 'GPU' and self.gpu_ext:
            config['CONFIG_FILE'] = self.gpu_ext
        return config

    def __init__(self, args):
        self.gpu_ext = ""
        self.perf_count = False
        ie = IECore()
        if args.cpu_lib and 'CPU' in {args.d_fd, args.d_lm, args.d_reid}:
            ie.add_extension(args.cpu_lib, 'CPU')
        self.action_boxes = ActionBoxes(ie, args.boxes_model)
        self.action_boxes.deploy(args.d_fd, self.get_config(args.d_fd))

        self.action_recog = ActionRecognition(ie, args.action_model)
        self.action_recog.deploy(args.d_fd, self.get_config(args.d_fd))

    def process(self, frame):
        boxes = self.action_boxes.box_for_action
        self.action_recog.saved_frames.append(frame)
        # Saved frame reaches half then inference box model
        if len(self.action_recog.saved_frames) == self.action_recog.frame_length:
            boxes = self.action_boxes.infer((frame,))

            # if there is no human present/detected, then the list of saved frame will clear and restart again
            if not boxes.any():
                print("No human detected")
                self.action_recog.saved_frames = []

        # Saved frame reaches sequence length, proceed to feed frames to action recognition model
        if len(self.action_recog.saved_frames) == self.action_recog.sqn_length:
            action_result = self.action_recog.infer((self.action_recog.saved_frames, boxes))
            self.action_recog.saved_frames = []
            return action_result





