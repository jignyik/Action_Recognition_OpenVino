import datetime
from openvino.inference_engine import IECore
from pathlib import Path
from landmarks_detector import LandmarksDetector
from face_detector import FaceDetector
from faces_database import FacesDatabase
from face_identifier import FaceIdentifier
from emotion_detection import EmotionDetector
import cv2
from utils import crop
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import MetaData
from sqlalchemy import Column,Integer,String,TIMESTAMP
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime


class Arguments:
    def __init__(self):
        self.fg = r"gallery"
        self.allow_grow = False
        self.m_fd = Path(r"models\face-detection-retail-0004\FP16-INT8\face-detection-retail-0004.xml")
        self.m_lm = Path(r"models\landmarks-regression-retail-0009\FP16-INT8\landmarks-regression-retail-0009.xml")
        self.m_reid = Path(r"models\face-reidentification-retail-0095\FP16-INT8\face-reidentification-retail-0095.xml")
        self.d_fd = "CPU"
        self.d_lm = "CPU"
        self.d_reid = "CPU"
        self.t_fd = 0.8
        self.t_id = 0.3
        self.exp_r_fd = 1.15

class FrameProcessor:
    QUEUE_SIZE = 16
    def get_config(self, device):
        config = {
            "PERF_COUNT": "YES" if self.perf_count else "NO",
        }
        return config

    def __init__(self):
        args = Arguments()
        self.perf_count = False
        self.allow_grow = args.allow_grow
        ie = IECore()
        self.face_detector = FaceDetector(ie, args.m_fd, (0, 0), confidence_threshold=args.t_fd, roi_scale_factor=args.exp_r_fd)
        self.landmarks_detector = LandmarksDetector(ie, args.m_lm)
        self.face_identifier = FaceIdentifier(ie, args.m_reid, match_threshold=args.t_id)
        self.face_detector.deploy(args.d_fd, self.get_config(args.d_fd))
        self.landmarks_detector.deploy(args.d_lm, self.get_config(args.d_lm), self.QUEUE_SIZE)
        self.face_identifier.deploy(args.d_reid, self.get_config(args.d_reid), self.QUEUE_SIZE)
        self.faces_database = FacesDatabase(args.fg, self.face_identifier, self.landmarks_detector, None, False)
        self.face_identifier.set_faces_database(self.faces_database)
        self.identity_and_box = []

        self.database = []
        self.timer = []
        self.output = None
        self.names = None
        self.rois = None
        self.box = None

    def process(self, frame, box):
        self.box = box
        self.names = None
        frame = frame[box[1]:box[3], box[0]:box[2]]
        orig_image = frame.copy()
        rois = self.face_detector.infer((frame,))
        if rois:
            self.rois = rois
            landmarks = self.landmarks_detector.infer((frame, rois))
            face_identities, unknowns = self.face_identifier.infer((frame, rois, landmarks))
            if self.allow_grow and len(unknowns) > 0:
                for i in unknowns:
                    # This check is preventing asking to save half-images in the boundary of images
                    if rois[i].position[0] == 0.0 or rois[i].position[1] == 0.0 or \
                            (rois[i].position[0] + rois[i].size[0] > orig_image.shape[1]) or \
                            (rois[i].position[1] + rois[i].size[1] > orig_image.shape[0]):
                        continue
                    crop_image = crop(orig_image, rois[i])
                    name = self.faces_database.ask_to_save(crop_image)
                    if name:
                        id = self.faces_database.dump_faces(crop_image, face_identities[i].descriptor, name)
                        face_identities[i].id = id

            self.names = [self.face_identifier.get_identity_label(face_identities[i].id) for i in range(len(face_identities))]
            return [self.face_identifier.get_identity_label(face_identities[i].id) for i in range(len(face_identities))]

    def draw_face(self, frame):
        if self.rois is not None and self.box is not None and self.names is not None:
            for o, i in enumerate(self.rois):
                xmin = self.box[0] + max(int(i.position[0]), 0)
                ymin = self.box[1] + max(int(i.position[1]), 0)
                xmax = self.box[0] + min(int(i.position[0] + i.size[0]), frame.shape[1])
                ymax = self.box[1] + min(int(i.position[1] + i.size[1]), frame.shape[0])
                frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 255), 1)
                frame = cv2.putText(frame, self.names[o], (xmin, ymax - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
        return frame