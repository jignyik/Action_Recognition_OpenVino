import os

os.environ['Path'] += 'Local\\openvino_2021\\deployment_tools\\ngraph\\lib;' \
                    'Local\\openvino_2021\\deployment_tools\\ngraph\\lib;' \
                    'Local\\openvino_2021\\deployment_tools\\inference_engine\\external\\tbb\\bin;' \
                    'Local\\openvino_2021\\deployment_tools\\inference_engine\\bin\\intel64\\Release;' \
                    'Local\\openvino_2021\\deployment_tools\\inference_engine\\bin\\intel64\\Debug;' \
                    'Local\\openvino_2021\\deployment_tools\\inference_engine\\external\\tbb\\bin;' \
                    'Local\\openvino_2021\\deployment_tools\\model_optimizer;'

from action_school import Arguments, FrameProcessor
import cv2

args = Arguments()

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cv2.ocl.setUseOpenCL(False)
frame_processor = FrameProcessor(args)
while True:
    a, frame = cap.read()
    frame = cv2.flip(frame,1)
    import datetime
    now = datetime.datetime.now()
    (frame_processor.process(frame))
    time = (datetime.datetime.now() - now).total_seconds()
    if time > .1 and frame_processor.action_recog.pred_labels is not None:
        print(frame_processor.action_recog.pred_labels)
        print("Inference Time: {} seconds".format(round(time,2)))

    cv2.imshow("", frame_processor.action_recog.draw_action_on_frame(frame_processor.action_boxes.draw_boxes(frame),
                                                                     frame_processor.action_boxes.boxes))

    key = cv2.waitKey(1)
    # Quit
    if key in {ord('q'), ord('Q'), 27}:
        break