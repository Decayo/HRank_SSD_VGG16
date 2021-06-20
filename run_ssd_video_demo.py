from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.utils.misc import Timer
import cv2
import sys
import logging
# to do list :
# LOAD 只LOAD DICT ? ... 還是說 LOSS == LOSS
# 改成 SHOW FPS
# 改成 LOAD video path

# if len(sys.argv) < 4:
#     print('Usage: python run_ssd_example.py <net type>  <model path> <label path> [video file]')
#     sys.exit(0)
net_type = sys.argv[1]
model_path = sys.argv[2]
label_path = sys.argv[3]
cap = cv2.VideoCapture(sys.argv[4])
save_path = sys.argv[5]
logging.basicConfig(level=logging.INFO,filename='output_video_config',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.info(f"Now calculate the frames...{sys.argv[4]},savepath:{save_path}")
# if len(sys.argv) >= 5:
#     cap = cv2.VideoCapture(sys.argv[4])  # capture from file
# else:
#     cap = cv2.VideoCapture(0)   # capture from camera
#     cap.set(3, 1920)
#     cap.set(4, 1080)

class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)


if net_type == 'vgg16-ssd':
    #net = create_vgg_ssd(len(class_names), is_test=True)
    net = create_vgg_ssd(len(class_names), is_test=True, compress_rate=[0.0]*100)
else:
    print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    sys.exit(1)
net.load(model_path)


if net_type == 'vgg16-ssd':
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)
else:
    print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    sys.exit(1)


timer = Timer()
flag_first = True
avg_time = 0
total = 1
while True:
    #ret, orig_image = cap.read()
    ret, orig_image = cap.read()
    if orig_image is None:
        if not flag_first:
            print("video end")
            break
        continue
    else:
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        total +=1
        timer.start()
        boxes, labels, probs = predictor.predict(image, 10, 0.4)
        interval = timer.end()
        print('Time: {:.4f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
        avg_time += interval
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
            cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)

            cv2.putText(orig_image, label,
                        (box[0]+20, box[1]+40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,  # font scale
                        (255, 0, 255),
                        2)  # line type
        if(total == 300):
            avg_time = avg_time/total
            logging.info(f"300 frames avg_num:{avg_time}")
            total = 1
            avg_time = 0
        if flag_first:
            flag_first = False
            fps, w, h = 30, orig_image.shape[1], orig_image.shape[0]
            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        vid_writer.write(orig_image)
        cv2.imshow('annotated', orig_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("end")
        break
cap.release()
cv2.destroyAllWindows()
