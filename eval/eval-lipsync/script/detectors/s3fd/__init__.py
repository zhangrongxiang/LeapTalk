import logging
import time

import cv2
import numpy as np
import torch

from .box_utils import nms_
from .nets import S3FDNet

img_mean = np.array([104.0, 117.0, 123.0])[:, np.newaxis, np.newaxis].astype("float32")


class S3FD:
    def __init__(self, net: S3FDNet, device="cuda"):
        """
        We now accept an *already-initialized* S3FDNet as `net`,
        instead of loading weights here.
        """
        tstamp = time.time()
        self.device = device
        self.net = net.to(self.device)
        self.net.eval()
        logging.info(
            f"[S3FD] S3FDNet instance is ready (initialized in {time.time()-tstamp:.4f} sec)."
        )

    def detect_faces(self, image, conf_th=0.8, scales=[1]):
        """
        Same detection code as before, but we no longer load the model here.
        """
        self.net.to(self.device)
        self.net.eval()
        w, h = image.shape[1], image.shape[0]
        bboxes = np.empty(shape=(0, 5))

        with torch.no_grad():
            for s in scales:
                scaled_img = cv2.resize(
                    image, dsize=(0, 0), fx=s, fy=s, interpolation=cv2.INTER_LINEAR
                )
                scaled_img = np.swapaxes(scaled_img, 1, 2)
                scaled_img = np.swapaxes(scaled_img, 1, 0)
                scaled_img = scaled_img[[2, 1, 0], :, :]
                scaled_img = scaled_img.astype("float32")
                scaled_img -= img_mean
                scaled_img = scaled_img[[2, 1, 0], :, :]
                x = torch.from_numpy(scaled_img).unsqueeze(0).to(self.device)

                y = self.net(x)  # forward pass
                detections = y.data.to(self.device)
                scale_tensor = torch.Tensor([w, h, w, h]).to(self.device)

                for i in range(detections.size(1)):
                    j = 0
                    while detections[0, i, j, 0] > conf_th:
                        score = detections[0, i, j, 0].item()
                        pt = (detections[0, i, j, 1:] * scale_tensor).cpu().numpy()
                        bbox = (pt[0], pt[1], pt[2], pt[3], score)
                        bboxes = np.vstack((bboxes, bbox))
                        j += 1

            # NMS, etc. (unchanged)
            keep = nms_(bboxes, 0.1)
            bboxes = bboxes[keep]
        return bboxes
