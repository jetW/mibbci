import cv2
import numpy as np


OVERLAY_COLOR = (255, 255, 255)
CURSOR_RADIUS = 10


def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)

    while True:
		ret_val, img_raw = cam.read()

		if mirror:
			img_raw = cv2.flip(img_raw, 1)

		img_overlay = np.zeros(img_raw.shape, np.uint8)
		cv2.circle(img_overlay, (img_raw.shape[1]/2, img_raw.shape[0]/2), CURSOR_RADIUS, OVERLAY_COLOR, thickness=-1)
		cv2.rectangle(img_overlay, (100, 100), (120, 110), OVERLAY_COLOR, thickness=-1)
		cv2.rectangle(img_overlay, (400, 100), (420, 110), OVERLAY_COLOR, thickness=-1)
		cv2.putText(img_overlay, 'Sample text', (200, 200),  cv2.FONT_HERSHEY_PLAIN, 2, OVERLAY_COLOR, thickness=2)


		#img_final = 0.5 * img_overlay + img_raw
		img_final = np.zeros(img_raw.shape, np.uint8)
		cv2.addWeighted(img_raw, 1.0, img_overlay, 0.5, 0.0, img_final)

		# Display the image
		cv2.imshow('my webcam', img_final)

		if cv2.waitKey(1) == 27:
			break  # esc to quit

    cv2.destroyAllWindows()

def main():
	show_webcam(mirror=True)

if __name__ == '__main__':
	main()