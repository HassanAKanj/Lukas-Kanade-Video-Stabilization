#include "Loaded.h"

Loaded::Loaded(string directory) {
	vc = cv::VideoCapture(directory);
	columns = vc.get(cv::CAP_PROP_FRAME_WIDTH);
	rows = vc.get(cv::CAP_PROP_FRAME_HEIGHT);
	fps = vc.get(cv::CAP_PROP_FPS);
	frames = vc.get(cv::CAP_PROP_FRAME_COUNT);
	loaded = false;
}

Loaded::Loaded() {
	loaded = false;
}

void Loaded::close() {
	vc.release();
}

cv::Mat Loaded::nextFrameLoad() {

	if (!loaded) {
		cv::Mat frame;
		vc >> frame;

		if (frame.empty()) {
			close();
			currFrame = 0;
			loaded = true;
		}
		else {
			frame.convertTo(frame, CV_32FC1);
			video.push_back(frame);
			currFrame++;
		}
		return frame;
	}
	else {
		return Video::nextFrame();
	}
}

cv::Mat Loaded::nextFrame() {

	vc >> frame;

	if (frame.empty()) {
		close();
		currFrame = 0;
	}
	else {
		frame.convertTo(frame, CV_32FC1);
		currFrame++;
	}

	return frame;
}
