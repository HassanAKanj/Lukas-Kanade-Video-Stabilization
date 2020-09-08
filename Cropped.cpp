#include "Cropped.h"
#include "Stabilized.h"
#include <opencv2/opencv.hpp>

Cropped::Cropped() { }

cv::Mat Cropped::getMean() { 
	return mean;
}

Cropped::Cropped(Stabilized src):src(src) {
	k = src.getK();
	columns = src.getColumns();
	rows = src.getRows();
	fps = src.getFPS();
	frames = src.getFrames();
	dst = src;
	mean = cv::Mat::zeros(rows, columns, CV_32FC1);
}

void Cropped::start() {

	cv::Mat frame = src.dst.nextFrame();
	cv::Mat addFrame;
	//use addFrame to keep frame in BGR

	float centerX = k * rows / 2;
	float centerY = k * columns / 2;

	while (!frame.empty()) {
		
		cv::resize(frame, frame, cv::Size(0,0), k, k, cv::INTER_LINEAR);

		cv::Rect ROI((int) centerY - columns/2, (int) centerX - rows / 2, columns, rows);
		frame = frame(ROI);

		dst.output(frame);

		cv::cvtColor(frame, addFrame, cv::COLOR_BGR2GRAY);

		cv::scaleAdd(addFrame, 1.0/frames, mean, mean);

		frame = src.dst.nextFrame();
	}

	dst.release();
}