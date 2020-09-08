#include "Outputtable.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

void Outputtable::setMode(int m, std::string dir) {
	mode = m;
	directory = dir;
}

void Outputtable::setDirectory(std::string dir)
{
	directory = dir;
}

Outputtable::Outputtable() {
	frames = 0;
	rows = 0;
	columns = 0;
}

Outputtable::Outputtable(Video src) {
	frames = src.getFrames();
	rows = src.getRows();
	columns = src.getColumns();
	writerInitialized = false;
	readerInitialized = false;
}

void Outputtable::release() {
	if (writerInitialized) {
		out.release();
	}
}

string Outputtable::getDirectory() {
	return directory;
}

void Outputtable::output(cv::Mat frame) {
	if (!mode) {
		if (!writerInitialized) {

			out = cv::VideoWriter(directory, cv::VideoWriter::fourcc('M', 'P', '4', 'V'),
				frames,
				cv::Size(columns, rows));

			writerInitialized = true;

		}

		frame.convertTo(frame, CV_8U);

		out.write(frame);
	}
	else {
		video.push_back(frame);
	}
}

cv::Mat Outputtable::nextFrame() {
	if (!mode) {

		if (!readerInitialized) {
			initializeRead();
		}

		cv::Mat frame;

		in >> frame;

		frame.convertTo(frame, CV_32F);

		return frame;
	}
	else {
		return Video::nextFrame();
	}
}

void Outputtable::reset() {
	if (readerInitialized){
		in.release();
		in = cv::VideoCapture(directory);
	}
}

cv::Mat Outputtable::getFrameAt(int f) {

	cv::Mat frame;

	if (!readerInitialized) {
		initializeRead();
	}

	in.set(cv::CAP_PROP_POS_FRAMES, f);

	in >> frame;

	return frame;
}

void Outputtable::initializeRead() {
	in = cv::VideoCapture(directory);
	readerInitialized = true;
}