#include "Video.h"
#include <vector>

using namespace std;

Video::Video() {
	currFrame = 0;
	columns = 0;
	rows = 0;
	frames = 0;
	fps = 0;
}

int Video::getRows() {
	return rows;
}

int Video::getColumns() {
	return columns;
}

int Video::getFrames() {
	return frames;
}

cv::Mat Video::nextFrame() {
	return video.at(currFrame++);
}

vector<cv::Mat> Video::getVideo() {
	return video;
}

void Video::reset() {
	currFrame = 0;
}

int Video::getCurrFrame() {
	return currFrame;
}

int Video::getFPS() {
	return fps;
}