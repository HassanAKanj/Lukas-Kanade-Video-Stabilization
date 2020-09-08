#pragma once
#include "Video.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

class Loaded : public Video
{
private:
	cv::VideoCapture vc;
	bool loaded;
	cv::Mat frame;
public:
	Loaded();
	Loaded(string video);
	void close();
	cv::Mat nextFrame();
	cv::Mat nextFrameLoad();
};

