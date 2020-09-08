#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>

using namespace std;

class Video
{
protected:
	int rows;
	int columns;
	int frames;
	vector<cv::Mat> video;
	int currFrame;
	int fps;

public:
	Video();
	cv::Mat nextFrame();
	int getRows();
	int getColumns();
	int getFrames();
	int getFPS();
	vector<cv::Mat> getVideo();
	void reset();
	int getCurrFrame();
};

