#pragma once

#include <opencv2/opencv.hpp>
#include <Video.h>

using namespace std;

class Outputtable : public Video
{
protected:
	cv::VideoWriter out;
	cv::VideoCapture in;
	bool writerInitialized;
	bool readerInitialized;
	std::string directory;
	int mode;
	void initializeRead();

public:

	//constructor and copy constructor
	Outputtable();
	Outputtable(Video src);

	//methods
	void output(cv::Mat frame);
	void release();
	cv::Mat nextFrame();
	cv::Mat getFrameAt(int f);
	void reset();

	//setters and getters
	void setMode(int m, std::string dir = "");
	void setDirectory(std::string dir);
	std::string getDirectory();
};

