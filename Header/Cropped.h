#pragma once
#include "Stabilized.h"
#include "Video.h"

class Cropped : public Video
{
private:
	Stabilized src;

	cv::Mat mean;
	float k;

public:

	Outputtable dst;

	Cropped(Stabilized src);
	Cropped();
	void start();
	cv::Mat getMean();
};

