#pragma once
#include "Video.h"
#include "Cropped.h"
#include "Outputtable.h"

class Masked : public Video
{
private:
	cv::Mat mean;
	Cropped src;
	vector<cv::Mat> masks;
public:
	Masked(Cropped src);
	Outputtable dst;
	vector<cv::Mat> getMasks();
	void computeMasks();
	void applyMask(int m, int f);
};

