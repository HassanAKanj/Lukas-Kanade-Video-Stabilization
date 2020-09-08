#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <Loaded.h>
#include <Outputtable.h>

using namespace std;

class Stabilized : public Video
{
private:
	//in and out objects
	Loaded src;

	//helper methods
	float k;
	cv::Mat warpImage(cv::Mat image, cv::Mat affine);
	float computeK(cv::Point2f coords, int point);
	void iterate(cv::Mat grayFrame, cv::Mat warp, cv::Mat firstGray, vector<vector<cv::Mat>> descentTranspose, cv::Mat invHessian, cv::Mat warped, cv::Mat diff);
	void updateWarp(cv::Mat del_p, cv::Mat warp);
	cv::Mat pseudoInverse(cv::Mat src);

	//settings
	int frameLimit;
	int numIterations;
public:

	Outputtable dst;

	//constructor, copy constructor
	Stabilized();
	Stabilized(Loaded src);

	//methods
	void start();

	//getters and setters
	void setIterations(int num);
	float getK();
	void setFrameLimit(int lim);
};

