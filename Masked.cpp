#include "Masked.h"
#include <Eigen/Core>

Masked::Masked(Cropped src):src(src)
{
	mean = src.getMean();
	rows = src.getRows();
	columns = src.getColumns();
	dst = src;
}

vector<cv::Mat> Masked::getMasks()
{
	return masks;
}

void Masked::computeMasks()
{
	cv::Mat frame = src.dst.nextFrame();
	
	cv::Mat ssd = cv::Mat::zeros(rows, columns, CV_32FC1);

	Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eig_ssd(ssd.ptr<float>(), rows, columns);

	cv::Mat diff(rows,columns,CV_32FC1);

	Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> eig_diff(diff.ptr<float>(), rows, columns);

	while (!frame.empty()) {

		cvtColor(frame, frame, cv::COLOR_BGR2GRAY);

		diff = frame - mean;

		eig_ssd.array() = eig_ssd.array() + eig_diff.array().square();

		frame = src.dst.nextFrame();
	}

	float m = eig_ssd.mean();

	float thresh = m + 0.5*sqrt((eig_ssd.array()-m).square().sum()/(rows*columns-1));

	threshold(ssd, ssd, thresh, 255, cv::THRESH_BINARY);

	ssd.convertTo(ssd, CV_8U);

	vector<vector<cv::Point>> contours;
	findContours(ssd, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

	vector<vector<cv::Point>> hull(contours.size(), vector<cv::Point>(contours[0].size()));

	for (int i = 0; i < contours.size(); i++) {
		if (contourArea(contours[i]) >= 0.001*rows*columns) {
			convexHull(contours[i], hull[i]);

			cv::Mat mask = cv::Mat::zeros(rows, columns, CV_32FC1);

			drawContours(mask, hull, i, cv::Scalar(255, 255, 255), cv::FILLED);

			mask.convertTo(mask, CV_8U);
			masks.push_back(mask);
		}
	}
}

void Masked::applyMask(int m, int f) {

	cv::Mat finMask;
	GaussianBlur(masks[m], finMask, cv::Size(33, 33), 0);
	cvtColor(finMask, finMask, cv::COLOR_GRAY2BGR);
	finMask.convertTo(finMask, CV_32FC3);
	finMask = finMask / 255;

	cv::Mat back = src.dst.getFrameAt(f);
	back.convertTo(back, CV_32F);

	back = (cv::Mat(rows, columns, CV_32FC3, cv::Scalar(1, 1, 1)) - finMask).mul(back);

	src.dst.reset();

	cv::Mat frame = src.dst.nextFrame();
	
	while (!frame.empty()) {

		dst.output(back + finMask.mul(frame));

		frame = src.dst.nextFrame();
	}
}
