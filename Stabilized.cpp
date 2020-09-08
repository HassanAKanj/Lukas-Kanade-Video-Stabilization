#include "Stabilized.h"
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <Loaded.h>

//tested
float Stabilized::computeK(cv::Point2f coords, int point) {
	int limX[4] = { 0, 0, rows, rows };
	int limY[4] = { 0, columns, columns, 0 };

	float kX = (limX[point] - rows / 2.0) / (coords.y - rows / 2.0);
	float kY = (limY[point] - columns / 2.0) / (coords.x - columns / 2.0);

	return max(kX,kY);
}

//tested
cv::Mat Stabilized::warpImage(cv::Mat image, cv::Mat affine) {

	//cout << endl << affine << endl;

	cv::Mat result;
	warpAffine(image, result, affine, cv::Size(columns, rows));

	cv::Mat adjustedAffine = cv::Mat::zeros(3, 3, CV_32FC1);
	affine.copyTo(adjustedAffine.rowRange(0, 2));
	adjustedAffine.at<float>(2, 2) = 1;

	vector<cv::Point2f> input(4), output(4);
	input[0] = cv::Point2f(0,0);
	input[1] = cv::Point2f(columns-1,0);
	input[2] = cv::Point2f(columns-1,rows-1);
	input[3] = cv::Point2f(0,rows-1);

	perspectiveTransform(input, output, adjustedAffine);

	float maxK = computeK(output[0], 0);

	for (int i = 1; i < 4; i++) {
		maxK = max(maxK, computeK(output[i], i));
	}

	k = max(k, maxK);

	return result;
}

void Stabilized::updateWarp(cv::Mat del_p, cv::Mat warp) {

	float p1 = warp.at<float>(0,0) - 1;
	float p2 = warp.at<float>(1,0);
	float p3 = warp.at<float>(0,1);
	float p4 = warp.at<float>(1,1) - 1;
	float p5 = warp.at<float>(0,2);
	float p6 = warp.at<float>(1,2);

	warp.at<float>(0, 0) = 1 + p1 + del_p.at<float>(0,0) + p1 * del_p.at<float>(0,0) + p3 * del_p.at<float>(1,0);
	warp.at<float>(1, 0) = p2 + del_p.at<float>(1,0) + p2 * del_p.at<float>(0,0) + p4 * del_p.at<float>(1,0);
	warp.at<float>(0, 1) = p3 + del_p.at<float>(2,0) + p1 * del_p.at<float>(2,0) + p3 * del_p.at<float>(3,0);
	warp.at<float>(1, 1) = 1 + p4 + del_p.at<float>(3,0) + p2 * del_p.at<float>(2,0) + p4 * del_p.at<float>(3,0);
	warp.at<float>(0, 2) = p5 + del_p.at<float>(4,0) + p1 * del_p.at<float>(4,0) + p3 * del_p.at<float>(5,0);
	warp.at<float>(1, 2) = p6 + del_p.at<float>(5,0) + p2 * del_p.at<float>(4,0) + p4 * del_p.at<float>(5,0);
}

void Stabilized::iterate(cv::Mat grayFrame, cv::Mat warp, cv::Mat firstGray, vector<vector<cv::Mat>> descentTranspose, cv::Mat invHessian, cv::Mat warped, cv::Mat diff) {

	float val = 1;
	for (int i = 0; val > 0.00001 && i < numIterations; i++) {

		warped = warpImage(grayFrame, warp);
		diff = warped - firstGray;

		cv::Mat before_del_p = cv::Mat::zeros(6, 1, CV_32FC1);

		for (int x = 0; x < rows; x++)
		{
			const float* r = diff.ptr<float>(x);

			for (int y = 0; y < columns; y++) {
				scaleAdd(descentTranspose[x][y], r[y], before_del_p, before_del_p);
			}
		}

		cv::Mat del_p = invHessian * before_del_p;

		updateWarp(del_p, warp);
		
		val = norm(del_p);
	}
}

Stabilized::Stabilized(Loaded src):src(src) {
	frameLimit = -1;
	rows = src.getRows();
	columns = src.getColumns();
	frames = src.getFrames();
	numIterations = 20;
	k = 1;
	fps = src.getFPS();
	dst = src;
}

Stabilized::Stabilized() {
	frameLimit = -1;
	numIterations = 20;
	k = 1;
}

cv::Mat Stabilized::pseudoInverse(cv::Mat src) {

	cv::Mat S, U, Vt;
	SVDecomp(src, S, U, Vt);

	double max;
	minMaxIdx(S, 0, &max);

	double cutoff = pow(10, -15) * max;
	threshold(S, S, cutoff, 0, cv::THRESH_TOZERO);

	cv::Mat diagonalS = cv::Mat::zeros(6, 6, CV_32FC1);
	for (int i = 0; i < 6; i++) {
		diagonalS.at<float>(i, i) = 1 / S.at<float>(i);
	}

	cv::Mat pseudo = Vt.t() * diagonalS * U.t();

	return pseudo;
}

void Stabilized::start() {

	src.reset();

	cv::Mat warp = cv::Mat::zeros(2,3,CV_32FC1);
	warp.at<float>(0, 0) = 1;
	warp.at<float>(1, 1) = 1;

	cv::Mat firstFrame = src.nextFrame();

	cv::Mat firstGray;
	cvtColor(firstFrame, firstGray, cv::COLOR_BGR2GRAY);

	cv::Mat sobelX;
	Sobel(firstGray, sobelX, CV_32FC1, 1, 0);

	cv::Mat sobelY;
	Sobel(firstGray, sobelY, CV_32FC1, 0, 1);

	cv::Mat hessian = cv::Mat::zeros(6, 6, CV_32FC1);

	vector<vector<cv::Mat>> descentTranspose(rows, vector<cv::Mat>(columns));

	cv::Mat jacobian = cv::Mat::zeros(2, 6, CV_32FC1);

	float* rJ1 = jacobian.ptr<float>(0);
	float* rJ2 = jacobian.ptr<float>(1);

	rJ1[4] = 1;
	rJ2[5] = 1;

	cv::Mat gradient(1, 2, CV_32FC1);
	float* g = gradient.ptr<float>(0);

	cv::Mat transpose(6, 1, CV_32FC1);
	cv::Mat descent(1, 6, CV_32FC1);

	for (int x = 0; x < rows; x++) {

		const float* rX = sobelX.ptr<float>(x);
		const float* rY = sobelY.ptr<float>(x);

		for (int y = 0; y < columns; y++) {

			rJ1[0] = y;
			rJ1[2] = x;
			rJ2[1] = y;
			rJ2[3] = x;

			g[0] = rX[y];
			g[1] = rY[y];
			
			descent = gradient * jacobian;
			transpose = descent.t();
			hessian = hessian + transpose * descent;

			descentTranspose[x][y] = transpose.clone();
		}
	}

	cv::Mat invHessian = pseudoInverse(hessian);

	cv::Mat frame = firstFrame;
	//video.push_back(frame);
	dst.output(frame);
	cout << src.getCurrFrame() << "\t";

	cv::Mat gray;

	int f = 0;

	while (frameLimit == -1 || src.getCurrFrame() < frameLimit) {
		frame = src.nextFrame();

		f = src.getCurrFrame();
		cout << f << "\t";

		if (!frame.empty()) {

			cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

			cv::Mat ptr1(rows, columns, CV_32FC1);
			cv::Mat ptr2(rows, columns, CV_32FC1);
			iterate(gray, warp, firstGray, descentTranspose, invHessian, ptr1, ptr2);

			cv::Mat warped = warpImage(frame, warp);

			dst.output(warped);
		}
		else {

			dst.release();

			break;
		}
	}
}

void Stabilized::setIterations(int num)
{
	numIterations = num;
}

float Stabilized::getK(){
	return k;
}

void Stabilized::setFrameLimit(int lim)
{
	frameLimit = lim;
}
