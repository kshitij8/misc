#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <stdio.h>

#define maX(a,b) a>b?a:b

using namespace std;
using namespace cv;

inline void sortCorners(std::vector<cv::Point2f>& corners, cv::Point2f center){
	std::vector<cv::Point2f> top, bot;

	for (int i = 0; i < corners.size(); i++){
		if (corners[i].y < center.y)
			top.push_back(corners[i]);
		else
			bot.push_back(corners[i]);
	}

	cv::Point2f tl = top[0].x > top[1].x ? top[1] : top[0];
	cv::Point2f tr = top[0].x > top[1].x ? top[0] : top[1];
	cv::Point2f bl = bot[0].x > bot[1].x ? bot[1] : bot[0];
	cv::Point2f br = bot[0].x > bot[1].x ? bot[0] : bot[1];

	corners.clear();
	corners.push_back(tl);
	corners.push_back(tr);
	corners.push_back(br);
	corners.push_back(bl);
}
Vector<float> RGB_score(Mat img_src){
	waitKey(0);
	int red = 0, green = 0, blue = 0, total = 0;
	int rmax = 0, gmax = 0, bmax = 0;
	int rmin = 0, gmin = 0, bmin = 0;
	for (int m = 0; m < img_src.rows; m++) {
		for (int n = 0; n < img_src.cols; n++){
			red += img_src.at<Vec3b>(m,n)[0];
			green += img_src.at<Vec3b>(m,n)[1];
			blue += img_src.at<Vec3b>(m,n)[2];
			if (rmax < img_src.at<Vec3b>(m,n)[0])
				rmax = img_src.at<Vec3b>(m,n)[0];
			if (gmax < img_src.at<Vec3b>(m,n)[1])
				gmax = img_src.at<Vec3b>(m,n)[1];
			if (bmax < img_src.at<Vec3b>(m,n)[2])
				bmax = img_src.at<Vec3b>(m,n)[2];
			if (rmin > img_src.at<Vec3b>(m,n)[0])
				rmin = img_src.at<Vec3b>(m,n)[0];
			if (gmin > img_src.at<Vec3b>(m,n)[1])
				gmin = img_src.at<Vec3b>(m,n)[1];
			if (bmin > img_src.at<Vec3b>(m,n)[2])
				bmin = img_src.at<Vec3b>(m,n)[2];
		}
	}
	total = img_src.cols*img_src.rows;
	Vector<float> res = Vector<float>(3);
	if (rmax == rmin)
		res[0] = rmax;
	else res[0] = (rmax - ((red*1.0) / total)) / (rmax - rmin);
	if (gmax == gmin)
		res[1] = gmax;
	else res[1] = (gmax - ((green*1.0) / total)) / (gmax - gmin);
	if (bmax == bmin)
		res[2] = bmax;
	else res[2] = (bmax - ((blue*1.0) / total)) / (bmax - bmin);
	//cout << "::" << res[0] << "::" << res[1] << "::" << res[2];
	return res;
}

Mat crop_and_perspect(Mat img_src){
	//return img_src;
	//imshow("grab" + std::to_string(rand()), img_src);
	int height = img_src.rows;
	int width = img_src.cols;
	resize(img_src, img_src, Size(height/2,width/2));
	Mat img = img_src.clone();
	Mat mask = Mat::ones(img.size(), CV_8U) * GC_BGD;
	float border = 0.05;
	for (int m = (int)img.rows*border; m<(int)img.rows*(1 - border); m++) {
		for (int n = 0; n<img.cols; n++){
			mask.at<uchar>(m, n, 0) = GC_PR_FGD;
		}
	}
	float fgd_center = 0.25;
	for (int m = (int)img.rows*fgd_center; m<(int)img.rows*(1 - fgd_center); m++) {
		for (int n = 0; n<img.cols; n++){
			mask.at<uchar>(m, n, 0) = GC_FGD;
		}
	}
	Mat bgdModel, fgdModel;
	Rect rectangle(0, 0, img.cols, img.rows);
	grabCut(img, mask, rectangle, bgdModel, fgdModel, 10, GC_INIT_WITH_MASK);
	mask = (mask == GC_PR_FGD ) | ( mask == GC_FGD);
	Mat foreground(img.size(), CV_8UC3, Scalar(0, 0, 0));
	img.copyTo(foreground, mask);
	//imshow("grab" + std::to_string(rand()), foreground);
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(mask, contours, hierarchy, RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point2f(0, 0));
	int perimeter = 0, contourIndex = -1;
	for (int i = 0; i< contours.size(); i++){
		int newPerimeter = arcLength(contours[i], true);
		if (newPerimeter>perimeter){
			perimeter = newPerimeter;
			contourIndex = i;
		}
	}
	approxPolyDP(contours[contourIndex], contours[contourIndex], 0.1*arcLength(contours[contourIndex], true), true);
	std::vector<Point2f> approx;
	Mat(contours[contourIndex]).copyTo(approx);

	
	/*
	Size winSize = Size(10, 10);
	Size zeroZone = Size(-1, -1);
	TermCriteria criteria = TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 100, 0.001);
	Mat imgGray;
	cvtColor(img, imgGray, CV_BGR2GRAY);
	cornerSubPix(imgGray, approx, winSize, zeroZone, criteria);
	*/

	Mat drawing = Mat::zeros(foreground.size(), CV_8UC3);
	drawContours(drawing, contours, contourIndex, Scalar(255, 0, 0), 2, 8, hierarchy, 0, Point());
	//imshow("grab" + std::to_string(rand()), drawing);

	if (approx.size() != 4){
		//printf("%d\n", approx.size());
	}
	Point2f center(0, 0);
	for (int i = 0; i < approx.size(); i++)
		center += approx[i];

	center *= (1. / approx.size());
	sortCorners(approx, center);
	Mat final = Mat::zeros(height, width, CV_8UC3);

	std::vector<cv::Point2f> final_pts;
	final_pts.push_back(cv::Point2f(0, 0));
	final_pts.push_back(cv::Point2f(final.cols, 0));
	final_pts.push_back(cv::Point2f(final.cols, final.rows));
	final_pts.push_back(cv::Point2f(0, final.rows));

	Mat transmtx = getPerspectiveTransform(approx, final_pts);
	warpPerspective(img, final, transmtx, final.size());
	//waitKey(0);
	return final;
}

Mat custom_equalize(Mat final){
	fastNlMeansDenoising(final, final, 5, 5, 15);
	//GaussianBlur(final, final, Size(1,1), 1);
	//cvtColor(final, final, CV_BGR2GRAY);
	//equalizeHist(final, final);
	
	return final;
}

int get_perimeter(Mat final){
	cvtColor(final, final, CV_BGR2GRAY);
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	Canny(final, final, 50, 200, 3);
	findContours(final, contours, hierarchy, RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point2f(0, 0));
	vector<Vec4i> lines;
	HoughLinesP(final, lines, 1, CV_PI / 180, 80, 30, 10);
	int perimeter = 0;
	Mat drawing = Mat::zeros(final.size(), CV_8UC3);
	for (size_t i = 0; i < lines.size(); i++){
		line(drawing, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(0, 0, 255), 3, 8);
		perimeter += (int) sqrt(pow(lines[i][0] - lines[i][2], 2) + pow(lines[i][1] - lines[i][3], 2));
	}
	//imshow("f" + std::to_string(perimeter), drawing);
	return perimeter;
}
int get_corners(Mat src) {
	cvtColor(src, src, CV_BGR2GRAY);
	std::vector<cv::Point2f> corners;
	int maxCorners = 1000;
	corners.reserve(maxCorners);
	Mat img;
	img = src.clone();
	goodFeaturesToTrack(img, corners, maxCorners, 0.01, 20);

	/*int r = 4;
	for (int i = 0; i < corners.size(); i++)
	{
		circle(img, corners[i], r, Scalar(255,0,0), -1, 8, 0);
	}*/

	//imshow("Corner" + std::to_string(rand()), img);
	//imshow("Corner" + std::to_string(rand()), src);
	//cout << corners.size();
	//waitKey(0);
	return corners.size();
}
Vector<float> get_data_points(string inbound, string outbound){
	//cout << inbound<<endl;
	//cout << outbound << endl;
	Vector<float> data_points = Vector<float>(5);
	Mat src_base, src_test1;

	src_base = custom_equalize(crop_and_perspect(imread(inbound, CV_LOAD_IMAGE_COLOR)));
	//imshow("Corner" + std::to_string(rand()), src_base);
	src_test1 = custom_equalize(crop_and_perspect(imread(outbound, CV_LOAD_IMAGE_COLOR)));
	/*Vector<float> data1 = RGB_score(src_base);
	Vector<float> data2 = RGB_score(src_test1);
	data_points[0] = data1[0] - data2[0];
	data_points[1] = data1[1] - data2[1];
	data_points[2] = data1[2] - data2[2];
	*///imshow("d", src_base);
	//imshow("e", src_test1);

	int imgCount = 1;
	int dims = 3;
	const int sizes[] = { 256, 256, 256 };
	const int channels[] = { 0, 1, 2 };
	float rRange[] = { 0, 256 };
	float gRange[] = { 0, 256 };
	float bRange[] = { 0, 256 };
	const float *ranges[] = { rRange, gRange, bRange };
	Mat mask = Mat();
	Mat s_hist, t_hist;

	calcHist(&src_base, imgCount, channels, mask, s_hist, dims, sizes, ranges);
	calcHist(&src_test1, imgCount, channels, mask, t_hist, dims, sizes, ranges);
	//normalize(s_hist, s_hist, 0, src_base.rows, NORM_MINMAX, -1, Mat());
	//normalize(t_hist, t_hist, 0, src_test1.rows, NORM_MINMAX, -1, Mat());

	/// Apply the histogram comparison methods
	for (int i = 3; i < 4; i++)
	{
		int compare_method = i;
		double base_base = compareHist(s_hist, s_hist, compare_method);
		double base_test1 = compareHist(s_hist, t_hist, compare_method);
		if (i == 0)		{data_points[i] = (base_test1 + 1.0) / 2;
		}
		else if (i == 1) {
			data_points[i] = base_test1;
		}
		else if (i == 2) {
			data_points[i] = base_test1/base_base;
		}
		else if (i == 3) data_points[i] = base_test1;
		//printf(" Method [%d] Perfect, Test : %f, %f \n", i, base_base, base_test1);
	}
	int corners1, corners2;
	corners1 = get_corners(src_base);
	corners2 = get_corners(src_test1);
	
	if (corners2 == 0 && corners1 == 0)
		data_points[4] = 0;
	else
		data_points[4] = (1.0*abs(corners1 - corners2)) / (maX(corners1, corners2));

	//printf("Corners : %d %d\n", corners1 , corners2);
	//printf("Perimeters : %d %d\n", get_perimeter(src_base), get_perimeter(src_test1));

	//printf("Done \n");
	//waitKey(0);
	return  data_points;

}
/**
* @function main
*/
int main2()
{
	string set = "2";
	int changed = 51;
	//int unchanged = 4;

	string base_dir = "C:\\Users\\guptaks\\distrib\\set2\\temp";
	
	Mat training_data = Mat(50, 5, CV_32FC1);
	Mat training_classifications = Mat(50, 1, CV_32FC1);

	Mat var_type = Mat(5 + 1, 1, CV_8U);
	var_type.setTo(Scalar(CV_VAR_NUMERICAL)); // all inputs are numerical

	// this is a classification problem (i.e. predict a discrete number of class
	// outputs) so reset the last (+1) output var_type element to CV_VAR_CATEGORICAL

	var_type.at<uchar>(5, 0) = CV_VAR_CATEGORICAL;

	double result; // value returned from a prediction

	// load training and testing data sets
	for (int i = 0; i < changed; i++)
	{

		// for each attribute
		String seq = std::to_string(i);
		String pad = std::string(4 - seq.size(), '0') + seq;
		cout << pad << " ";
		Vector<float> data = get_data_points(base_dir + "\\pair_" + pad + "_inbound.jpg", base_dir + "\\pair_" + pad + "_outbound.jpg");

		for (int attribute = 0; attribute < (5 + 1); attribute++)
		{
			
				training_data.at<float>(i, attribute) = data[attribute];

		}
		training_classifications.at<float>(i,0) = i%2;
	}

		// define the parameters for training the random forest (trees)

		float priors[] = { 1, 1};  // weights of each classification for classes
		// (all equal as equal samples of each digit)

		CvRTParams params = CvRTParams(25, // max depth
			5, // min sample count
			0, // regression accuracy: N/A here
			false, // compute surrogate split, no missing data
			15, // max number of categories (use sub-optimal algorithm for larger numbers)
			priors, // the array of priors
			false,  // calculate variable importance
			4,       // number of variables randomly selected at node and used to find the best split(s).
			100,	 // max number of trees in the forest
			0.01f,				// forrest accuracy
			CV_TERMCRIT_ITER | CV_TERMCRIT_EPS // termination cirteria
			);

		CvRTrees* rtree = new CvRTrees;

		rtree->train(training_data, CV_ROW_SAMPLE, training_classifications,
			Mat(), Mat(), var_type, Mat(), params);

		Mat test_data = Mat(1, 5, CV_32FC1);
		for (int i = 0; i < changed; ++i){
			for (int attribute = 0; attribute < (5 + 1); attribute++)
			{

				test_data.at<float>(0, attribute) = training_data.at<float>(i,attribute);

			}
			float response = rtree->predict(test_data);
			cout << response * 100 << "\n";
		}
		rtree->save("C:\\Users\\guptaks\\tree.xml");


}
int main(int argc, char **argv)
{
	CvRTrees* rtree = new CvRTrees;
	Mat training_data = Mat(1, 5, CV_32FC1);

	rtree->load("C:\\Users\\guptaks\\tree.xml");

	Vector<float> data = get_data_points(argv[1], argv[2]);

	for (int attribute = 0; attribute < (5 + 1); attribute++)
	{

		training_data.at<float>(0, attribute) = data[attribute];

	}

	float response = rtree->predict(training_data);
	cout << response * 100;

	return 0;

}
/*
int main(int argc, char **argv)
{
	CvSVM SVM;

	SVM.load("classifier.xml");

	float trainingData[5];
	Vector<float> data = get_data_points(argv[1], argv[2]);
	for (int j = 0; j < 5; ++j)
		trainingData[j] = data[j];
	Mat sampleMat = (Mat_<float>(1, 5) << trainingData[0], trainingData[1], trainingData[2], trainingData[3], trainingData[4]);
	float response=SVM.predict(sampleMat);
	cout << response*100;

	return 0;

}
*/

int ma2in2(int argc, char **argv)
{
	string set = "2";
	int changed = 51;
	//int unchanged = 4;

	string base_dir = "C:\\Users\\guptaks\\distrib\\set2\\temp";
	//string files[250] = { "pair_0001", "pair_0003", "pair_0005", "pair_0007", "pair_0009", "pair_0011", "pair_0013", "pair_0015", "pair_0000", "pair_0002", "pair_0004", "pair_0006" };
	// Set up training data
	int labels[250];
	for (int i = 0; i < changed; ++i){
		labels[i] = i%2;
	}
	/*for (int i = changed; i < changed + unchanged; ++i){
		labels[i] = 1;
	}*/

	Mat labelsMat(changed, 1, CV_32SC1, labels);

	float trainingData[250][5];
	for (int i = 0; i < changed; ++i){
		String seq = std::to_string(i);
		String pad = std::string(4 - seq.size(), '0') + seq;
		cout << pad << " ";
		Vector<float> data = get_data_points(base_dir + "\\pair_" + pad + "_inbound.jpg", base_dir + "\\pair_" + pad + "_outbound.jpg");
		for (int j = 0; j < 5; ++j)
			trainingData[i][j] = data[j];
	}
	/*cout << "\ndone1\n";
	for (int i = changed; i < changed+unchanged; ++i){
		Vector<float> data = get_data_points(base_dir + set + "\\unchanged\\" + files[i] + "_inbound.jpg", base_dir + set + "\\unchanged\\" + files[i] + "_outbound.jpg");
		for (int j = 0; j < 5; ++j)
			trainingData[i][j] = data[j];
	}
	*/
	Mat trainingDataMat(changed, 5, CV_32FC1, trainingData);

	// Set up SVM's parameters
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::RBF;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

	// Train the SVM
	CvSVM SVM;
	SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);

	for (int i = 0; i < changed; ++i){
		String seq = std::to_string(i);
		String pad = std::string(4 - seq.size(), '0') + seq;
		cout << pad << " : ";
		Vector<float> data = get_data_points(base_dir + "\\pair_" + pad + "_inbound.jpg", base_dir + "\\pair_" + pad + "_outbound.jpg");
		for (int j = 0; j < 5; ++j)
			trainingData[i][j] = data[j];
		Mat sampleMat = (Mat_<float>(1, 5) << trainingData[i][0], trainingData[i][1], trainingData[i][2], trainingData[i][3], trainingData[i][4]);
		float response = SVM.predict(sampleMat);
		cout << response * 100 << "\n";
	}

	SVM.save("C:\\Users\\guptaks\\classifier.xml");

	return 0;

}
