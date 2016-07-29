#include "opencv2/core/core.hpp"

cv::Mat extract_gabor_filters(cv::Mat input);
cv::Mat convert_color_space(cv::Mat input);
cv::Mat down_sample(cv::Mat input_src, int max_size);

