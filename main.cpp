#include <iostream>
/* #include <filesystem> */
#include <memory>

#include <boost/filesystem.hpp>

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/ximgproc.hpp>

namespace fs = boost::filesystem;

auto get_lr_pair(const fs::path& base_dir, unsigned int start, unsigned int end) {
    struct LR_Pair {
        cv::Mat left, right;
        LR_Pair() = default;
        LR_Pair(cv::Mat a, cv::Mat b): left(a), right(b) {}
    };
    std::vector<LR_Pair> pair;

    assert(fs::exists(base_dir / "left"));
    assert(fs::exists(base_dir / "right"));

    unsigned int idx = static_cast<unsigned int>(-1);
    std::cout << "Reading files: \n";
    for (const auto& l_it: fs::directory_iterator(base_dir / "left")) {
        ++idx;
        if (idx < start) {
            continue;
        }
        if (idx >= end) {
            break;
        }
        assert(is_regular_file(l_it));
        const auto l_name = l_it.path();

        std::string r_name_part = l_name.stem().string();
        r_name_part = r_name_part.substr(0, r_name_part.size() - 4);
        r_name_part += "right";
        r_name_part += l_name.extension().c_str();
        const auto r_name = l_name.parent_path().parent_path() / "right" / r_name_part;
        std::cout << l_name << '\t' << r_name << '\n';
        pair.emplace_back(cv::imread(l_name.c_str(), cv::ImreadModes::IMREAD_COLOR), cv::imread(r_name.c_str(), cv::ImreadModes::IMREAD_COLOR));
    }
    std::cout << "\n Files read\nPress key to compute per-file";
    return pair;
}

auto get_lr_pair(const fs::path& base_dir, unsigned int count) {
    return get_lr_pair(base_dir, 0, count);
}

auto get_lr_pair(const fs::path& base_dir) {
    return get_lr_pair(base_dir, 0, static_cast<unsigned int>(-1));
}

struct SGBM_config {
    int num_disparities=2, block_size = 1, P1 = 8, P2 = 32, max_disp_value = 0, pre_filter_cap = 0;
    int unique_ratio = 0, speckle_window = 3, speckle_range = 0, mode = cv::StereoSGBM::MODE_SGBM;
    double lambda = 8000, sigma = 1.5, lrc_threshold = 24;
};

struct DisparityFiltered {
    std::shared_ptr<cv::StereoMatcher> left_matcher, right_matcher;
    std::shared_ptr<cv::ximgproc::DisparityWLSFilter> filter;
    DisparityFiltered(SGBM_config config) {
        config.num_disparities = config.num_disparities * 16; // multiple of 16
        config.block_size = 2 * config.block_size + 1; // always odd, >=1
        int P_factor = 3 * config.block_size * config.block_size; // 3 is number of channels

        left_matcher = cv::StereoSGBM::create(
            0, // min disparity
            config.num_disparities,
            config.block_size,
            config.P1 * P_factor,
            config.P2 * P_factor,
            config.max_disp_value,
            config.pre_filter_cap,
            config.unique_ratio,  // 5-15
            config.speckle_window,  // 50-200
            config.speckle_range,  // 1 or 2
            config.mode  // MODE_SGBM, MODE_HH, MODE_SGBM_3WAY, MODE_HH4
        );
        right_matcher = cv::ximgproc::createRightMatcher(left_matcher);

        filter = cv::ximgproc::createDisparityWLSFilter(left_matcher);
        filter->setLambda(config.lambda);  // 8000
        filter->setSigmaColor(config.sigma);  // 0.8 - 2.0
        filter->setLRCthresh(config.lrc_threshold);  // 24 is 1.5 pixels
        filter->setDepthDiscontinuityRadius(config.block_size + 1);
    }

    cv::Mat compute(cv::Mat left, cv::Mat right) {
        cv::Mat left_disp, right_disp, filtered_disp;

        left_matcher->compute(left, right, left_disp);
        left_matcher->compute(right, left, right_disp);

        filter->filter(left_disp, left, filtered_disp, right_disp, cv::Rect(), right);  // no idea about the ROI, let it be default
        return filtered_disp;
    }
};

int main() {
    const auto pairs = get_lr_pair("./Holopix50k/val", 5);

    DisparityFiltered filter({});
    for (auto& pair: pairs) {
        auto disparity = filter.compute(pair.left, pair.right);
        cv::imshow("left", pair.left);
        cv::imshow("disp", disparity);
        cv::waitKey();  // @TODO: make it possible to write the disparity image
    }
    return 0;
}
