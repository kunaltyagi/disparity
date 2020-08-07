#! /usr/bin/env python

import argparse
import cv2 as cv
import os
import numpy as np

alpha_slider_max = 100
title_window = "Linear Blend"


class DisparityExplorer:
    def __init__(self):
        self.min_disp = (
            -16  # usually 0, adjust for offset introduced by rectification algo
        )
        self.num_disp = self.min_disp * -2  # needs to be divisible by 16
        self.block_size = 7  # odd, between 3 and 11
        self.param_factor = (
            3 * self.block_size ** 2
        )  # larger factor means smoother disparity, 3 is num_channels
        self.p1 = 8 * self.param_factor
        self.p2 = 4 * self.p1  # needs to be more than p1
        self.max_disp_diff = 50  # non-positive value disables
        self.pre_filter_cap = 0  # clip the x derivate to this absolute value
        self.unique_ratio = 15  # bet 5 and 15
        self.speckle_window_size = 100  # 0 to disable, else between 50-200
        self.speckle_range = 2  # 1 or 2, automatically multiplied by 16
        self.mode = cv.StereoSGBM_MODE_HH

    def get_stereo_computer(self):
        return cv.StereoSGBM_create(
            self.min_disp,
            self.num_disp,
            self.block_size,
            self.p1,
            self.p2,
            self.max_disp_diff,
            self.pre_filter_cap,
            self.unique_ratio,
            self.speckle_window_size,
            self.speckle_range,
            self.mode,
        )


class UI:
    def __init__(self, location):
        self.dora = DisparityExplorer()
        self.computer = self.dora.get_stereo_computer()
        self.left_images = []
        self.right_images = []
        self.idx = 0

        cv.namedWindow("left")
        cv.namedWindow("right")
        cv.namedWindow("disparity")
        """
        cv.createTrackbar(
            "min_disp", "disparity", self.dora.min_disp, 20, self.min_disp,
        )
        """
        cv.createTrackbar(
            "num_disp", "disparity", (self.dora.num_disp // 16) - 1, 10, self.num_disp,
        )
        cv.createTrackbar(
            "block_size",
            "disparity",
            (self.dora.block_size - 1) // 2,
            6,
            self.block_size,
        )
        cv.createTrackbar(
            "p1", "disparity", (self.dora.p1 // self.dora.param_factor), 32, self.p1,
        )
        cv.createTrackbar(
            "p2", "disparity", (self.dora.p2 // self.dora.p1), 16, self.p2,
        )
        cv.createTrackbar(
            "unique_ratio", "disparity", self.dora.unique_ratio, 50, self.unique_ratio,
        )
        cv.createTrackbar(
            "speckle_window_size",
            "disparity",
            self.dora.speckle_window_size,
            300,
            self.speckle_window_size,
        )
        cv.createTrackbar(
            "speckle_range",
            "disparity",
            self.dora.speckle_range,
            3,
            self.speckle_range,
        )
        cv.createTrackbar(
            "mode", "disparity", self.dora.mode, 3, self.mode,
        )
        """
        self.pre_filter_cap = 0  # clip the x derivate to this absolute value
        self.speckle_window_size = 50  # 0 to disable, else between 50-200
        self.speckle_range = 2  # 1 or 2, automatically multiplied by 16
        self.mode = cv.StereoSGBM_MODE_HH
        """

        self._collate_images(location)

    def _collate_images(self, location):
        print("Looking for database in '{}'".format(location))
        contents = os.listdir(location)
        if not contents:
            print(f"No content found in '{location}'")
            return
        if len(contents) == 2 and all(
            [os.path.isdir(os.path.join(location, x)) for x in contents]
        ):
            print("Choosing based on ordering. Eg: 'left' < 'right', '0' < '1'")
            contents = sorted(contents)
            left_dir = os.path.join(location, contents[0])
            right_dir = os.path.join(location, contents[1])
        else:
            left_dir = right_dir = location

        if left_dir != right_dir:
            print("Assuming bijection between files in the directories")
            self.left_images = [
                os.path.join(left_dir, x) for x in sorted(os.listdir(left_dir))
            ]
            self.right_images = [
                os.path.join(right_dir, x) for x in sorted(os.listdir(right_dir))
            ]
        else:
            print("TODO")
            return
        cv.createTrackbar(
            "idx", "disparity", self.idx, len(self.left_images) - 1, self.index
        )

    def min_disp(self, val):
        self.dora.min_disp = val
        self.computer = self.dora.get_stereo_computer()
        self.show()

    def num_disp(self, val):
        self.dora.num_disp = (val + 1) * 16
        self.computer = self.dora.get_stereo_computer()
        self.show()

    def block_size(self, val):
        self.dora.block_size = (2 * val) + 1
        self.computer = self.dora.get_stereo_computer()
        self.show()

    def p1(self, val):
        self.dora.p1 = self.dora.param_factor * val
        self.computer = self.dora.get_stereo_computer()
        self.show()

    def p2(self, val):
        self.dora.p2 = self.dora.p1 * val
        self.computer = self.dora.get_stereo_computer()
        self.show()

    def max_disp_diff(self, val):
        self.dora.max_disp_diff = val
        self.computer = self.dora.get_stereo_computer()
        self.show()

    def unique_ratio(self, val):
        self.dora.unique_ratio = val
        self.computer = self.dora.get_stereo_computer()
        self.show()

    def speckle_window_size(self, val):
        self.dora.speckle_window_size = val
        self.computer = self.dora.get_stereo_computer()
        self.show()

    def speckle_range(self, val):
        self.dora.speckle_range = val
        self.computer = self.dora.get_stereo_computer()
        self.show()

    def mode(self, val):
        self.dora.mode = val
        self.computer = self.dora.get_stereo_computer()
        self.show()

    def index(self, val):
        self.idx = val
        self.show()

    def show(self):
        img_left = cv.imread(self.left_images[self.idx])
        img_right = cv.imread(self.right_images[self.idx])
        disparity = self.computer.compute(img_left, img_right).astype(np.float32)
        disparity = disparity / np.max(disparity)

        cv.imshow("left", img_left)
        cv.imshow("right", img_right)
        cv.imshow("disparity", disparity)


def parse_known():
    parser = argparse.ArgumentParser(
        description="Explore stereo disparity config space"
    )
    parser.add_argument(
        "--db",
        help="Path to the database. Images are found via pattern matching",
        default=".",
    )
    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_known()
    args.db = os.path.abspath(args.db)

    ui = UI(args.db)
    ui.show()
    cv.waitKey(0)


if __name__ == "__main__":
    main()
