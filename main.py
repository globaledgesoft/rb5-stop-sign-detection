import rclpy
import argparse

from utils.sign_detector import SignDetector

def main(model_path):
    rclpy.init(args=None)
    detector = SignDetector(model_path)
    rclpy.spin(detector)
    detector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', '-p', required=True,
                        help="path to tflite model", type=str)
    args = parser.parse_args()
    main(args.model_path)