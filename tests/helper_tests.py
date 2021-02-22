import unittest
import os

from logging.handlers import TimedRotatingFileHandler

from context import helper_module


class HelperTest(unittest.TestCase):
    def test_get_parser(self):
        """ Argument parser knows config argument """

        arguments = ["--config", "test.yaml"]
        parser = helper_module.get_parser()
        args = parser.parse_args(arguments)

        self.assertTrue(args.config == "test.yaml")

    def test_get_logger(self):
        """ Logger is correctly configured """

        logger_name = "testlogger"
        logger = helper_module.get_logger("./", name=logger_name)

        self.assertTrue(logger.name == logger_name)
        self.assertTrue(len(logger.handlers) == 2)
        self.assertTrue(type(logger.handlers[0]) is TimedRotatingFileHandler)

        for i in logger.handlers:
            i.flush()
            i.close()

        try:
            # file exists sometimes ;/
            os.remove("./testlogger")
        except OSError:
            pass


if __name__ == "__main__":
    unittest.main()
