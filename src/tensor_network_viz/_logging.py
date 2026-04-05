from __future__ import annotations

import logging

LOGGER_NAME: str = "tensor_network_viz"
package_logger = logging.getLogger(LOGGER_NAME)

if not any(isinstance(handler, logging.NullHandler) for handler in package_logger.handlers):
    package_logger.addHandler(logging.NullHandler())


def get_package_logger() -> logging.Logger:
    return package_logger


__all__ = ["LOGGER_NAME", "get_package_logger", "package_logger"]
