from typing import Any, Callable, Optional

import warnings
from functools import wraps

from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


def deprecated(alternative_usage: Optional[str] = None) -> Callable:
    """Deprecation decorator with message."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            deprecation_warning_message = (
                f"\n{func.__name__} will be deprecated. "
                "It will be removed in the next update.\n"
            )
            if alternative_usage:
                deprecation_warning_message += (
                    f"Alternatively, use {alternative_usage} instead.\n"
                    "Please refer to the documentation for more details.\n"
                )
            deprecation_warning_message += "\n"
            logger.warning("DeprecationWarning:" + deprecation_warning_message)
            warnings.warn(
                deprecation_warning_message,
                category=DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator


# Enable DeprecationWarning
warnings.simplefilter("always", DeprecationWarning)
