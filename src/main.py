"""
Main entry point for the application
"""

import logging
import sys

from src.camera import CameraError, display_video_stream, load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def main():
    """
    Main function - starts camera debug display.

    Loads configuration and displays video stream from USB camera.
    """
    try:
        logger.info("Starting LighthouseForCycles - Camera Debug Mode")
        logger.info("Loading configuration...")

        config = load_config()
        camera_config = config.get("camera", {})
        logger.info(f"Camera configuration: index={camera_config.get('index', 0)}")

        logger.info("Starting video stream display...")
        display_video_stream(config=config)

        logger.info("Application finished successfully")

    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {str(e)}")
        logger.error("Please ensure config.json exists in the project root")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Configuration error: {str(e)}")
        logger.error("Please check config.json format and values")
        sys.exit(1)
    except CameraError as e:
        logger.error(f"Camera error: {str(e)}")
        logger.error("Please check:")
        logger.error("  - USB camera is connected")
        logger.error("  - Camera is not being used by another application")
        logger.error("  - Camera index in config.json is correct")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
