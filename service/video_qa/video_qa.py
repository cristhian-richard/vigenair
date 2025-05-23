
import logging

import config as ConfigService
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import utils as Utils


class VideoQA:

  def __init__(self, gcs_bucket_name: str, render_file: Utils.TriggerFile):
    """Initialiser.

    Args:
      gcs_bucket_name: The GCS bucket to read from and store files in.
      render_file: Path to the input rendering file, which is in a
        `<timestamp>-combos` subdirectory of the root video folder (see
        `extractor.Extractor` for more information on root folder naming).
    """
    self.gcs_bucket_name = gcs_bucket_name
    self.render_file = render_file

    vertexai.init(
        project=ConfigService.GCP_PROJECT_ID,
        location=ConfigService.GCP_LOCATION,
    )
    self.text_model = GenerativeModel(ConfigService.CONFIG_TEXT_MODEL)
    self.vision_model = GenerativeModel(ConfigService.CONFIG_VISION_MODEL)


  def get_video_feedback(self):
    """Sends all video files in the GCS directory and a prompt to Gemini for QA feedback.

        Returns:
            A dictionary with video file names as keys and their QA feedback as values,
            or an empty dictionary if feedback could not be retrieved.
        """
        try:
            feedback_dict = {}
            gcs_folder_path = f'gs://{self.gcs_bucket_name}/{self.render_file.gcs_folder_path}'
            video_files = Utils.list_gcs_files(gcs_folder_path, file_extension='.mp4')

            for video_file in video_files:
                gcs_video_path = f'{gcs_folder_path}/{video_file}'
                response = GenerativeModel.generate_content(
                    [
                        Part.from_uri(gcs_video_path, mime_type='video/mp4'),
                        ConfigService.VIDEO_QA_PROMPT,
                    ],
                    safety_settings=ConfigService.CONFIG_DEFAULT_SAFETY_CONFIG,
                )
                if (
                    response.candidates and response.candidates[0].content.parts
                    and response.candidates[0].content.parts[0].text
                ):
                    feedback = response.candidates[0].content.parts[0].text.strip()
                    feedback_dict[video_file] = feedback
                    logging.info("QA Feedback received for %s: %s", video_file, feedback)
                else:
                    logging.warning("No QA feedback received for the video: %s", video_file)
            return feedback_dict
        except Exception:
            logging.exception("Error occurred while getting QA feedback from Gemini.")
        return {}
            logging.info("QA Feedback received: %s", feedback)
            return feedback
        else:
            logging.warning("No QA feedback received for the video.")
    except Exception:
        logging.exception("Error occurred while getting QA feedback from Gemini.")
    return None
