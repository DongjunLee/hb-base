
import json
import os.path

from hbconfig import Config
import requests



def send_message_to_slack(config_name):
    project_name = os.path.basename(os.path.abspath("."))

    data = {
        "text": f"The learning is finished with {project_name} Project using {config_name} config."
    }

    requests.post(Config.slack.webhook_url, data=json.dumps(data))
