import os
import requests
import traceback

class Slack():
    def __init__(self) -> None:
        self.slack_url = os.getenv('SLACK_URL')
        if not self.slack_url:
            raise ValueError('SLACK_URL environment variable not found.')
        
        self.slack_channel = os.getenv('SLACK_CHANNEL')
        if not self.slack_channel:
            raise ValueError('SLACK_CHANNEL environment variable not found.')
        

    def send_message(self, message):
        payload = {
            "channel": self.slack_channel,
            "text": message
        }
        requests.post(self.slack_url, json=payload)

    def send_exception(self, message):
        payload = {
            "channel": self.slack_channel,
            "text": message,
            "attachments": [
                {
                    "color": "#2eb886",
                    "text": "`" + message + "`"
                }
            ],
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "An exception was found"
                    }
                },
                {
                    "type": 'section',
                    "text": {
                        "type": 'mrkdwn',
                        "text": '```' + traceback.format_exc() + '```'
                    }
                },
            ]
        }
        requests.post(self.slack_url, json=payload)

    def send_results(self, message, results):
        payload = {
            'text': message,
            'attachments': [
                {
                    'fallback': 'Attachment fallback text',
                    'color': '#36a64f',
                    'pretext': 'Optional text before the attachment',
                    'author_name': 'Author Name',
                    'title': 'Attachment Title',
                    'text': 'Attachment Text',
                    'fields': [
                        {'title': 'Field 1', 'value': 'Value 1', 'short': True},
                        {'title': 'Field 2', 'value': 'Value 2', 'short': True}
                    ],
                    'footer': 'Attachment Footer',
                    'ts': 123456789
                }
            ]
        }

        requests.post(self.slack_url, json=payload)