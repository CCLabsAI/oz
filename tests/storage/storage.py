from google.cloud import storage
from oauth2client.service_account import ServiceAccountCredentials
import os
import time


class CCLabsStorage:
    def save_string_bunket(str):
        timestr = time.strftime("%d-%m-%Y_%H-%M-%S")
        Client = storage.Client()
        client = Client.from_service_account_json('cclabs.gcp.json')
        bucket = client.get_bucket('oz-results')
        blob = bucket.blob('test-result-{}.txt'.format(timestr))
        blob.upload_from_string(str)
        


