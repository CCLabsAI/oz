from google.cloud import storage
from oauth2client.service_account import ServiceAccountCredentials
import os
import time
import sys

def save_string_bunket(configFile, resultFile, name):
    timestr = time.strftime("%d-%m-%Y_%H-%M-%S")
    Client = storage.Client()
    client = Client.from_service_account_json(configFile)
    bucket = client.get_bucket('oz-results')
    blob = bucket.blob('test-result-{}-{}.txt'.format(name, timestr))
    with open(resultFile, 'rb') as my_file:
        blob.upload_from_file(my_file)

i = 0
config_file=''
send_file=''
name=''
while i < len(sys.argv):
    if sys.argv[i] == "-c":
        try:
            config_file=sys.argv[i + 1]
        except IndexError:
            pass
    if sys.argv[i] == "-f":
        try:
            send_file=sys.argv[i + 1]
        except IndexError:
            pass
    if sys.argv[i] == "-n":
        try:
            name=sys.argv[i + 1]
        except IndexError:
            pass
    i += 1

if config_file != '' and send_file != ''  and name != '':
    print(config_file)
    print(send_file)
    print(name)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config_file

    save_string_bunket(config_file, send_file, name)
else:
    print ("-p or -f arguments were missed"  )
