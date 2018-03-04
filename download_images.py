import os
import shutil
import threading
import time

import numpy as np
import pandas as pd
import requests


class DownloadData(object):
    def __init__(self, input_csv, output_dir, num_threads=10):
        self.input_csv = input_csv
        self.output_dir = output_dir
        self.num_threads = num_threads

    def download_thread(self, urls, files):
        for (url, img) in zip(urls, files):
            img_path = os.path.join(self.output_dir, img + '.jpg')

            if os.path.exists(img_path):
                print("{} already exists, skipping".format(img_path))
                continue
            try:
                response = requests.get(url, stream=True)
                if response.status_code != 200:
                    print("Unable to access {}".format(url))
                    continue
            except Exception as e:
                continue

            with open(img_path, 'wb') as out_file:
                shutil.copyfileobj(response.raw, out_file)

    def download_data(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        df = pd.read_csv(self.input_csv)

        url_chunks = np.array_split(df["image_url"].values, self.num_threads)
        file_chunks = np.array_split(df["filename"].values, self.num_threads)

        threads = []
        for (url_chunk, file_chunk) in zip(url_chunks, file_chunks):
            thread = threading.Thread(target=self.download_thread, args=(url_chunk, file_chunk))
            thread.daemon = True
            threads.append(thread)
            thread.start()

        while threading.active_count() > 1:
            time.sleep(0.1)

        print("Total images downloaded: {}".format(len(os.listdir(self.output_dir))))
