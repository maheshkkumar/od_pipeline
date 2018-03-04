import os
import shutil
import tarfile

import pandas as pd
import progressbar
import requests

from download_images import DownloadData
from tf_utils import XML_TO_CSV, GenerateTFRecords, GeneratePaths

bar = progressbar.ProgressBar()

IMAGENET_WORDS_URL = "http://image-net.org/archive/words.txt"
IMAGENET_TAR_URL = "http://image-net.org/api/download/imagenet.bbox.synset?wnid={}"
IMAGENET_IMAGE_URL = "http://www.image-net.org/api/text/imagenet.synset.geturls.getmapping?wnid={}"

class ObjectDetectionPipeline(object):
    def __init__(self, output_dir, synset_ids):
        self.output_dir = output_dir
        self.paths = self.generate_paths()
        self.synset_ids = synset_ids

    def generate_paths(self):
        g_paths = GeneratePaths(self.output_dir)
        return g_paths.generate()

    def download_tars(self):
        try:
            for id in bar(self.synset_ids):
                tar_url = IMAGENET_TAR_URL.format(id)
                download_subpart = requests.get(tar_url).text.encode('utf-8').split(";")[-1].split("=")[-1].strip().split('"')[0]
                tar_path = "wget {} -P {}".format(IMAGENET_IMAGE_URL.split('/api')[0] + download_subpart, self.paths['tar_files'])
                os.system(tar_path)
        except requests.HTTPError as e:
            print("HTTP Exception: {}".format(e.message))
        except requests.ConnectionError as e:
            print("Connection Error: {}".format(e.message))
        except requests.ConnectTimeout as e:
            print("ConnectTimeout: {}".format(e.message))
        except Exception as e:
            print("Exception: {}".format(e.message))

    def extract_tars(self):
        base_path = self.paths['extracted_tars']
        tar_dir = self.paths['tar_files']
        tar_files = [os.path.join(tar_dir, t) for t in os.listdir(tar_dir)]
        extracted_path = self.paths['extract_tars']

        if not os.path.exists(extracted_path):
            os.makedirs(extracted_path)

        try:
            for tf in bar(tar_files):
                tf_extracted_path = os.path.join(extracted_path, tf.split('/')[-1].split('.')[0])
                with tarfile.open(tf) as t_file:
                    t_file.extractall(path=tf_extracted_path)
        except Exception as e:
            print("Exception occurred while extracting tar files: {}".format(e))

        extracted_folders = [os.path.join(extracted_path, f) for f in os.listdir(extracted_path)]
        for f in extracted_folders:
            folder_name = f.split(os.path.sep)[-1]
            input_folder_path = os.path.join(f, 'Annotation', f.split(os.path.sep)[-1])
            output_folder_path = os.path.join(base_path, folder_name)
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)
            images = [os.path.join(input_folder_path, i) for i in os.listdir(input_folder_path)]
            for i in images:
                shutil.copy(i, output_folder_path)

        shutil.rmtree(extracted_path)

    def download_images(self):

        image_dir = self.paths['image_files']
        image_data_dir = self.paths['image_dataset']
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        try:
            for id in bar(self.synset_ids):
                syn_path = IMAGENET_IMAGE_URL.format(id)
                urls = requests.get(syn_path, timeout=10).text
                image_urls = map(lambda _: _.split(' '), map(lambda _: _.strip(), urls.encode('utf-8').split('\n')))
                data_dict = dict((i[0], i[1]) for i in image_urls if len(i) == 2)
                data_df = pd.DataFrame(data_dict.items(), columns=['filename', 'image_url'])
                data_df.to_csv(os.path.join(image_dir, id + '.csv'), index=False)

            image_dir_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
            for i in image_dir_files:
                dd = DownloadData(i, image_data_dir)
                dd.download_data()

        except requests.HTTPError as e:
            print("HTTP Exception: {}".format(e.message))
        except requests.ConnectionError as e:
            print("Connection Error: {}".format(e.message))
        except requests.ConnectTimeout as e:
            print("ConnectTimeout: {}".format(e.message))
        except Exception as e:
            print("Exception occurred while downloading image files: {}".format(e))

    def clean_dataset(self):
        image_folder = self.paths['image_dataset']
        images = [os.path.join(image_folder, i) for i in os.listdir(image_folder)]
        for img in images:
            print(img)
            try:
                if os.path.getsize(img) < 100000:
                    os.remove(img)
                else:
                    with open(img, 'r+') as im:
                        img_data = im.read()
            except Exception as e:
                if os.path.exists(img):
                    os.remove(img)
                else:
                    continue

    def clean_image_urls_from_xmls(self):
        cleaned_xml_dir = self.paths['cleaned_xml_dir']
        extracted_path = self.paths['extracted_tars']
        extracted_tar_files = [os.path.join(extracted_path, f) for f in os.listdir(extracted_path)]
        print(os.listdir(extracted_path))
        images = [os.path.join(self.paths['image_dataset'], i) for i in os.listdir(self.paths['image_dataset'])]
        image_files = map(lambda _: _.split('/')[-1].split('.')[0], images)

        cleaned_images_path = self.paths['cleaned_image_dir']
        if not os.path.exists(cleaned_images_path):
            os.makedirs(cleaned_images_path)
        if not os.path.exists(cleaned_xml_dir):
            os.makedirs(cleaned_xml_dir)

        for t_dir in extracted_tar_files:
            cleaned_xml_dir_sub = os.path.join(cleaned_xml_dir, t_dir.split(os.path.sep)[-1])
            if not os.path.exists(cleaned_xml_dir_sub):
                os.makedirs(cleaned_xml_dir_sub)
            print(t_dir)
            t_files = os.listdir(t_dir)
            for tf in t_files:
                print(image_files[0])
                print(tf.split('.')[0])
                filename = tf.split('.')[0]
                if filename in image_files:
                    shutil.copy(images[image_files.index(filename)], cleaned_images_path)
                    shutil.copy(os.path.join(extracted_path, t_dir, filename + '.xml'), cleaned_xml_dir_sub)

    def convert_xml_to_csv(self):
        extracted_tar_files = self.paths['cleaned_xml_dir']
        image_dir = self.paths['cleaned_image_dir']
        xml_to_csv = XML_TO_CSV(self.output_dir, extracted_tar_files, image_dir)
        xml_to_csv.convert_format()

    def generate_tf_records(self):
        csv_path = self.paths['train_test_csv']
        image_dir = self.paths['cleaned_image_dir']
        tf_path = self.paths['tf_records']
        g_tf = GenerateTFRecords(csv_path, image_dir, tf_path, label_map='./label_map.json')
        g_tf.generte_tf_records()
