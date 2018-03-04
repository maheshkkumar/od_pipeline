from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import json
import os
import xml.etree.ElementTree as ET
from PIL import Image
from collections import namedtuple

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from object_detection.utils import dataset_util

XMLDetail = namedtuple("XMLDetail", ['filename', 'width', 'height', 'label', 'xmin', 'ymin', 'xmax', 'ymax'])

class GeneratePaths(object):
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def create_directories(self, paths):
        for key, value in paths.iteritems():
            if os.path.exists(value):
                print("{} path exits".format(key))
            else:
                print("Creating {} path".format(key))
                os.makedirs(value)

    def generate(self):
        paths = {}
        paths['tar_files'] = os.path.join(self.output_dir, 'tar_files')
        paths['extracted_tars'] = os.path.join(self.output_dir, 'extracted_tars')
        paths['extract_tars'] = os.path.join(self.output_dir, 'extract_tars')
        paths['image_files'] = os.path.join(self.output_dir, 'image_files')
        paths['image_dataset'] = os.path.join(self.output_dir, 'image_dataset')
        paths['cleaned_xml_dir'] = os.path.join(self.output_dir, 'cleaned_xml_dir')
        paths['cleaned_image_dir'] = os.path.join(self.output_dir, 'cleaned_image_dir')
        paths['train_test_csv'] = os.path.join(self.output_dir, 'train_test_csv')
        paths['tf_records'] = os.path.join(self.output_dir, 'tf_records')
        self.create_directories(paths)
        return paths

class XML_TO_CSV(object):
    def __init__(self, base_dir, xml_dir, image_dir):
        self.base_dir = base_dir
        self.xml_dir = xml_dir
        self.image_dir = image_dir

    def xml_to_csv(self, input_path):

        xmls = []

        xml_files = [os.path.join(input_path, i) for i in os.listdir(input_path)]
        image_list = map(lambda _: os.path.join(self.image_dir, _.split('/')[-1].split('.')[0] + '.jpg'), xml_files)
        xml_list = list(map(lambda _:ET.parse(_).getroot(), xml_files))
        assert len(image_list) == len(xml_list)
        for root, image_path in zip(xml_list, image_list):
            for member in root.findall('object'):
                xmls.append(XMLDetail(image_path.format(root.find('filename').text),
                                      int(root.find('size')[0].text),
                                      int(root.find('size')[1].text),
                                      root.find('folder').text,
                                      int(member[4][0].text),
                                      int(member[4][1].text),
                                      int(member[4][2].text),
                                      int(member[4][3].text)))

        return pd.DataFrame(xmls)

    def convert_format(self):
        output_path = os.path.join(self.base_dir, 'train_test_csv')
        output_csv = os.path.join(self.base_dir, 'output_csv.csv')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print("XML path: {}".format(self.xml_dir))
        xml_folders = [os.path.join(self.xml_dir, i) for i in os.listdir(self.xml_dir)]
        for xml in xml_folders:
            final_df = self.xml_to_csv(xml)
            with open(output_csv, 'a') as f:
                final_df.to_csv(f, header=False, index=False)

        final_df = pd.read_csv(output_csv)
        final_df.columns = XMLDetail._fields
        final_df.rename(columns={'label': 'class'}, inplace=True)
        train, test = train_test_split(final_df, test_size=0.2)
        train.to_csv(os.path.join(output_path, "train.csv"), index=False)
        test.to_csv(os.path.join(output_path, "test.csv"), index=False)
        print("Created train and test XML to CSV convertions.")


class GenerateTFRecords(object):
    def __init__(self, csv_path, image_dir, output_dir, label_map):
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.label_map = json.load(open(label_map), encoding='utf-8')

    def group_annotations(self, annotations, grouping_parameter):
        GroupedData = namedtuple('GroupedData', ['filename', 'object'])
        return [GroupedData(filename, annotations[annotations[grouping_parameter] == filename]) \
                for filename in annotations[grouping_parameter]]

    def create_tf_example(self, group, image_path):
        with tf.gfile.GFile(os.path.join(image_path, '{}'.format(group.filename)), 'rb') as fid:
            encoded_image = fid.read()
        width, height = Image.open(io.BytesIO(encoded_image)).size

        filename = group.filename.encode('utf8')
        image_format = b'jpg'
        xmins, xmaxs, ymins, ymaxs, classes_text, classes = [], [], [], [], [], []

        for index, row in group.object.iterrows():
            xmins.append(row['xmin'] / width)
            xmaxs.append(row['xmax'] / width)
            ymins.append(row['ymin'] / height)
            ymaxs.append(row['ymax'] / height)
            classes_text.append(row['class'].encode('utf8'))
            classes.append(self.label_map[row['class']])

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_image),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
        return tf_example


    def generte_tf_records(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        csv_files = [os.path.join(self.csv_path, _) for _ in os.listdir(self.csv_path) \
                     if _.endswith('.csv')]
        print("CSV File: {}".format(csv_files))
        for csv in csv_files:
            tf_name = csv.split('/')[-1].split('.')[0]
            print("Generating {} TF records".format(tf_name))
            tf_record = os.path.join(self.output_dir, tf_name + '.record')
            writer = tf.python_io.TFRecordWriter(tf_record)
            annotations = pd.read_csv(csv)
            grouped = self.group_annotations(annotations, 'filename')
            for group in grouped:
                tf_example = self.create_tf_example(group, self.image_dir)
                writer.write(tf_example.SerializeToString())

            writer.close()
            print('Successfully created the TFRecords: {}'.format(tf_record))