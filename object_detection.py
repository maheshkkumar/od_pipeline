import argparse

from object_detection_pipeline import ObjectDetectionPipeline


def read_txt_file(input_csv):
    with open(input_csv, 'r') as fp:
        s_ids = map(lambda _:_.strip(), fp.readlines())
    return s_ids

def pipeline(input_file, output_dir):

    synset_ids = read_txt_file(input_file)
    pipeline = ObjectDetectionPipeline(output_dir, synset_ids)

    pipeline.download_tars()
    pipeline.download_images()
    pipeline.clean_dataset()
    pipeline.extract_tars()
    pipeline.clean_image_urls_from_xmls()
    pipeline.convert_xml_to_csv()
    pipeline.generate_tf_records()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i',
                        '--input_file',
                        help="Path of the input file containing the synset ids to train",
                        required=True)

    parser.add_argument('-o',
                        '--output_dir',
                        help="Path of the output directory to store all the necessary files",
                        required=True)

    args = parser.parse_args()

    pipeline(args.input_file, args.output_dir)