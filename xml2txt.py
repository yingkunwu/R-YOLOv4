import xml.etree.ElementTree as ET
import glob
import numpy as np
import os
import argparse


def del_xml(files):
    for file in files:
        if not os.path.isfile(os.path.join(file.split(".")[0] + ".txt")):
            raise AssertionError("You haven't convert xml to txt files yet!")
        os.remove(os.path.join(file.split(".")[0] + ".xml"))
    print("Xml files were deleted.")

def gen_txt(data_dir, files):
    for file in files:
        tree = ET.parse(file)
        root = tree.getroot()
        with open(os.path.join(data_dir, file.split("/")[-1].split(".")[0] + ".txt"), 'w') as f:
            for object in root.findall('object'):
                label = object.find('name').text
                x = object.find('robndbox').find('cx').text
                y = object.find('robndbox').find('cy').text
                w = object.find('robndbox').find('w').text
                h = object.find('robndbox').find('h').text
                a = float(object.find('robndbox').find('angle').text)
                while a > np.pi:
                    a = a - np.pi
                while a <= -np.pi:
                    a = a + np.pi
                line_to_write = x + " " + y + " " + w + " " + h + " " + str(a) + " " + label + '\n'
                f.write(line_to_write)

    print("Finished converting xml to txt files.")

def main(args):
    if len(args.data_folder) == 0:
        raise AssertionError("Please specify the path of your data")

    files = sorted(glob.glob(os.path.join(args.data_folder, "*.xml")))
    if len(files) == 0:
        raise AssertionError("No xml file was found")

    if args.action == "gen_txt":
        gen_txt(args.data_folder, files)
    elif args.action == "del_xml":
        del_xml(files)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, default="", help="path to data folder")
    parser.add_argument("--action", type=str, default="gen_txt", choices=["gen_txt", "del_xml"], help="convert xml to txt files or delete xml files")
    args = parser.parse_args()
    main(args)
