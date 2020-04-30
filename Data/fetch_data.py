import cityscapesscripts.download.downloader as dl
import argparse
import os
import zipfile




def parse_arguments():
    description = "Download and organise the Cityscapes Dataset for use with the semantic image inpainting pipeline."
    epilog = "Requires an account that can be created via via https://www.cityscapes-dataset.com/register/"
    parser = argparse.ArgumentParser(description=description, epilog=epilog)
    parser.add_argument('-p', '--preserve')

def main():
    try:
        session = dl.login()
        dl.download_packages(session=session, package_names=["leftImg8bit_trainvaltest.zip", "gtFine_trainvaltest.zip"], destination_path='.')
    except Exception as e:
        print(e.args[0])


if __name__ == "__main__":
    main()

