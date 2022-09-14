from os import listdir
from os.path import join
import cv2
from numpy import ndarray
from progress.bar import Bar
from pathlib import PurePath


spectralSaliency = cv2.saliency.StaticSaliencySpectralResidual_create()

def readDirectory(dir: str) ->  list:
    files: list = listdir(dir)
    filepaths: list = [join(dir, f) for f in files]
    return filepaths

def computeSpectralSaliency(imagePath: str) ->  ndarray:
    image: ndarray = cv2.imread(imagePath)
    (success, saliencyMap) = spectralSaliency.computeSaliency(image)
    saliencyMap: ndarray = (saliencyMap * 255).astype("uint8")
    return saliencyMap
#     cv2.imwrite("test.png", saliencyMap)

def writeImage(image: ndarray, imagePath: str)  ->  None:
    cv2.imwrite(imagePath, image)

def main()  ->  None:
    imagePaths: list = readDirectory(dir="pascalVOC/images")
    with Bar("Creating saliency maps of PascalVOC images...", max=len(imagePaths)) as bar:
        imagePath: str
        for imagePath in imagePaths:
            outputName: str = PurePath(imagePath).name
            outputPath: str = join("data", outputName)
            saliencyMap: ndarray = computeSpectralSaliency(imagePath)
            writeImage(saliencyMap, outputPath)
            bar.next()


if __name__ == "__main__":
    main()
