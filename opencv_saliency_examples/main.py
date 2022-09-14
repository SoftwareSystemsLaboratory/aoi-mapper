from os import listdir
from os.path import join
import cv2
from numpy import ndarray
from progress.bar import Bar
from pathlib import PurePath


spectralSaliency = cv2.saliency.StaticSaliencySpectralResidual_create()
fineGrainSaliency = cv2.saliency.StaticSaliencyFineGrained_create()

def readDirectory(dir: str) ->  list:
    files: list = listdir(dir)
    filepaths: list = [join(dir, f) for f in files]
    return filepaths

def computeSpectralSaliency(imagePath: str, outputFolder: str = "data") ->  None:
    imageName: str = PurePath(imagePath).with_suffix('').name + "_spectralResidual.jpg"
    outputPath: str = join(outputFolder, imageName)
    image: ndarray = cv2.imread(imagePath)
    (success, saliencyMap) = spectralSaliency.computeSaliency(image)
    saliencyMap: ndarray = (saliencyMap * 255).astype("uint8")
    cv2.imwrite(outputPath, saliencyMap)


def computeFineGrainSaliency(imagePath: str, outputFolder: str = "data") -> None: 
    imageName: str = PurePath(imagePath).with_suffix('').name + "_fineGrain.jpg"
    outputPath: str = join(outputFolder, imageName)
    image: ndarray = cv2.imread(imagePath)
    (success, saliencyMap) = fineGrainSaliency.computeSaliency(image)
    saliencyMap: ndarray = (saliencyMap * 255).astype("uint8")
    cv2.imwrite(outputPath, saliencyMap)


def writeImage(image: ndarray, imagePath: str)  ->  None:
    cv2.imwrite(imagePath, image)

def main()  ->  None:
    imagePaths: list = readDirectory(dir="pascalVOC/images")
    with Bar("Creating saliency maps of PascalVOC images...", max=len(imagePaths)) as bar:
        imagePath: str
        for imagePath in imagePaths:
            computeSpectralSaliency(imagePath)
            computeFineGrainSaliency(imagePath)
            bar.next()


if __name__ == "__main__":
    main()
