from os import listdir
from os.path import join
from pathlib import PurePath

import cv2
import torch
from numpy import ndarray
from progress.bar import Bar

spectralSaliency = cv2.saliency.StaticSaliencySpectralResidual_create()
fineGrainSaliency = cv2.saliency.StaticSaliencyFineGrained_create()
depth_DPTLarge: str = "DPT_Large"
depth_DPTHybrid: str = "DPT_Hybrid"
depth_MiDaSsmall: str = "MiDaS_small"


def estimateDepth(imagePaths: list, modelType: str, outputFolder: str = "data") -> None:
    midas = torch.hub.load("intel-isl/MiDaS", modelType)
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    if modelType == "DPT_Large" or modelType == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    with Bar(f"Estimating depth with {modelType}...", max=(len(imagePaths))) as bar:
        imagePath: str
        for imagePath in imagePaths:
            imageName: str = (
                PurePath(imagePath).with_suffix("").name
                + f'_{modelType.replace("_", "-")}.jpg'
            )
            outputPath: str = join(outputFolder, imageName)

            image: ndarray = cv2.imread(imagePath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_batch = transform(image).to(device)

            with torch.no_grad():
                prediction = midas(input_batch)

                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=image.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            output = prediction.cpu().numpy()
            cv2.imwrite(outputPath, output)
            bar.next()


def readDirectory(dir: str) -> list:
    files: list = listdir(dir)
    filepaths: list = [join(dir, f) for f in files]
    return filepaths


def computeSpectralSaliency(imagePath: str, outputFolder: str = "data") -> None:
    imageName: str = PurePath(imagePath).with_suffix("").name + "_spectralResidual.jpg"
    outputPath: str = join(outputFolder, imageName)
    image: ndarray = cv2.imread(imagePath)
    (success, saliencyMap) = spectralSaliency.computeSaliency(image)
    saliencyMap: ndarray = (saliencyMap * 255).astype("uint8")
    cv2.imwrite(outputPath, saliencyMap)


def computeFineGrainSaliency(imagePath: str, outputFolder: str = "data") -> None:
    imageName: str = PurePath(imagePath).with_suffix("").name + "_fineGrain.jpg"
    outputPath: str = join(outputFolder, imageName)
    image: ndarray = cv2.imread(imagePath)
    (success, saliencyMap) = fineGrainSaliency.computeSaliency(image)
    saliencyMap: ndarray = (saliencyMap * 255).astype("uint8")
    cv2.imwrite(outputPath, saliencyMap)


def writeImage(image: ndarray, imagePath: str) -> None:
    cv2.imwrite(imagePath, image)


def main() -> None:
    imagePaths: list = readDirectory(dir="pascalVOC/images")
    with Bar(
        "Creating saliency maps of PascalVOC images...", max=len(imagePaths)
    ) as bar:
        imagePath: str
        for imagePath in imagePaths:
            computeSpectralSaliency(imagePath)
            computeFineGrainSaliency(imagePath)
            bar.next()

    estimateDepth(imagePaths, depth_DPTHybrid, "DPT_Hybrid")
    estimateDepth(imagePaths, depth_DPTHybrid, "DPT_Large")
    estimateDepth(imagePaths, depth_DPTHybrid, "DPT_MiDaS_small")


if __name__ == "__main__":
    main()
