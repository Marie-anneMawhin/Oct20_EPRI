# Oct20_EPRI
For Team EPRI S2DS October 2020.


This project, EPRI, contains 3 sub-projects in this repo.
 * Microstructure Characterization
 * Fracture Toughness
 * Early Stage Fatigue Detection
 
The main objective of this project is to build a system or algorithm which is able to identify failures of steel grades 91 and 92 as well as other unseen steel grades by using non-destructive evaluation (NDE) techniques. The current NDE method is hardness test which lacks meaningful results and measurement quality. It is found that the microstructures of the steel as well as operational conditions play a significant role in failures.

## Microstructure Characterization (Task1)
The first task is to understand responses of different microstructures given NDE methods - Magnetic, Ultrasonics, and Thermo-electric power techniques. This will allow us to cluster steels which give NDE responses in the same manner.
Small size experimental data was given for a total of 24 specimens 16 tubes and 8 pipes. The tube sample was split in two sections 8 known tube MC and 8 unknow tube MC. The pipe sample had missing data for the unknown MC. The project proceeded using the tube sample for modelling and analysis.
Simuluated data was generated to account for measuements uncertainties and better clustering.

The objectives of this subproject was to:
- identify features that have high influence on the clustering of specimens based on their MC
- explore various unsupervised learning models
- consider measurement errors in models
- find consensus between different techniques on clustering groups for the tube specimens

Clustering techniques used:
- Hierarchical Clustering 
- K-Means Clustering
- K-Means Clustering on PCA plots
- Sklearn library exploration


## Fracture Toughness
The purpose of this project is to design a combined approach that uses magnetic, ultrasonic, and thermoelectric power NDE techniques to better predict fracture toughness related to component failure.

The objectives of this subproject was to:
- predict fracture toughness using various NDE measurements and using different models

Technic used:
- Data augmentation with CopulaGAN
- Regression using sklearn library
- GridSearch CV for hyperparameter tuning

## Early Stage Fatigue Detection
Lorem ipsum


## Repository Structure
For each subtask, there are two main folders - `Data` and `NB`. The former one contains data in both raw and processed form. The latter one stores jupyter notebooks which are used for data extractions and data explorations.

# Appendix

## Links to relevant files

[Meeting_Notes](https://docs.google.com/document/d/1_8HSxKLifdZpNco2R6hCWIW6vQlPRSUpxpNqa5SYH8E)

[Schedule](https://docs.google.com/document/d/1Up_pa0ke6wyo4jn19nN48EyxbP5PSi_h/edit)

[Task1_ppt](https://docs.google.com/presentation/d/1QcZ-V8CXSpTUVbvfbHhydaB4qMhmnLku/edit?usp=drive_web&ouid=108700018416396420286&dls=true)

[Taks2_ppt](https://drive.google.com/drive/folders/1trCtQS9SmXAruXdWSbc9mPTWhp05Srtz?ths=true)

[Drive](https://drive.google.com/drive/folders/18NV_jjDFdq_Y-7V8B-2fPNCj60IIbeJs)

## How to update new packages

### > I want to install new package

```
cd PROJECT_ROOT_DIRECTORY
pip install PACKAGE_NAME
pip-chill > requirements.txt
git add requirements.txt
git commit -m "COMMENT"
git pull
git push
```
### > I want to update packages according to other's changes
```
cd PROJECT_ROOT_DIRECTORY
git pull
pip install -r requirements.txt
```
**Note** The commands in capital letters are placeholders.
