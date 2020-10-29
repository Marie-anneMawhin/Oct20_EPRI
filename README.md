# Oct20_EPRI
For Team EPRI S2DS October 2020.


This project, EPRI, contains 3 sub-projects in this repo.
 * Microstructure Characterization
 * Fracture Toughness
 * Early Stage Fatigue Detection
 
The main objective of this project is to build a system or algorithm which is able to identify failures of steel grades 91 and 92 as well as other unseen steel grades by using non-destructive evaluation (NDE) techniques. The current NDE method is hardness test which lacks meaningful results and measurement quality. It is found that the microstructures of the steel as well as operational conditions play a significant role in failures.

## Microstructure Characterization (Task1)
The first task is to understand responses of different microstructures given NDE methods - Magnetic, Ultrasonics, and Thermo-electric techniques. This will allow us to cluster steels which give NDE responses in the same manner.



## Fracture Toughness
Lorem ipsum



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
