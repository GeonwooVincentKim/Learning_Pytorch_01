# Learning_Pytorch_01
Using Anaconda3 Version 1.9.12 and Pycharm 2020.01 to learn how to use Pytorch.


## What does Squeeze for? And How does it work?
- 1. Squeeze is one of command code that decresase or increase Matrix Dimension.
     And Squeeze has an attribute to increase or decrease Matrix Dimension.
     
     - UnSqueeze is increase Matrix Dimension data to Existing Matrix Data.
       For example, if x Variable has 2 Tensor(Two-Dimension Matrix), 
       and users trying to add another Matrix Dimension, 
       it shows Current Tensor and Add another Parameter where the user wants to put it. 
       For example, "x = torch.squeeze(x, 0)".
       And The Dimension Data became 3 Tensor(Three-Dimension Matrix).
       
     - Squeeze is decrease Matrix Dimension data to Existing Matrix Data.
       For example, if x Variable has 3 Tensor(Three-Dimension Matrix),
       and Users trying to release another Matrix Dimension,
       it shows Current Tensor and release it. It doesn't need to use Another Parameter such as 0 or 1 or 2.
       

 - 2. Dimension is one of the Mathematical Conception that is really needs in Pytorch Code. 

## Gradient Descent Methods
- 1. Gradient Descent is a method of grabbing a specific point X, calculating the slope of the cost function, and changing 
     the value of X according to the slope.
     
     - Before studying Gradient Descent Methods, you should know about what cost function meaning and what does cost function
       stands for. The meaning of Cost Function is a function that corresponds to the minimum production cost at a certain output.
       
     - The other meaning of Gradient Descent Methods is Linear Regression.

- 2. Set Gradient Value as 0 by calling zero_grad() function to calculate new Gradient values every epochs.
     - We can get better results by setting Gradient Value as 0.
     
# Further Information

## Further Information for DenseNet
https://pytorch.org/hub/pytorch_vision_densenet/

## How to upload files overs 100MB by using git-lfs?
- 1. git init
- 2. git status
- 3. git lfs install
- 4. git lfs track "File Name or File_Extension"
- 5. git add .
- 6. git status
- 7. git commit -m "Commitment Contents."
- 8. git push origin master

If "git push origin master" doesn't work,
then make sure you have to download this file.

"bfg-1.12.15.jar"

## Uploade files that overs 100MB by Using 'bfg-1.12.15.jar' or 'bfg-1.12.16.jar
- 0. Move bfg-1.12.15.jar to your File Directory which you commit
   on your Github Repository.
- 1. git clone --mirror https://github.com/UserName/Repository_Name.git
- 2. java -jar bfg-1.12.15.jar --strip-blobs-bigger-than 100M Repository_Name.git
- 3. git reflog expire --expire=now --all && git gc --prune=now --aggressive
- 4. git push origin master

If still "git push origin master" doesn't work.
You have to download Sourcetree from Google, Chrome.

## Upload files by using 'Sourcetree'.
- 1. Click on the "Repository" Button at the top.
- 2. You can find Git LFS by scrolling your Mouse or check lists of
   "Repository" Button.
- 3. Click on the "Initialize Repository".

If you cannot click on "Initialize Repository" Button, 
Click the "Trace File" under the "Initialize Repository" Button.

- 1. Click "Add"
- 2. Trace "*.tar.gz" files by click "Confirm".
- 3. Click "Confirm" when this process finished.
- 4. Click "Push" to post your project on your Github Repository.

If still shows the error that you cannot add a files,
Delete "Repository_Name.git" File Directory which is under your Project Folder.
For the example, if your Project Folder name is "01", 
make sure you have to delete your own File Directory, "Repository_Name.git"
under the "01".


## Further information
- 1. https://dobby-the-house-elf.tistory.com/75
- 2. https://hwiyong.tistory.com/318
- 3. https://velog.io/@29been/Github-100MB%EB%B3%B4%EB%8B%A4-%ED%81%B0-%ED%8C%8C%EC%9D%BC-%EC%98%AC%EB%A6%AC%EA%B8%B0
- 4. https://rtyley.github.io/bfg-repo-cleaner/
- 5. https://stackoverflow.com/questions/37049901/how-to-use-jar-tool-bfg-repo-cleaner-and-reduce-git-repository
- 6. https://anaconda.org/conda-forge/bfg
- 7. https://github.com/rtyley/bfg-repo-cleaner/
- 8. https://git-lfs.github.com/

## When you trying to back before commit or push..
- Sourcetree
https://stackoverflow.com/questions/23865335/sourcetree-undo-unpushed-commits/36619186

- Git Bash
https://rocksea.tistory.com/436
https://gmlwjd9405.github.io/2018/05/25/git-add-cancle.html
