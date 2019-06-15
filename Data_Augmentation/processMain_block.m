close all;clear;clc
% NAVIGATE IT TO THE DATASET FOLDER (UPON
% RUNNING THE PROGRAM YOU WILL BE PROMPTED
% TO SELECT A FOLDER). e.g. IF 'D:\DATASET'
% FOLDER IS SELECTED, A NEW FOLDER WITH THE
% NAME 'D:\DATASET Processed' WILL BE CREATED
% WITH ALL SUBFOLDERS CONTAING FIRST 50 
% PROCESSED IMAGES

% select folder to process
topFolder = uigetdir(cd,'Select a Folder to Process');
if topFolder == 0
	return
end

destFolder = uigetdir(cd,'Select a Destination Folder');
if destFolder == 0
    error('Select a destination folder location');
end
destFolderName = 'Data (Processed)';
[~,name] = fileparts(destFolder);
if ~strcmp(destFolderName, name)
    destFolder = fullfile(destFolder, destFolderName);
end


disp('running...');

folders = split(genpath(topFolder), pathsep);
folders = folders(1:end-1);

% loop through each folder
for param = [512]
    operation = 'block';
    writeFmt = 'png';
    qf = 100;
    
    for i = 1:length(folders)
        thisFolder = folders{i};
        pattern = fullfile(thisFolder, '*.jpg');
        files = dir(pattern);
        if isempty(files)
            continue
        end

        if floor(param) == param % if isInteger(param)
            loc = sprintf('%s_%d_%s',operation,param,writeFmt);
        else
            loc = sprintf('%s_%0.1f_%s',operation,param,writeFmt);
        end
        loc = replace(loc, '.', '_');
        loc = replace(thisFolder, topFolder, fullfile(destFolder, loc));
        if ~exist(loc,'dir')
            mkdir(loc);
        end
        disp(loc);
        delete(fullfile(thisFolder,'Thumbs.db'));
        dirProc(thisFolder,loc,operation,param,writeFmt,qf);
    end
end
disp('complete.');