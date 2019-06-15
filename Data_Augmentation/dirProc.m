function dirProc(readDirName,writeDirName,operation,param,writeFmt,writeQF)
% 
% dirProc - This function processes all images in a user specified 'read'
%   directory containing only images and writes the processed images to a
%   user specified 'write' directory.
%
% Written by Matthew C. Stamm
%   v1.0 Released 12/05/17
% 
% INPUTS
%   readDirName -  The name of the directory containing the image files to
%      be processed.  This name should be provided as a string.
%
%   writeDirName -  The name of the directory where the processed image
%       files should be written to.  This name should be provided as a 
%       string.
%
%   operation - A string specifying the operation that the user wishes to
%       perform on all of the images in the 'read' directory.
%
%   param - The parameter associated with the chosen operation.  More
%       detail regarding this input is provided below.
%
%   writeFmt (optional) - The image file format that should be used when 
%       writing the processed images.  The file format specified by the 
%       user should be the same as those available when using the 'imwrite'
%       operation, e.g. 'tif', 'png', 'jpg', etc.  If no format is
%       specified, the TIFF file format is chosen by default.
%
%   writeQF (optional) - The quality factor used during JPEG compression if
%       the write format is chosen as JPEG.
%
%
%   Operations available:
%       'block' - This operation divides each image into blocks.  The block
%           size is specified in the 'param' input variable (e.g. using the
%           value of 512 for 'param' will divide the image into 512 x 512
%           pixel blocks).
%
%       'jpeg' - This option saves each image as a JPEG whose quality
%           factor is specified in the 'param input variable (e.g. using
%           value of 75 for 'param' will JPEG compress each image with a
%           quality factor of 75).
%
%       'resize' - This option resizes each image using the scaling factor
%           specified in the 'param' input variable (e.g. using the value
%           of 1.5 for 'param' scales each image by a factor of 1.5).
%
%       'gamma' - This option contrast enhances each image using the gamma
%           correction operation where the gamma is specified in the
%           'param' input variable (e.g. using the value of 0.7 for 'param'
%           gamma corrects each image with a gamma of 0.7).
%
%       'rename' - This option creates a renamed version of each image,
%           where the renamed version of each image corresponds to a string
%           specified in the 'param' input variable appended with a number
%           assigned to each file (e.g. using the string 'test' for param
%           will produce images named 'test01.tif', 'test02.tif', ...)
%       
%       'suffix' - This option appends a string to the end of the filename
%           (excluding the file extension) of each image where the suffix
%           to be appended is specified in the 'param' input variable (e.g.
%           if the files in the 'read' directory are named 'test01.tif',
%           'test02.tif', test03.tif',...  then using string '_abc' for the
%           param variable will produce files in the 'write directory named
%           'test01_abc.tif', 'test02_abc.tif', 'test03_abc.tif', ...
%           
%
% OUTPUTS - None.  All processed/modified files are written to the 'write'
%        directory.
% 

%%

% NOTE: This function assumes that ONLY image files are in the 'read' 
% directory.  The first two files provided by the 'dir' function are '.' 
% and '..' which are not image files, thus should be skipped.

% check to see if there are enough input arguments
if nargin < 4 
    disp(char(10))
    disp('ERROR: Not enough input arguments are specified');
    disp(char(10));
    return
    
% if the write file type is not specified, set it to 'tif'
elseif nargin <5
    writeFmt= 'tif';
    
end

% check to see if the 'write' directory already exists
if isdir(writeDirName) == 0
    
    % if it does not, then make a new directory to write to
    mkdir(writeDirName);
    
end

% get the name/path of the current directory
currentDirName= cd;


% get information about the directory containing image files to be
% processed
readDir= dir(readDirName);
dirlen= length(readDir);    % get number of files in the directory
% dirlen = min(dirlen,20); %% for debugging purpose

switch operation
    
    case {'block','Block','BLOCK'}
        blocksize= param;
        
        % loop through all files in the directory
        for fnum= 3:dirlen
            
            % read in the current image
            imname= readDir(fnum).name;
            im= imread(strcat(readDirName,'/',imname));
            
            % get the image's size
            [imrows,imcols,colors]= size(im);
            
            
            % loop through all blocks in the image

            % create the filename for this new image block
            [filepath,name,ext]=fileparts(imname);
            writename= strcat(writeDirName,'/',name,'-b','.',writeFmt);
%                     writename= strcat(name,'_',num2str(blocknum),...
%                         '.',writeFmt)                    

            % write the image block to the 'write' directory
            if sum(strcmp(writeFmt,{'jpeg','jpg','JPEG','JPG'}))
                imwrite(midBlock(im),writename,writeFmt,...
                    'Quality',writeQF);
            else
                imwrite(midBlock(im),writename,writeFmt);
            end
            
        end % end loop through images  
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
        
    case {'jpeg','jpg','JPEG','JPG'}
            
        if (nargin > 4) & ...
                ~ sum(strcmp(writeFmt,{'jpeg','jpg','JPEG','JPG'}))
            
            disp(char(10))
            disp(strcat('ERROR: The ''operation'' variable is chosen ',...
                'as ''JPEG'' but the'))
            disp(strcat('       ''writeFmt'' variable specifies a ',...
                ' format other than JPEG.')) 
            disp(strcat('       Please ensure that these variables ',...
                'match or do not specify'))
            disp('       the ''writeFmt'' variable.')
            disp(char(10))
            return
            
        elseif (nargin == 6) & (writeQF ~= param)
            
            disp(char(10))
            disp(strcat('ERROR: JPEG quality factor specified in',...
                ' the ''param'' variable does'))
            disp(strcat('       not match the quality factor specified',...
                ' in the ''writeQF''')) 
            disp(strcat('       variable. Either ensure both match or',...
                ' specify only one (i.e. '))
            disp(strcat('       do not enter a value for the ',...
                '''writeFmt'' and ''writeQF'' variables.'))
            disp(char(10))
            return
        end
        
        writeQF= param;
        
        % loop through all files in the directory
        for fnum= 3:dirlen
            
            % read in the current image
            imname= readDir(fnum).name;
            im= imread(strcat(readDirName,'/',imname));
            
            
            % create the write filename for this image 
            [filepath,name,ext]=fileparts(imname);
            writename= strcat(writeDirName,'/',name,'-j',...
                num2str(writeQF),'.','jpg');
            
            % write the image to the 'write' directory
            imwrite(midBlock(im),writename,'JPEG','Quality',writeQF);
            
        end % end loop through images  
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    
    case {'gamma','Gamma','GAMMA'}
        
        
        g= param;
        
        % create string to append to the end of the filename noting the
        % gamma value used to modify the image
        gammaStr= num2str(g);
        decInd= find(gammaStr == '.');
        fNameEnd= strcat('-g',gammaStr(1:(decInd-1)),'_',...
            gammaStr((decInd+1):end));
        
        
        % loop through all files in the directory
        for fnum= 3:dirlen
            
            % read in the current image
            imname= readDir(fnum).name;
            im= imread(strcat(readDirName,'/',imname));
            
            % gamma correct the image
            gim= 255*((double(im)./255).^g);
            gim= uint8(gim);
            
            % create the filename for this new image block
            [filepath,name,ext]=fileparts(imname);
            writename= strcat(writeDirName,'/',name,fNameEnd,'.',writeFmt);
            
            % write the image to the 'write' directory
            if sum(strcmp(writeFmt,{'jpeg','jpg','JPEG','JPG'}))
                imwrite(midBlock(gim),writename,writeFmt,...
                    'Quality',writeQF);
            else
                imwrite(midBlock(gim),writename,writeFmt);
            end
            
        end % end loop through images  
        
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    case {'resize','Resize','RESIZE'}
        
        scale= param;
        
        % create string to append to the end of the filename noting the
        % scaling factor used to modify the image
        scaleStr= num2str(scale);
        decInd= find(scaleStr == '.');
        fNameEnd= strcat('-r',scaleStr(1:(decInd-1)),'_',...
            scaleStr((decInd+1):end));
        
        
        % loop through all files in the directory
        for fnum= 3:dirlen
            
            % read in the current image
            imname= readDir(fnum).name;
            im= imread(strcat(readDirName,'/',imname));
            
            % resize the image
            resizedIm = imresize(im,scale);
            
            % create the filename for the resized image 
            [filepath,name,ext]=fileparts(imname);
            writename= strcat(writeDirName,'/',name,fNameEnd,'.',writeFmt);
            
            % write the image to the 'write' directory
            if sum(strcmp(writeFmt,{'jpeg','jpg','JPEG','JPG'}))
                imwrite(midBlock(resizedIm),writename,writeFmt,...
                    'Quality',writeQF);
            else
                imwrite(midBlock(resizedIm),writename,writeFmt);
            end
            
        end % end loop through images  
        
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    case {'rename','Rename','RENAME'}
        
        % get new filename (excluding image number)
        newname= param;
        
        % determine how many digits should be used to add the image
        % number to the output (write) filename
        numdig= floor(log10(dirlen-2))+1;
        numdigstr= strcat('%0',num2str(numdig),'d');
        
        % loop through all files in the directory
        for fnum= 3:dirlen
            
            % read in the current image
            imname= readDir(fnum).name;
            im= imread(strcat(readDirName,'/',imname));
            
            
            % create the filename for this new image block
            [filepath,name,ext]=fileparts(imname);
            imnum= fnum-2;
            writename= strcat(writeDirName,'/',newname,...
                sprintf(numdigstr,imnum),'.',writeFmt);
            
            % write the image to the 'write' directory
            if sum(strcmp(writeFmt,{'jpeg','jpg','JPEG','JPG'}))
                imwrite(midBlock(im),writename,writeFmt,...
                    'Quality',writeQF);
            else
                imwrite(midBlock(im),writename,writeFmt);
            end
            
        end % end loop through images  
        
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    case {'suffix','Suffix','SUFFIX'}
        
        % get new filename (excluding image number)
        suffix= param;
        
        % loop through all files in the directory
        for fnum= 3:dirlen
            
            % read in the current image
            imname= readDir(fnum).name;
            im= imread(strcat(readDirName,'/',imname));
            
            
            % create the filename for this new image block
            [filepath,name,ext]=fileparts(imname);
            imnum= fnum-2;
            writename= strcat(writeDirName,'/',name,suffix,'.',writeFmt);
            
            % write the image to the 'write' directory
            if sum(strcmp(writeFmt,{'jpeg','jpg','JPEG','JPG'}))
                imwrite(midBlock(im),writename,writeFmt,...
                    'Quality',writeQF);
            else
                imwrite(midBlock(im),writename,writeFmt);
            end
            
        end % end loop through images  
        
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    otherwise
        disp(char(10))
        disp('ERROR: You have chosen an unknown operation');
        disp(char(10));
        return
end

    function xb = midBlock(x)
        sz = size(x);
        blocksz = 512;
        idx = floor((sz - blocksz)/2) + [1;blocksz];
        xb = x(idx(1,1):idx(2,1), idx(1,2):idx(2,2),:);
    end

end