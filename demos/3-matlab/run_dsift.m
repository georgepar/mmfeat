%
% Example Matlab function for generating dsift/phow
% descriptors using VLFeat.
%
% Call with e.g. run_dsift('/path/to/some/image/dir/')
%
% For DSIFT instead of PHOW, change the vl_phow() call to vl_dsift()
%
function files = run_dsift(directory)
    run('/path/to/vlfeat/toolbox/vl_setup'); % hardcoded path!
    binSize = 10;

    files = dir(fullfile(directory, '*.jpg'));
    for file = files'
        disp(file)
		path = fullfile(directory, file.name);
		newpath = strcat(directory, '/', file.name, '-dsift.mat')

		if exist(newpath)
			disp('File already exists')
			continue
		end

        % PHOW:
        %   im = imread(path);
        %   [frames, descrs] = vl_phow(im2single(im));

        % DSIFT:
        im = imread(path);
        [rows columns numberOfColorChannels] = size(im);
        if numberOfColorChannels > 1
            imgray = rgb2gray(im);
        else
            imgray = im; % It's already gray.
        end
        im = single(imgray);
        [frames, descrs] = vl_dsift(im, 'size', binSize);

        % Save the descriptors
        save(newpath, 'descrs');
    end
end
