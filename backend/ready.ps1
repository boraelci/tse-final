Set-Location ..\..\
git clone https://github.com/microsoft/methods2test.git
Set-Location methods2test
git lfs pull
Set-Location ..\..\backend
Expand-7Zip -ArchiveFileName '..\..\methods2test\corpus\raw\fm_fc_co\train.tar.bz2' -TargetPath '..\..\methods2test\corpus\raw\fm_fc_co'
Expand-7Zip -ArchiveFileName '..\..\methods2test\corpus\raw\fm_fc_co\eval.tar.bz2' -TargetPath '..\..\methods2test\corpus\raw\fm_fc_co'
Expand-7Zip -ArchiveFileName '..\..\methods2test\corpus\raw\fm_fc_co\test.tar.bz2' -TargetPath '..\..\methods2test\corpus\raw\fm_fc_co'
Copy-Item -Path '..\..\methods2test\corpus\raw\fm_fc_co\train' -Destination '..\data\raw\' -Recurse
Copy-Item -Path '..\..\methods2test\corpus\raw\fm_fc_co\eval' -Destination '..\data\raw\' -Recurse
Copy-Item -Path '..\..\methods2test\corpus\raw\fm_fc_co\test' -Destination '..\data\raw\' -Recurse