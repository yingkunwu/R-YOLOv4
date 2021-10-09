wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1sVD2d_y9VDirA-XOdcVDKCDrQw3e7ZJY' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1sVD2d_y9VDirA-XOdcVDKCDrQw3e7ZJY" -O yolov4.pth && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=13LXboG6W7kXWkN7yTeMZ8PKzwcSUZJR2' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=13LXboG6W7kXWkN7yTeMZ8PKzwcSUZJR2" -O UCAS_AOD.zip && rm -rf /tmp/cookies.txt
unzip UCAS_AOD.zip
rm UCAS_AOD.zip
mkdir weights
cd weights
mkdir pretrained
cd ..
mv yolov4.pth weights/pretrained
mv -r UCAS_AOD weights