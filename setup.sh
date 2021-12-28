wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1qi-EWYPGJjZ_CkYh1LatDgfMSfQ0aqhk' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1qi-EWYPGJjZ_CkYh1LatDgfMSfQ0aqhk" -O weights.zip && rm -rf /tmp/cookies.txt
unzip weights.zip
rm weights.zip
rm -rf __MACOSX
