filename="digit_five.tar"
file_id="1SyF_CfsfgC3Q1f53dHTCvO1ka7fql8C3"
query=`curl -c ./cookie.txt -s -L "https://drive.google.com/uc?export=download&id=${file_id}" \
| perl -nE'say/uc-download-link.*? href="(.*?)\">/' \
| sed -e 's/amp;//g' | sed -n 2p`

url="https://drive.google.com$query"

curl -b ./cookie.txt -L -o ${filename} $url

tar -xvf digit_five.tar -C Digit/dataset/


filename="OfficeHome.zip"
file_id="0B81rNlvomiwed0V1YUxQdC1uOTg"
query=`curl -c ./cookie.txt -s -L "https://drive.google.com/uc?export=download&id=${file_id}" \
| perl -nE'say/uc-download-link.*? href="(.*?)\">/' \
| sed -e 's/amp;//g' | sed -n 2p`

url="https://drive.google.com$query"

curl -b ./cookie.txt -L -o ${filename} $url

unzip -o -d Office/dataset/OfficeHome/ OfficeHome.zip
mv Office/dataset/OfficeHome/OfficeHomeDataset_10072016/* Office/dataset/OfficeHome/imgs/

filename="Office31.tar"
file_id="12qxRlmjimscPCjLvleYDSU3hRU3GBCfW"
query=`curl -c ./cookie.txt -s -L "https://drive.google.com/uc?export=download&id=${file_id}" \
| perl -nE'say/uc-download-link.*? href="(.*?)\">/' \
| sed -e 's/amp;//g' | sed -n 2p`

url="https://drive.google.com$query"

curl -b ./cookie.txt -L -o ${filename} $url

tar -xvf Office31.tar -C Office/dataset/Office31/

filename="bvlc_model.pth"
file_id="1AOGCX-Jasieeu458E2JpMguh8uEoHfvu"
query=`curl -c ./cookie.txt -s -L "https://drive.google.com/uc?export=download&id=${file_id}" \
| perl -nE'say/uc-download-link.*? href="(.*?)\">/' \
| sed -e 's/amp;//g' | sed -n 2p`

url="https://drive.google.com$query"

curl -b ./cookie.txt -L -o ${filename} $url

mv bvlc_model.pth Office/bvlc_model/




