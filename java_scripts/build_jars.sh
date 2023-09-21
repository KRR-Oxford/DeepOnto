set -e
cd gateway
gradle distZip
cd build/distributions
yes | unzip gateway.zip
cd ../../../
rm -f mowl/lib/*.jar
cp -r gateway/build/distributions/gateway/lib mowl
